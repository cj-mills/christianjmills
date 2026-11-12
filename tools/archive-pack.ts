// Packs the frozen copy of the old site from a commit of `gh-pages`, puts it in the public bucket
// and pins it in legacy/archive.lock.json (tools/lib/archive.ts). Run once; the archive never
// changes after that, since pages leave the copy at build time.
//
//   sh tools/cloudflare.sh node tools/archive-pack.ts <commit> --built YYYY-MM-DD --r2
//                                                           to the R2 bucket, by wrangler
//   DELIVERABLES_PUBLIC_URL=file:///some/dir/ node tools/archive-pack.ts <commit> --built YYYY-MM-DD
//                                                           to a local folder laid out like the bucket
//
// Every file of the commit goes in, in parts small enough for wrangler to upload, each a
// deterministic tar.gz: the files sorted by path and cut where the next would take a part past
// PART_BYTES. Every check that can refuse runs before anything is uploaded or written; the pin is
// written last, once every part is in the bucket.
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buckets } from '../deliverables.config.ts';
import { ROOT, archiveLock, lockPath, packArchive, sha256, unpackArchive, type ArchiveLock } from './lib/archive.ts';

// wrangler's `r2 object put` takes objects up to 315 MB (doc: developers.cloudflare.com/r2/objects/upload-objects/).
const MAX_PART_BYTES = 300 * 1024 * 1024;
// Where a part is cut, before compression. Most of the build's bytes are images and video, which
// gzip barely shrinks, so a part is about this size packed.
const PART_BYTES = 256 * 1024 * 1024;
// Workers static assets serve files up to 25 MiB each.
const MAX_FILE_BYTES = 25 * 1024 * 1024;

const args = process.argv.slice(2);
const toR2 = args.includes('--r2');
const built = args[args.indexOf('--built') + 1];
const commit = args.find((a) => /^[0-9a-f]{7,40}$/.test(a));
if (!commit || !args.includes('--built') || !/^\d{4}-\d{2}-\d{2}$/.test(built ?? '')) {
  console.error('usage: node tools/archive-pack.ts <gh-pages commit> --built YYYY-MM-DD [--r2]');
  process.exit(2);
}
const git = (...a: string[]) => execFileSync('git', a, { cwd: ROOT, maxBuffer: 2 ** 31 - 1 });
const full = git('rev-parse', '--verify', `${commit}^{commit}`).toString().trim();

// The commit's files, read from git's objects: every blob in its tree, by its full path.
const tree = git('ls-tree', '-r', '-z', '--full-tree', full).toString().split('\0').filter(Boolean);
const files = new Map<string, Uint8Array>();
for (const line of tree) {
  const [meta, path] = [line.slice(0, line.indexOf('\t')), line.slice(line.indexOf('\t') + 1)];
  const [mode, type, object] = meta.split(' ');
  if (type !== 'blob' || mode === '120000') throw new Error(`${path} is a ${mode === '120000' ? 'symlink' : type}; only files are archived`);
  const data = new Uint8Array(git('cat-file', 'blob', object));
  if (data.byteLength > MAX_FILE_BYTES) throw new Error(`${path} is ${data.byteLength} bytes, over the 25 MiB a static asset may be`);
  files.set(path, data);
}
if (files.size !== tree.length) throw new Error('some files of the commit were read twice');

// Cut into parts, in path order.
const groups: Map<string, Uint8Array>[] = [new Map()];
let bytes = 0;
for (const path of [...files.keys()].sort()) {
  const data = files.get(path)!;
  if (bytes + data.byteLength > PART_BYTES && groups.at(-1)!.size) {
    groups.push(new Map());
    bytes = 0;
  }
  groups.at(-1)!.set(path, data);
  bytes += data.byteLength;
}
const parts = groups.map((group) => {
  const archive = packArchive(group);
  if (archive.byteLength > MAX_PART_BYTES) throw new Error(`a part is ${archive.byteLength} bytes, over the ${MAX_PART_BYTES} wrangler can upload`);
  // Each part must round-trip to the same files, or its hash pins something other than what was read.
  const back = unpackArchive(archive);
  if (back.size !== group.size || [...group].some(([k, v]) => Buffer.compare(Buffer.from(back.get(k) ?? []), Buffer.from(v)) !== 0)) {
    throw new Error('a part did not round-trip');
  }
  const hash = sha256(archive);
  return { archive, pin: { key: `sha256/${hash}.tar.gz`, sha256: hash, size: archive.byteLength, files: group.size } };
});
const lock: ArchiveLock = archiveLock.parse({
  archiveVersion: 1,
  source: { repo: 'cj-mills/christianjmills', branch: 'gh-pages', commit: full, built },
  bucket: 'public',
  files: files.size,
  parts: parts.map((p) => p.pin),
});

let base: URL | undefined;
if (toR2) {
  if (!process.env.CLOUDFLARE_API_TOKEN) throw new Error('--r2 needs Cloudflare credentials; run it through tools/cloudflare.sh');
} else {
  const url = buckets.public.url;
  if (!url?.startsWith('file:')) throw new Error('the public bucket is not a local stand-in; pack with --r2');
  base = new URL(url);
}
const stored: string[] = [];
for (const { archive, pin } of parts) {
  if (base) {
    const target = new URL(pin.key, base);
    await mkdir(dirname(fileURLToPath(target)), { recursive: true });
    await writeFile(target, archive);
    stored.push(target.href);
    continue;
  }
  const tmp = await mkdtemp(join(tmpdir(), 'archive-'));
  try {
    const file = join(tmp, 'part.tar.gz');
    await writeFile(file, archive);
    const object = `${buckets.public.name}/${pin.key}`;
    execFileSync(
      join(ROOT, 'node_modules/.bin/wrangler'),
      ['r2', 'object', 'put', object, '--remote', '--file', file, '--content-type', 'application/gzip', '--cache-control', 'public, max-age=31536000, immutable'],
      { stdio: 'inherit' },
    );
    stored.push(`r2://${object}`);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
}

await writeFile(lockPath, JSON.stringify(lock, null, 2) + '\n');
console.log(`packed the frozen copy of ${lock.source.branch}@${full.slice(0, 8)}: ${files.size} files in ${parts.length} parts`);
for (const [i, { pin }] of parts.entries()) console.log(`  ${pin.files} files, ${pin.size} bytes, sha256 ${pin.sha256}: ${stored[i]}`);
console.log(`  pinned in ${lockPath}`);
