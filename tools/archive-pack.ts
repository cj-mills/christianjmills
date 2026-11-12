// Packs the frozen copy of the old site from a commit of `gh-pages`, puts it in the public bucket
// and pins it in legacy/archive.lock.json (tools/lib/archive.ts). Run once; the archive never
// changes after that, since pages leave the copy at build time.
//
//   sh tools/cloudflare.sh node tools/archive-pack.ts <commit> --built YYYY-MM-DD --r2
//                                                           to the R2 bucket, by wrangler
//   DELIVERABLES_PUBLIC_URL=file:///some/dir/ node tools/archive-pack.ts <commit> --built YYYY-MM-DD
//                                                           to a local folder laid out like the bucket
//
// Every check that can refuse runs before anything is uploaded or written; the pin is written last.
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buckets } from '../deliverables.config.ts';
import { ROOT, archiveLock, lockPath, packArchive, retiredPath, retiredSchema, sha256, unpackArchive, type ArchiveLock } from './lib/archive.ts';

// wrangler's `r2 object put` takes objects up to 315 MB (doc: developers.cloudflare.com/r2/objects/upload-objects/).
const MAX_ARCHIVE_BYTES = 300 * 1024 * 1024;
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

const retired = retiredSchema.parse(JSON.parse(await readFile(retiredPath, 'utf8')));
const exclude = Object.keys(retired.posts).map((p) => p.slice(1)).sort();

// The commit's files, read from git's objects: every blob in its tree, by its full path.
const tree = git('ls-tree', '-r', '-z', '--full-tree', full).toString().split('\0').filter(Boolean);
const blobs = tree.map((line) => {
  const [meta, path] = [line.slice(0, line.indexOf('\t')), line.slice(line.indexOf('\t') + 1)];
  const [mode, type, object] = meta.split(' ');
  if (type !== 'blob' || mode === '120000') throw new Error(`${path} is a ${mode === '120000' ? 'symlink' : type}; only files are archived`);
  return { path, object };
});
const files = new Map<string, Uint8Array>();
const left: string[] = [];
for (const { path, object } of blobs) {
  if (exclude.some((dir) => path.startsWith(dir))) {
    left.push(path);
    continue;
  }
  const data = new Uint8Array(git('cat-file', 'blob', object));
  if (data.byteLength > MAX_FILE_BYTES) throw new Error(`${path} is ${data.byteLength} bytes, over the 25 MiB a static asset may be`);
  files.set(path, data);
}
if (files.size + left.length !== blobs.length) throw new Error('some files of the commit were read twice or not at all');
for (const dir of exclude) if (!left.some((f) => f.startsWith(dir))) throw new Error(`legacy/retired.json names ${dir}, which ${full.slice(0, 8)} doesn't have`);

const archive = packArchive(files);
if (archive.byteLength > MAX_ARCHIVE_BYTES) throw new Error(`the archive is ${archive.byteLength} bytes, over the ${MAX_ARCHIVE_BYTES} wrangler can upload`);
const back = unpackArchive(archive);
if (back.size !== files.size || [...files].some(([k, v]) => Buffer.compare(Buffer.from(back.get(k) ?? []), Buffer.from(v)) !== 0)) {
  throw new Error('archive did not round-trip');
}
const hash = sha256(archive);
const key = `sha256/${hash}.tar.gz`;
const lock: ArchiveLock = archiveLock.parse({
  archiveVersion: 1,
  source: { repo: 'cj-mills/christianjmills', branch: 'gh-pages', commit: full, built },
  exclude,
  bucket: 'public',
  key,
  sha256: hash,
  size: archive.byteLength,
  files: files.size,
});

let stored: string;
if (toR2) {
  if (!process.env.CLOUDFLARE_API_TOKEN) throw new Error('--r2 needs Cloudflare credentials; run it through tools/cloudflare.sh');
  const tmp = await mkdtemp(join(tmpdir(), 'archive-'));
  try {
    const file = join(tmp, 'archive.tar.gz');
    await writeFile(file, archive);
    const object = `${buckets.public.name}/${key}`;
    execFileSync(
      join(ROOT, 'node_modules/.bin/wrangler'),
      ['r2', 'object', 'put', object, '--remote', '--file', file, '--content-type', 'application/gzip', '--cache-control', 'public, max-age=31536000, immutable'],
      { stdio: 'inherit' },
    );
    stored = `r2://${object}`;
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
} else {
  const base = buckets.public.url;
  if (!base?.startsWith('file:')) throw new Error('the public bucket is not a local stand-in; pack with --r2');
  const target = new URL(key, base);
  await mkdir(dirname(fileURLToPath(target)), { recursive: true });
  await writeFile(target, archive);
  stored = target.href;
}

await writeFile(lockPath, JSON.stringify(lock, null, 2) + '\n');
console.log(`packed the frozen copy of ${lock.source.branch}@${full.slice(0, 8)}`);
console.log(`  ${files.size} files, ${archive.byteLength} bytes, sha256 ${hash}`);
console.log(`  left out: ${left.length} files in the ${exclude.length} retired posts' folders`);
console.log(`  public bucket: ${stored}`);
console.log(`  pinned in ${lockPath}`);
