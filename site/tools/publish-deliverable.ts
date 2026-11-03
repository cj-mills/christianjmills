// The publish command a content project runs: verify a deliverable's working tree against the
// site's spec, pack it into a deterministic bundle, put it in the bucket, and pin it in the
// site's lockfile.
//
//   node tools/publish-deliverable.ts <deliverable-dir>              to a local stand-in bucket
//   sh tools/cloudflare.sh node tools/publish-deliverable.ts <deliverable-dir> --r2
//                                                                     to the R2 bucket, by wrangler
//
// Keys are content addresses, so uploading the same bundle again writes the same object.
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pack, readBundleDir, sha256, unpack, verify } from '../spec/bundle.ts';
import { FORMAT_VERSION, lockfile as lockfileSchema } from '../spec/manifest.ts';
import { assetBase, buckets, lockfilePath } from '../deliverables.config.ts';

const args = process.argv.slice(2);
const toR2 = args.includes('--r2');
const dir = args.find((a) => !a.startsWith('--'));
if (!dir) {
  console.error('usage: node tools/publish-deliverable.ts <deliverable-dir> [--r2]');
  process.exit(2);
}

const files = await readBundleDir(dir);
const { manifest } = await verify(files, { assetBase: (m) => assetBase(m.id, m.version) });
const archive = pack(files);
const hash = sha256(archive);

// The archive must round-trip to the same files, or the hash pins something other than what was verified.
const back = unpack(archive);
if (back.size !== files.size || [...files].some(([k, v]) => Buffer.compare(Buffer.from(back.get(k) ?? []), Buffer.from(v)) !== 0)) {
  throw new Error('bundle did not round-trip');
}

const bucket = manifest.tier.access === 'free' ? 'public' : 'private';
const key = `sha256/${hash}.tar.gz`;
let stored: string;
if (toR2) {
  if (!process.env.CLOUDFLARE_API_TOKEN) throw new Error('--r2 needs Cloudflare credentials; run it through tools/cloudflare.sh');
  const tmp = await mkdtemp(join(tmpdir(), 'deliverable-'));
  try {
    const file = join(tmp, 'bundle.tar.gz');
    await writeFile(file, archive);
    const wrangler = fileURLToPath(new URL('../node_modules/.bin/wrangler', import.meta.url));
    const object = `${buckets[bucket].name}/${key}`;
    execFileSync(
      wrangler,
      ['r2', 'object', 'put', object, '--remote', '--file', file, '--content-type', 'application/gzip', '--cache-control', 'public, max-age=31536000, immutable'],
      { stdio: 'inherit' },
    );
    stored = `r2://${object}`;
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
} else {
  const base = buckets[bucket].url;
  if (!base?.startsWith('file:')) throw new Error(`the ${bucket} bucket is not a local stand-in; publish with --r2`);
  const target = new URL(key, base);
  await mkdir(dirname(fileURLToPath(target)), { recursive: true });
  await writeFile(target, archive);
  stored = target.href;
}

const lock = lockfileSchema.parse(JSON.parse(await readFile(lockfilePath, 'utf8').catch(() => '{"lockfileVersion":1,"deliverables":{}}')));
const prior = lock.deliverables[manifest.id];
if (prior && prior.version === manifest.version && prior.sha256 !== hash) {
  throw new Error(`${manifest.id}@${manifest.version} is already pinned with different contents; bump the version and add a revision note`);
}
lock.deliverables[manifest.id] = { version: manifest.version, format: FORMAT_VERSION, bucket, key, sha256: hash, size: archive.byteLength };
lock.deliverables = Object.fromEntries(Object.entries(lock.deliverables).sort(([a], [b]) => a.localeCompare(b)));
await writeFile(lockfilePath, JSON.stringify(lock, null, 2) + '\n');

console.log(`published ${manifest.id}@${manifest.version}`);
console.log(`  ${files.size} files, ${archive.byteLength} bytes, sha256 ${hash}`);
console.log(`  ${bucket} bucket: ${stored}`);
console.log(`  pinned in ${fileURLToPath(lockfilePath)}`);
