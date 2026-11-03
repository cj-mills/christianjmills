// The publish command a content project runs: verify a deliverable's working tree against the
// site's spec, pack it into a deterministic bundle, put it in the bucket, and pin it in the
// site's lockfile.
//
//   node tools/publish-deliverable.ts <deliverable-dir>
//
// PROTOTYPE: the bucket is the local stand-in folder named in deliverables.config.ts. Uploading to
// R2 waits for a bucket and credentials the user creates.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pack, readBundleDir, sha256, unpack, verify } from '../spec/bundle.ts';
import { FORMAT_VERSION, lockfile as lockfileSchema } from '../spec/manifest.ts';
import { assetBase, buckets, lockfilePath } from '../deliverables.config.ts';

const dir = process.argv[2];
if (!dir) {
  console.error('usage: node tools/publish-deliverable.ts <deliverable-dir>');
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
const base = buckets[bucket];
if (!base?.startsWith('file:')) throw new Error(`no local stand-in configured for the ${bucket} bucket; R2 upload is not implemented yet`);
const key = `sha256/${hash}.tar.gz`;
const target = new URL(key, base);
await mkdir(dirname(fileURLToPath(target)), { recursive: true });
await writeFile(target, archive);

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
console.log(`  ${bucket} bucket: ${target.href}`);
console.log(`  pinned in ${fileURLToPath(lockfilePath)}`);
