// The pinned deliverables' manifests, read from their bundles as the content loader reads them
// (src/loaders/deliverables.ts): for the build steps that run after Astro, which need each page's
// addresses (tools/archive-merge.ts).
import { readFile } from 'node:fs/promises';
import { sha256, unpack } from '../../spec/bundle.ts';
import { lockfile as lockfileSchema, manifest as manifestSchema, type Manifest } from '../../spec/manifest.ts';
import { buckets, lockfilePath } from '../../deliverables.config.ts';

export async function pinnedManifests(): Promise<Manifest[]> {
  const lock = lockfileSchema.parse(JSON.parse(await readFile(lockfilePath, 'utf8')));
  const out: Manifest[] = [];
  for (const [id, pin] of Object.entries(lock.deliverables)) {
    const base = buckets[pin.bucket].url;
    if (!base) throw new Error(`${id}: no URL configured for the ${pin.bucket} bucket`);
    const url = new URL(pin.key, base);
    const bytes =
      url.protocol === 'file:'
        ? new Uint8Array(await readFile(url))
        : await fetch(url).then(async (res) => {
            if (!res.ok) throw new Error(`GET ${url.href}: ${res.status} ${res.statusText}`);
            return new Uint8Array(await res.arrayBuffer());
          });
    if (bytes.byteLength !== pin.size || sha256(bytes) !== pin.sha256) throw new Error(`${id}: bundle does not match its pin`);
    const raw = unpack(bytes).get('manifest.json');
    if (!raw) throw new Error(`${id}: bundle has no manifest.json`);
    out.push(manifestSchema.parse(JSON.parse(new TextDecoder().decode(raw))));
  }
  return out;
}
