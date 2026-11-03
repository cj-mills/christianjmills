// Content loader for deliverables: reads the lockfile, fetches each pinned bundle, checks its size
// and SHA-256, validates the manifest and the Markdoc content, and stores the rendered page.
// Any failure throws, which fails the build.
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Loader } from 'astro/loaders';
import { readBundleDir, sha256, unpack, verify } from '../../spec/bundle.ts';
import { lockfile as lockfileSchema } from '../../spec/manifest.ts';
import { assetBase, buckets, localOverridesPath, lockfilePath } from '../../deliverables.config.ts';

async function fetchBytes(url: URL): Promise<Uint8Array> {
  if (url.protocol === 'file:') return new Uint8Array(await readFile(url));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url.href}: ${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function readOverrides(): Promise<Record<string, string>> {
  if (process.env.DELIVERABLES_ALLOW_LOCAL !== '1') return {};
  try {
    return JSON.parse(await readFile(localOverridesPath, 'utf8'));
  } catch {
    return {};
  }
}

export function deliverablesLoader(): Loader {
  return {
    name: 'deliverables',
    async load({ store, logger, parseData, config, watcher }) {
      const lock = lockfileSchema.parse(JSON.parse(await readFile(lockfilePath, 'utf8')));
      const overrides = await readOverrides();
      const servedRoot = join(fileURLToPath(config.publicDir), '_deliverables');
      await rm(servedRoot, { recursive: true, force: true });
      store.clear();

      for (const [id, pin] of Object.entries(lock.deliverables)) {
        let files: Map<string, Uint8Array>;
        let digest: string;
        const local = overrides[id];
        if (local) {
          const dir = fileURLToPath(new URL(local, localOverridesPath));
          logger.warn(`${id}: using the local working tree ${dir} instead of the pinned bundle (preview only)`);
          files = await readBundleDir(dir);
          digest = sha256(Buffer.concat([...files].flatMap(([k, v]) => [Buffer.from(k), v])));
          watcher?.add(dir);
        } else {
          const base = buckets[pin.bucket].url;
          if (!base) throw new Error(`${id}: no URL configured for the ${pin.bucket} bucket`);
          const url = new URL(pin.key, base);
          const archive = await fetchBytes(url);
          if (archive.byteLength !== pin.size) throw new Error(`${id}: bundle is ${archive.byteLength} bytes; the lockfile pins ${pin.size}`);
          const actual = sha256(archive);
          if (actual !== pin.sha256) throw new Error(`${id}: bundle hash ${actual} does not match the lockfile's ${pin.sha256}`);
          files = unpack(archive);
          digest = actual;
          logger.info(`${id}@${pin.version}: verified ${url.href}`);
        }

        const d = await verify(files, {
          assetBase: (m) => assetBase(m.id, m.version),
          expect: local ? undefined : { id, version: pin.version },
        });

        // The bundle's files are served as static files under the asset base.
        const servedDir = join(servedRoot, d.manifest.id, d.manifest.version);
        for (const [name, bytes] of files) {
          if (!name.startsWith('assets/')) continue;
          const out = join(servedDir, name);
          await mkdir(dirname(out), { recursive: true });
          await writeFile(out, bytes);
        }

        const data = await parseData({
          id,
          data: {
            ...d.manifest,
            pinned: local ? null : { sha256: pin.sha256, key: pin.key },
            refs: d.refs,
            markdown: d.markdown,
          },
        });
        store.set({ id, data, digest, rendered: { html: d.html, metadata: { headings: d.headings } } });
      }
    },
  };
}
