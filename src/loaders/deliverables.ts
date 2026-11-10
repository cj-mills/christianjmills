// Content loader for deliverables: reads the lockfile, fetches each pinned bundle, checks its size
// and SHA-256, validates the manifest and the Markdoc content, and stores the rendered page.
// Any failure throws, which fails the build.
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Loader } from 'astro/loaders';
import { readBundleDir, sha256, unpack, verify } from '../../spec/bundle.ts';
import { checkBundle } from '../../spec/english.ts';
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

      // The fixture build (npm run check:fixtures) renders the format's own fixture pages and
      // nothing else, so the site's checks can run on every part of the dialect.
      const fixtures = process.env.SITE_FIXTURES === '1';
      const fixturesDir = new URL('../../spec/fixtures/', import.meta.url);
      const entries: [string, (typeof lock.deliverables)[string] | undefined][] = fixtures
        ? (await readdir(fixturesDir, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => [`fixture-${e.name}`, undefined])
        : Object.entries(lock.deliverables);

      for (const [id, pin] of entries) {
        let files: Map<string, Uint8Array>;
        let digest: string;
        const local = pin ? overrides[id] : new URL(`${id.replace(/^fixture-/, '')}/`, fixturesDir).href;
        if (local) {
          const dir = fileURLToPath(new URL(local, localOverridesPath));
          if (pin) logger.warn(`${id}: using the local working tree ${dir} instead of the pinned bundle (preview only)`);
          files = await readBundleDir(dir);
          digest = sha256(Buffer.concat([...files].flatMap(([k, v]) => [Buffer.from(k), v])));
          watcher?.add(dir);
        } else if (pin) {
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
        } else continue;

        const d = await verify(files, {
          assetBase: (m) => assetBase(m.id, m.version),
          expect: pin && !local ? { id, version: pin.version } : undefined,
        });

        // A working tree is told now what the publish command will refuse. A pinned bundle was
        // checked when it was published, or predates the check.
        if (local) for (const problem of checkBundle(files, d.manifest)) logger.warn(`${id}: ${problem}`);

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
            pinned: pin && !local ? { sha256: pin.sha256, key: pin.key } : null,
            refs: d.refs,
            notes: d.notes,
            markdown: d.markdown,
          },
        });
        store.set({ id, data, digest, rendered: { html: d.html, metadata: { headings: d.headings } } });
      }
    },
  };
}
