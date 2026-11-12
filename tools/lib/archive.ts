// The frozen copy of the old site (hub decision 2026-11-03-christianjmills-legacy-urls): the last
// Quarto build, `gh-pages`, kept as one archive in the public R2 bucket and pinned by hash in
// `legacy/archive.lock.json`, so a clean checkout builds without the `gh-pages` branch. It is not a
// deliverable: it is too big for a bundle's limits, and it is served as it was, not rendered.
//
// The archive holds the build as it was served, less the folders of the 71 posts retired on
// 2026-10-09 (`legacy/retired.json`), which redirect to their new homes. It never changes: a page
// leaves the copy when the build leaves it out (tools/archive-merge.ts), not by packing again.
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import { parseTar } from 'nanotar';
import { z } from 'zod';
import { buckets } from '../../deliverables.config.ts';

export const ROOT = fileURLToPath(new URL('../../', import.meta.url));
export const lockPath = join(ROOT, 'legacy/archive.lock.json');
export const retiredPath = join(ROOT, 'legacy/retired.json');
/** Where a fetched archive is kept between builds, by its hash. */
const CACHE = join(ROOT, 'node_modules/.cache/christianjmills-archive');

const hex64 = z.string().regex(/^[0-9a-f]{64}$/);
export const archiveLock = z.strictObject({
  archiveVersion: z.literal(1),
  source: z.strictObject({
    repo: z.string(),
    branch: z.string(),
    commit: z.string().regex(/^[0-9a-f]{40}$/),
    /** The day the build was made, as the banner says it. */
    built: z.iso.date(),
  }),
  /** Folders of the source left out of the archive, each with a trailing slash. */
  exclude: z.array(z.string().regex(/^(?!\/).+\/$/)),
  bucket: z.literal('public'),
  key: z.string().regex(/^sha256\/[0-9a-f]{64}\.tar\.gz$/),
  sha256: hex64,
  size: z.int().positive(),
  files: z.int().positive(),
});
export type ArchiveLock = z.infer<typeof archiveLock>;

export const retiredSchema = z.object({
  posts: z.record(z.string().regex(/^\/posts\/.+\/$/), z.string().regex(/^(https:\/\/|\/)/)),
});

export const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

/**
 * A deterministic tar.gz, as bundles are packed: entries sorted, mtime 0, no owner. 140 of the old
 * build's paths are longer than the 100 bytes a tar header's name holds, and nanotar's writer cuts
 * them off, so a longer name goes first as a GNU long-name record, which nanotar's reader takes.
 */
export function packArchive(files: Map<string, Uint8Array>): Uint8Array {
  const parts: Uint8Array[] = [];
  const entry = (name: string, data: Uint8Array, type = '0') => {
    const header = new Uint8Array(512);
    const put = (s: string, at: number, size: number) => header.set(new TextEncoder().encode(s).subarray(0, size), at);
    put(name, 0, 100);
    put('0000644', 100, 8);
    put('0000000', 108, 8);
    put('0000000', 116, 8);
    put(data.byteLength.toString(8).padStart(11, '0'), 124, 12);
    put('00000000000', 136, 12);
    put('        ', 148, 8);
    put(type, 156, 1);
    put('ustar', 257, 6);
    put('00', 263, 2);
    const sum = header.reduce((a, b) => a + b, 0);
    put(`${sum.toString(8).padStart(6, '0')}\0 `, 148, 8);
    parts.push(header, data, new Uint8Array((512 - (data.byteLength % 512)) % 512));
  };
  for (const name of [...files.keys()].sort()) {
    const bytes = new TextEncoder().encode(name);
    if (bytes.byteLength > 100) entry('././@LongLink', new Uint8Array([...bytes, 0]), 'L');
    entry(name, files.get(name)!);
  }
  parts.push(new Uint8Array(1024));
  return gzipSync(Buffer.concat(parts), { level: 9 });
}

export function unpackArchive(archive: Uint8Array): Map<string, Uint8Array> {
  const files = new Map<string, Uint8Array>();
  for (const entry of parseTar(new Uint8Array(gunzipSync(archive)))) {
    if (entry.type === 'directory') continue;
    if (entry.type !== 'file') throw new Error(`archive entry "${entry.name}" is a ${entry.type}; only files are allowed`);
    if (entry.name.startsWith('/') || entry.name.split('/').some((p) => p === '..' || p === '')) throw new Error(`archive path "${entry.name}" is unsafe`);
    files.set(entry.name, entry.data ?? new Uint8Array());
  }
  return files;
}

export async function readLock(): Promise<ArchiveLock> {
  return archiveLock.parse(JSON.parse(await readFile(lockPath, 'utf8')));
}

/** The archive's bytes, checked against its pin: from the cache, or fetched from the bucket into it. */
export async function fetchArchive(lock: ArchiveLock, log = console.log): Promise<Uint8Array> {
  const cached = join(CACHE, `${lock.sha256}.tar.gz`);
  try {
    const bytes = new Uint8Array(await readFile(cached));
    if (bytes.byteLength === lock.size && sha256(bytes) === lock.sha256) return bytes;
  } catch {}
  const base = buckets[lock.bucket].url;
  if (!base) throw new Error(`no URL configured for the ${lock.bucket} bucket`);
  const url = new URL(lock.key, base);
  const bytes =
    url.protocol === 'file:'
      ? new Uint8Array(await readFile(url))
      : await fetch(url).then(async (res) => {
          if (!res.ok) throw new Error(`GET ${url.href}: ${res.status} ${res.statusText}`);
          return new Uint8Array(await res.arrayBuffer());
        });
  if (bytes.byteLength !== lock.size) throw new Error(`archive is ${bytes.byteLength} bytes; legacy/archive.lock.json pins ${lock.size}`);
  const actual = sha256(bytes);
  if (actual !== lock.sha256) throw new Error(`archive hash ${actual} does not match legacy/archive.lock.json's ${lock.sha256}`);
  await mkdir(CACHE, { recursive: true });
  await writeFile(`${cached}.part`, bytes);
  await rename(`${cached}.part`, cached);
  log(`frozen copy: verified ${url.href}`);
  return bytes;
}

// Quarto's alias pages: a page that sends the reader on with JavaScript, written for each of a
// post's `aliases`. Each becomes a 301 (tools/archive-merge.ts).
const ALIAS = /^<html xmlns="http:\/\/www\.w3\.org\/1999\/xhtml">\s*<head>\s*<title>Redirect<\/title>\s*<script type="text\/javascript">\s*var redirects = (\{.*?\});\n/s;

/** An alias page's target, resolved against the page's own address; null for any other file. */
export function aliasTarget(path: string, bytes: Uint8Array): string | null {
  if (!path.endsWith('.html')) return null;
  const m = ALIAS.exec(new TextDecoder().decode(bytes.subarray(0, 2048)));
  if (!m) return null;
  const redirects: Record<string, string> = JSON.parse(m[1]);
  const keys = Object.keys(redirects);
  // A per-fragment target can't be a server redirect: the fragment never reaches the server.
  if (keys.length !== 1 || keys[0] !== '') throw new Error(`alias page ${path} sends fragments to different pages: ${m[1]}`);
  const url = new URL(redirects[''], `https://archive.invalid/${path}`);
  return url.pathname;
}

/** The addresses a file in the build answers at: a folder's page with and without its slash and by name. */
export function addressesOf(path: string): string[] {
  if (path === 'index.html') return ['/', '/index.html'];
  if (path.endsWith('/index.html')) {
    const dir = `/${path.slice(0, -'index.html'.length)}`;
    return [dir, `${dir}index.html`, dir.slice(0, -1)];
  }
  if (path.endsWith('.html')) return [`/${path}`, `/${path.slice(0, -'.html'.length)}`];
  return [`/${path}`];
}

/** The file an address names, as the old site resolved it: a folder's index, or the file. */
export function fileOf(address: string): string {
  const p = address.replace(/^\//, '');
  return p === '' || p.endsWith('/') ? `${p}index.html` : p;
}

export async function writeFileIn(root: string, path: string, bytes: Uint8Array | string) {
  const out = join(root, path);
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, bytes);
}
