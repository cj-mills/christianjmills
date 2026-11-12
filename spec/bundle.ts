// Deliverable bundles, format 0.1 (draft): how a bundle is laid out, packed, unpacked and verified.
// Shared by the site's content loader and by the publish command projects run.
//
// Layout inside the archive:
//   manifest.json      the manifest (spec/manifest.ts)
//   content/…          the Markdoc entry named by manifest.content.entry
//   assets/…           images, interactive modules and other files the page serves
//
// The archive is a deterministic tar.gz: entries sorted, mtime 0, owner and group cleared, so the
// same bundle contents always hash the same.
import { createHash } from 'node:crypto';
import { gunzipSync, gzipSync } from 'node:zlib';
import { createTar, parseTar } from 'nanotar';
import { createCssVariablesTheme, createHighlighter } from 'shiki';
import { z } from 'zod';
import { BUNDLE_LIMITS, manifest as manifestSchema, type Manifest } from './manifest.ts';
import { LANGUAGES, renderDocument, type Heading, type Note, type Reference } from './dialect.ts';
import { toMarkdown } from './markdown.ts';

const TOP_LEVEL = /^(manifest\.json|content\/.+|assets\/.+)$/;

export function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

export function pack(files: Map<string, Uint8Array>): Uint8Array {
  const entries = [...files.keys()].sort().map((name) => ({
    name,
    data: new Uint8Array(files.get(name)!),
    attrs: { mode: '644', mtime: 0, uid: 0, gid: 0, user: '', group: '' },
  }));
  return gzipSync(createTar(entries), { level: 9 });
}

export function unpack(archive: Uint8Array): Map<string, Uint8Array> {
  if (archive.byteLength > BUNDLE_LIMITS.maxCompressedBytes) throw new Error('bundle exceeds the size limit');
  const files = new Map<string, Uint8Array>();
  // Copy into a fresh array: nanotar reads the underlying ArrayBuffer and ignores a Buffer's byteOffset.
  for (const entry of parseTar(new Uint8Array(gunzipSync(archive)))) {
    if (entry.type === 'directory') continue;
    if (entry.type !== 'file') throw new Error(`bundle entry "${entry.name}" is a ${entry.type}; only files are allowed`);
    checkPath(entry.name);
    const data = entry.data ?? new Uint8Array();
    if (data.byteLength > BUNDLE_LIMITS.maxFileBytes) throw new Error(`bundle file "${entry.name}" exceeds the per-file limit`);
    files.set(entry.name, data);
  }
  if (files.size > BUNDLE_LIMITS.maxFiles) throw new Error('bundle has too many files');
  return files;
}

/** Reads a deliverable's working tree into bundle files: manifest.json, content/ and assets/ only. */
export async function readBundleDir(dir: string): Promise<Map<string, Uint8Array>> {
  const { readFile, readdir } = await import('node:fs/promises');
  const { join, relative, sep } = await import('node:path');
  const files = new Map<string, Uint8Array>();
  files.set('manifest.json', await readFile(join(dir, 'manifest.json')));
  for (const top of ['content', 'assets']) {
    for (const entry of await readdir(join(dir, top), { recursive: true, withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const full = join(entry.parentPath, entry.name);
      const name = relative(dir, full).split(sep).join('/');
      checkPath(name);
      files.set(name, await readFile(full));
    }
  }
  return files;
}

export function checkPath(name: string) {
  if (name.length > 100) throw new Error(`bundle path "${name}" is longer than 100 characters`);
  if (name.startsWith('/') || name.split('/').some((p) => p === '..' || p === '' || p === '.') || !TOP_LEVEL.test(name)) {
    throw new Error(`bundle path "${name}" is outside the bundle layout (manifest.json, content/, assets/)`);
  }
}

const GIF_MAGIC = ['GIF87a', 'GIF89a'];

/**
 * No GIFs in a bundle, by name or by content (hub decision 2026-11-03-site-video-and-large-media):
 * a clip is encoded as AV1 with an H.264 fallback, a still is a PNG. The frozen copy of the old site
 * keeps its GIFs as served; it isn't a bundle.
 */
export function refuseGifs(files: Map<string, Uint8Array>) {
  const gifs = [...files]
    .filter(([name, bytes]) => /\.gif$/i.test(name) || GIF_MAGIC.includes(new TextDecoder().decode(bytes.subarray(0, 6))))
    .map(([name]) => name);
  if (gifs.length) {
    throw new Error(
      `[asset-gif] ${gifs.map((g) => `"${g}"`).join(', ')}: the site takes no GIFs. Encode a clip as AV1 with an H.264 fallback, and a still as PNG`,
    );
  }
}

export interface VerifiedDeliverable {
  manifest: Manifest;
  html: string;
  markdown: string;
  refs: Reference[];
  /** What each numbered marker in the HTML stands for, for the page's references page. */
  notes: Note[];
  headings: Heading[];
  files: Map<string, Uint8Array>;
}

let highlighter: ReturnType<typeof createHighlighter> | undefined;

/** Validates a bundle's manifest and content against the spec and renders it. Throws with every problem found. */
export async function verify(
  files: Map<string, Uint8Array>,
  opts: { assetBase: (m: Manifest) => string; expect?: { id: string; version: string } },
): Promise<VerifiedDeliverable> {
  const raw = files.get('manifest.json');
  if (!raw) throw new Error('bundle has no manifest.json');
  const parsed = manifestSchema.safeParse(JSON.parse(new TextDecoder().decode(raw)));
  if (!parsed.success) throw new Error(`manifest.json does not match format 0.1:\n${z.prettifyError(parsed.error)}`);
  const manifest = parsed.data;
  if (opts.expect && (manifest.id !== opts.expect.id || manifest.version !== opts.expect.version)) {
    throw new Error(
      `bundle holds ${manifest.id}@${manifest.version}, but the lockfile pins ${opts.expect.id}@${opts.expect.version}`,
    );
  }
  const last = manifest.revisions.at(-1)!;
  if (last.version !== manifest.version) throw new Error(`the last revision is ${last.version}, not the manifest's ${manifest.version}`);
  for (const p of manifest.provenance) if (!files.has(p.asset)) throw new Error(`provenance names "${p.asset}", which is not in the bundle`);
  refuseGifs(files);

  const entry = files.get(manifest.content.entry);
  if (!entry) throw new Error(`content entry "${manifest.content.entry}" is not in the bundle`);

  // Code is highlighted into the site's own syntax roles: each token's color is a CSS variable
  // (--shiki-token-keyword and so on) that the site's stylesheet maps to the theme's tokens.
  highlighter ??= createHighlighter({ themes: [createCssVariablesTheme({ name: 'site', fontStyle: true })], langs: LANGUAGES });
  const rendered = renderDocument(new TextDecoder().decode(entry), manifest.content.entry, {
    manifest,
    files,
    assetBase: opts.assetBase(manifest),
    highlighter: await highlighter,
  });
  return { manifest, files, html: rendered.html, refs: rendered.refs, notes: rendered.notes, headings: rendered.headings, markdown: toMarkdown(rendered.tree, manifest) };
}
