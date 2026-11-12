// Helpers shared by the build checks: list the built pages and serve dist/ over HTTP.
import { createReadStream } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import { createServer, type Server } from 'node:http';
import { extname, join, normalize, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

// The build to check: dist/, or the fixture build's folder (npm run check:fixtures).
export const DIST = fileURLToPath(new URL(`../../${process.env.SITE_DIST ?? 'dist'}/`, import.meta.url));
/** Interactives that are meant to fail to mount: only the fixture's, which shows that a fallback stays. */
export const EXPECTED_MOUNT_FAILURES = (process.env.EXPECT_MOUNT_FAILURES ?? '').split(',').filter(Boolean);
/** Interactives that are meant to move the page when they mount: only the fixture's, which shows that the check sees it. */
export const EXPECTED_MOUNT_SHIFTS = (process.env.EXPECT_MOUNT_SHIFTS ?? '').split(',').filter(Boolean);

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.webmanifest': 'application/manifest+json',
  '.wasm': 'application/wasm',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.pagefind': 'application/octet-stream',
  '.pf_meta': 'application/octet-stream',
  '.pf_fragment': 'application/octet-stream',
  '.pf_index': 'application/octet-stream',
};

/**
 * The files the frozen copy of the old site put into dist/ (tools/archive-merge.ts writes the list,
 * and build:pages clears it with dist/). The checks are the new site's, so its pages alone.
 */
export const MERGED = fileURLToPath(new URL('../../.build/merged.json', import.meta.url));

/** Site paths of every built HTML page, e.g. "/", "/search/", "/404.html". Pagefind's own files and the old site's are excluded. */
export async function pages(): Promise<string[]> {
  const merged = new Set<string>(await readFile(MERGED, 'utf8').then(JSON.parse, () => []));
  const out: string[] = [];
  for (const entry of await readdir(DIST, { recursive: true, withFileTypes: true })) {
    if (entry.isFile() && entry.name === '404.html' && entry.parentPath.replace(/\/$/, '') === DIST.replace(/\/$/, '')) out.push('/404.html');
    if (!entry.isFile() || entry.name !== 'index.html') continue;
    const rel = relative(DIST, entry.parentPath).split(sep).join('/');
    if (rel.startsWith('pagefind') || merged.has(rel ? `${rel}/index.html` : 'index.html')) continue;
    out.push(rel ? `/${rel}/` : '/');
  }
  // CHECK_PAGES narrows a run to some pages, e.g. the fixture build's one page.
  const only = (process.env.CHECK_PAGES ?? '').split(',').filter(Boolean);
  return out.filter((p) => !only.length || only.includes(p)).sort();
}

export function fileFor(urlPath: string): string {
  const clean = normalize(decodeURIComponent(urlPath.split('?')[0])).replace(/^(\.\.[/\\])+/, '');
  const p = join(DIST, clean);
  return p.endsWith(sep) || urlPath.endsWith('/') ? join(p, 'index.html') : p;
}

export async function serve(): Promise<{ origin: string; server: Server }> {
  const server = createServer(async (req, res) => {
    const file = fileFor(req.url ?? '/');
    try {
      if (!(await stat(file)).isFile()) throw new Error();
      res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' });
      createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as { port: number };
  return { origin: `http://127.0.0.1:${port}`, server };
}
