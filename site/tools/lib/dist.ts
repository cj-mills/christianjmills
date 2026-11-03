// Helpers shared by the build checks: list the built pages and serve dist/ over HTTP.
import { createReadStream } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import { createServer, type Server } from 'node:http';
import { extname, join, normalize, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

export const DIST = fileURLToPath(new URL('../../dist/', import.meta.url));

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.pagefind': 'application/octet-stream',
  '.pf_meta': 'application/octet-stream',
  '.pf_fragment': 'application/octet-stream',
  '.pf_index': 'application/octet-stream',
};

/** Site paths of every built HTML page, e.g. "/", "/search/". Pagefind's own files are excluded. */
export async function pages(): Promise<string[]> {
  const out: string[] = [];
  for (const entry of await readdir(DIST, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || entry.name !== 'index.html') continue;
    const rel = relative(DIST, entry.parentPath).split(sep).join('/');
    if (rel.startsWith('pagefind')) continue;
    out.push(rel ? `/${rel}/` : '/');
  }
  return out.sort();
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
