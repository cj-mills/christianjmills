// How the frozen copy and the new site share the addresses (tools/archive-merge.ts, which does the
// reading and writing): which files of the copy are served, with the banner; what leaves it; every
// redirect; and what each address must answer. A function of its inputs alone, so
// tools/check-legacy.ts can try it on cases the real site doesn't have yet.
import { ARCHIVE_SITEMAP_PATH, sitemap } from '../../src/lib/lists.ts';
import { addressesOf, aliasTarget, fileOf } from './archive.ts';

const SITE = new URL('https://christianjmills.com/');
/** Root files of the old build whose addresses the new site serves itself. */
export const REPLACED = new Set(['index.html', '404.html', 'blog.xml', 'sitemap.xml', 'robots.txt']);
/** Files that were GitHub Pages' settings, not the site's. */
const DROPPED = new Set(['CNAME', '.nojekyll']);
// Workers static assets: a _redirects file holds up to 2,000 static and 100 dynamic rules
// (doc: developers.cloudflare.com/workers/static-assets/redirects/).
const LIMITS = { static: 2000, dynamic: 100 };

export interface Expectation {
  path: string;
  /** 200: the address answers itself. A string: it answers 301 to that address, which must end at a 200. */
  expect: 200 | string;
  why: string;
}

export interface Plan {
  /** Files of the copy to write into dist/, by path; pages with the banner. */
  writes: Map<string, Uint8Array | string>;
  /** Files of the copy that left it, and why. */
  leaving: Map<string, string>;
  redirects: Map<string, { to: string; why: string }>;
  /** dist/_redirects. */
  rules: string;
  /** dist/sitemap-archive.xml. */
  archiveSitemap: string;
  expectations: Expectation[];
  stats: { pages: number; rules: number; oldSitemap: number; stillInSitemap: number };
}

export function planMerge(input: {
  archive: Map<string, Uint8Array>;
  /** The day the old build was made, YYYY-MM-DD. */
  builtOn: string;
  retired: Record<string, string>;
  manifests: { id: string; urls: { canonical: string; supersedes: string[] } }[];
  /** Every file the new site built into dist/. */
  built: Set<string>;
}): Plan {
  const { archive, builtOn, retired, manifests, built } = input;
  if (built.has('_redirects')) throw new Error('dist/ already has a _redirects file; the site writes it here, and only here');
  // The 301s, by source address, as the old site named it (a rule is written with its spaces encoded).
  const redirects = new Map<string, { to: string; why: string }>();
  const redirect = (from: string, to: string, why: string) => {
    const had = redirects.get(from);
    if (had && had.to !== to) throw new Error(`${from} would redirect both to ${had.to} (${had.why}) and to ${to} (${why})`);
    redirects.set(from, { to, why });
  };
  const leaving = new Map<string, string>();

  for (const [path, bytes] of archive) {
    const target = aliasTarget(path, bytes);
    if (target === null) continue;
    leaving.set(path, 'an alias page');
    for (const a of addressesOf(path)) redirect(a, target, `alias page ${path}`);
  }
  // A retired post's page leaves the copy and sends the reader to its new home; its images and videos
// stay at their addresses, as every other file of the old build does.
for (const [dir, target] of Object.entries(retired)) {
  if (!archive.has(fileOf(dir))) throw new Error(`legacy/retired.json names ${dir}, which the frozen copy has no page for`);
  leaving.set(fileOf(dir), 'retired');
  for (const a of addressesOf(fileOf(dir))) redirect(a, target, `retired post ${dir}`);
}
  for (const m of manifests) {
    const own = fileOf(m.urls.canonical);
    if (archive.has(own)) leaving.set(own, `taken over by ${m.id}`);
    for (const old of m.urls.supersedes) {
      const file = fileOf(old);
      if (archive.has(file)) leaving.set(file, `superseded by ${m.id}`);
      for (const a of addressesOf(file)) redirect(a, m.urls.canonical, `superseded by ${m.id}`);
    }
  }

  // A chain of redirects ends where its last one points: the reader gets one hop.
  const resolve = (to: string, seen = new Set<string>()): string => {
    const next = to.startsWith('/') ? redirects.get(to) : undefined;
    if (!next) return to;
    if (seen.has(to)) throw new Error(`redirect loop through ${[...seen].join(' → ')}`);
    return resolve(next.to, seen.add(to));
  };
  for (const [from, r] of redirects) redirects.set(from, { ...r, to: resolve(r.to) });

  // The copy goes in, around the new site.
  const BANNER =
    `<div role="note" data-archived style="margin:0 0 1.5rem;padding:.75rem 1rem;border-radius:4px;background:#1f2328;color:#f6f8fa;font:15px/1.45 system-ui,sans-serif">` +
    `This page is from the previous version of this site, archived as it was on ${new Date(`${builtOn}T00:00:00Z`).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' })}. ` +
    `It keeps its old design and is no longer updated. <a href="/" style="color:inherit;text-decoration:underline">Go to the current site</a>.</div>`;
  // The copy, around the new site: what to write, once everything below has passed.
  const collisions: string[] = [];
  const writes = new Map<string, Uint8Array | string>();
  for (const [path, bytes] of archive) {
    if (DROPPED.has(path) || leaving.has(path)) continue;
    if (built.has(path)) {
      if (!REPLACED.has(path)) collisions.push(path);
      continue;
    }
    let out: Uint8Array | string = bytes;
    if (path.endsWith('.html')) {
      const html = new TextDecoder().decode(bytes);
      const main = /<main\b[^>]*>/.exec(html);
      if (!main) throw new Error(`${path}: an old page with no <main>, so no place for the archive banner`);
      out = html.slice(0, main.index + main[0].length) + BANNER + html.slice(main.index + main[0].length);
    }
    writes.set(path, out);
  }
  for (const r of REPLACED) if (!built.has(r)) throw new Error(`the new site should serve /${r} in place of the old site's, and doesn't`);
  if (collisions.length) {
    throw new Error(
      `the new site builds files the frozen copy also has. A deliverable that takes over an old page names it in urls.canonical or urls.supersedes:\n  ${collisions.join('\n  ')}`,
    );
  }
  const served = [...writes.keys()];
  const pages = served.filter((p) => p.endsWith('.html')).length;
  const final = new Set([...built, ...served]);

  // A folder's page answers at its slash; the address without it is sent there, as GitHub Pages did.
  // An old page named .html also answers without the extension there, so it is sent to its name.
  for (const f of final) {
    if (f.endsWith('/index.html')) {
      const dir = `/${f.slice(0, -'index.html'.length)}`;
      if (!redirects.has(dir.slice(0, -1))) redirects.set(dir.slice(0, -1), { to: dir, why: 'the slash' });
    } else if (f.endsWith('.html') && f !== '404.html' && served.includes(f)) {
      const bare = `/${f.slice(0, -'.html'.length)}`;
      if (!redirects.has(bare) && !final.has(`${bare.slice(1)}/index.html`)) redirects.set(bare, { to: `/${f}`, why: 'the old .html name' });
    }
  }
  // Every address a redirect lands on in the site must be served.
  for (const [from, { to }] of redirects) {
    if (to.startsWith('/') && !final.has(fileOf(to))) throw new Error(`${from} redirects to ${to}, which the site doesn't serve`);
    if (to.startsWith('/') && final.has(fileOf(from)) && !leaving.has(fileOf(from)) && !from.endsWith('/') && fileOf(from) !== fileOf(to)) {
      throw new Error(`${from} is a served file and also redirects to ${to}`);
    }
  }

  const rules = [...redirects].sort(([a], [b]) => a.localeCompare(b)).map(([from, { to }]) => `${encodeURI(from)} ${to.startsWith('/') ? encodeURI(to) : to} 301`);
  const rewrites = ['/ /index.html 200', '/*/ /:splat/index.html 200'];
  const dynamic = [...rules, ...rewrites].filter((r) => r.includes('*') || r.includes('/:')).length;
  if (rules.length + rewrites.length - dynamic > LIMITS.static || dynamic > LIMITS.dynamic) {
    throw new Error(`_redirects has ${rules.length + rewrites.length - dynamic} static and ${dynamic} dynamic rules, over Cloudflare's ${LIMITS.static} and ${LIMITS.dynamic}`);
  }
  const rulesFile = [
    '# Written by tools/archive-merge.ts; never edited by hand. Redirects run before files are served.',
    ...rules,
    '# A folder is served by its index.html (wrangler.jsonc serves files only by their names).',
    ...rewrites,
    '',
  ].join('\n');

  // The old sitemap's pages that the copy still serves keep their place in search.
  const oldSitemap = new TextDecoder().decode(archive.get('sitemap.xml') ?? new Uint8Array());
  const oldEntries = [...oldSitemap.matchAll(/<url>\s*<loc>([^<]+)<\/loc>\s*(?:<lastmod>([^<]+)<\/lastmod>)?/g)].map((m) => ({
    path: new URL(m[1]).pathname,
    lastmod: m[2]?.slice(0, 10),
  }));
  const stillServed = oldEntries.filter((e) => writes.has(fileOf(e.path)));

  // What the URL check asks of the served site.
  const expectations: Expectation[] = [];
  const seen = new Set<string>();
  const want = (path: string, expect: 200 | string, why: string) => {
    if (seen.has(path)) return;
    seen.add(path);
    expectations.push({ path, expect, why });
  };
  for (const e of oldEntries) {
    const r = redirects.get(e.path);
    want(e.path, r ? r.to : 200, r ? `old sitemap, ${r.why}` : writes.has(fileOf(e.path)) ? 'old sitemap, in the copy' : 'old sitemap, now a page of the new site');
  }
  for (const [from, { to, why }] of redirects) if (why !== 'the slash' && why !== 'the old .html name') want(from, to, why);
  for (const f of served) {
  if (f.endsWith('.html')) for (const a of addressesOf(f)) want(a, redirects.get(a)?.to ?? 200, 'an old page, in the copy');
  else want(`/${f}`, 200, 'an old file, in the copy');
}
  for (const f of built) if (f.endsWith('index.html') && !f.startsWith('pagefind/')) want(addressesOf(f)[0], 200, 'a page of the new site');
  for (const p of ['/blog.xml', '/sitemap.xml', ARCHIVE_SITEMAP_PATH, '/robots.txt']) want(p, 200, 'a machine-readable list');

  return {
    writes,
    leaving,
    redirects,
    rules: rulesFile,
    archiveSitemap: sitemap(SITE, stillServed),
    expectations,
    stats: { pages, rules: rules.length + rewrites.length, oldSitemap: oldEntries.length, stillInSitemap: stillServed.length },
  };
}
