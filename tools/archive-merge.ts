// Puts the frozen copy of the old site into dist/ after the new site is built and checked, and writes
// the site's routing (hub decision 2026-11-03-christianjmills-legacy-urls):
//
// - Every file of the archive (legacy/archive.lock.json) is served at its old address, unchanged but
//   for a banner at the start of each page's <main> saying it is archived.
// - A page leaves the copy when a new deliverable takes its address (`urls.canonical`) or replaces it
//   (`urls.supersedes`, a 301 to the new page). Quarto's alias pages become 301s, and the 71 posts
//   retired on 2026-10-09 redirect to their new homes (legacy/retired.json).
// - A file of the copy at a path the new site also builds fails the build, unless it is one of the
//   root files the new site replaces (REPLACED) or an old page a deliverable has taken over.
// - dist/_redirects: those 301s, a 301 to the slash for each folder's page, and two rewrites that
//   serve a folder's index.html. The Worker serves files as they are named (wrangler.jsonc,
//   `html_handling: "none"`), so the old site's `.html` and `/index.html` addresses answer 200 at
//   those addresses, as GitHub Pages answered them, where Cloudflare's default would redirect them.
// - dist/sitemap-archive.xml: the old sitemap's pages still served from the copy.
// - .build/urls.json: what every old and new address must answer, for tools/check-urls.ts.
import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { ARCHIVE_SITEMAP_PATH } from '../src/lib/lists.ts';
import { ROOT, fetchArchive, readLock, retiredPath, retiredSchema, unpackArchive, writeFileIn } from './lib/archive.ts';
import { planMerge } from './lib/legacy-plan.ts';
import { pinnedManifests } from './lib/manifests.ts';

const DIST = join(ROOT, process.env.SITE_DIST ?? 'dist');

const lock = await readLock();
const archive = unpackArchive(await fetchArchive(lock));
if (archive.size !== lock.files) throw new Error(`the archive holds ${archive.size} files; its pin says ${lock.files}`);
const retired = retiredSchema.parse(JSON.parse(await readFile(retiredPath, 'utf8'))).posts;
for (const dir of Object.keys(retired)) {
  if (!lock.exclude.includes(dir.slice(1))) throw new Error(`legacy/retired.json names ${dir}, which the archive still holds`);
}
const built = new Set<string>();
for (const e of await readdir(DIST, { recursive: true, withFileTypes: true })) {
  if (e.isFile()) built.add(relative(DIST, join(e.parentPath, e.name)).split(sep).join('/'));
}

// Every check runs in the plan, before anything is written, so a refused build leaves dist/ as
// Astro left it.
const plan = planMerge({ archive, builtOn: lock.source.built, retired, manifests: await pinnedManifests(), built });
for (const [path, out] of plan.writes) await writeFileIn(DIST, path, out);
await writeFile(join(DIST, '_redirects'), plan.rules);
await writeFile(join(DIST, ARCHIVE_SITEMAP_PATH.slice(1)), plan.archiveSitemap);
await mkdir(join(ROOT, '.build'), { recursive: true });
await writeFile(join(ROOT, '.build/urls.json'), JSON.stringify(plan.expectations, null, 1) + '\n');

console.log(
  `frozen copy merged: ${plan.writes.size} files (${plan.stats.pages} pages, each with the banner) from ${lock.source.branch}@${lock.source.commit.slice(0, 8)}; ` +
    `${plan.leaving.size} files left the copy; ${plan.stats.rules} rules in _redirects; ${plan.stats.stillInSitemap} of the old sitemap's ${plan.stats.oldSitemap} pages in ${ARCHIVE_SITEMAP_PATH}`,
);
