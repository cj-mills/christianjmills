// The tab icon check, run after every build: on every built page, desktop Chromium picks the
// monogram SVG as the tab icon, and the .ico fallback is still linked, served, and holds each size
// its link names (hub decision 2026-11-07-christianjmills-mark: the SVG is the favicon, and the
// cave is for browsers without SVG favicons).
//
// Why a check: HTML has no way to mark an icon as a fallback, and Chromium doesn't prefer an SVG.
// It scores each <link rel="icon"> by its `sizes` against the tab's sizes (16 px at each scale it
// supports: 16 and 32 px on desktop), downloads the best first, and stops when no later link's
// `sizes` promises better than what it has. An SVG is drawn at the largest size only, so it is
// never a perfect match, and an .ico whose `sizes` lists 16x16 and 32x32 (or `any`, or nothing)
// is downloaded after it and wins. A profile that already holds the SVG keeps using it until a
// reload, so this doesn't show in a browser that has seen the site before.
//
// The rule below follows Chromium 157's components/favicon/core/favicon_handler.cc and
// components/favicon_base/select_favicon_frames.cc (BSD-3-Clause). OBSERVED is what Chrome
// 157.0.8103.38 and Brave 1.99.55 stored for each declaration in a fresh profile on Linux
// (November 9, 2026); the check first confirms the rule reproduces it. Firefox 159 fetched the SVG
// for every one of them.
import { readFile } from 'node:fs/promises';
import { fileFor, pages } from './lib/dist.ts';

type Link = { href: string; type: string; sizes: string | null };
type Kind = 'svg' | 'ico';

// The tab icon's sizes in pixels on desktop: 16 px at 1x and at 2x.
const DESIRED = [16, 32];
const ICO_FRAMES = [16, 32, 48];

const kindOf = (l: Link): Kind | null => (l.type === 'image/svg+xml' || l.href.endsWith('.svg') ? 'svg' : l.href.endsWith('.ico') ? 'ico' : null);
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

/** How well the best of `sizes` serves one desired size: 1 for the exact size, far less otherwise. */
function scoreFor(sizes: number[], desired: number): number {
  if (sizes.includes(desired)) return 1;
  return Math.max(...sizes.map((s) => (s >= desired * 8 ? Math.min(1, desired / s) * 0.01 : s >= desired ? (desired / s) * 0.01 + 0.15 : Math.min(1, s / desired) * 0.01 + 0.1)));
}
const score = (sizes: number[]) => (sizes.length ? mean(DESIRED.map((d) => scoreFor(sizes, d))) : 0);

/** What a link promises: the best score for a link without sizes or with `any`, else its sizes' score. */
function promised(l: Link): number {
  const tokens = (l.sizes ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!tokens.length || tokens.includes('any')) return 1;
  return score(tokens.map((t) => Number(t.split('x')[0])));
}
/** What a downloaded icon gives: an SVG drawn at the largest desired size, or an .ico's frames up to it. */
function delivered(kind: Kind, icoFrames: number[]): number {
  const max = Math.max(...DESIRED);
  if (kind === 'svg') return score([max]);
  // Larger frames are dropped; if every frame is larger, the smallest is kept and scaled down.
  const kept = icoFrames.filter((s) => s <= max);
  return score(kept.length || !icoFrames.length ? kept : [Math.min(...icoFrames)]);
}

/** The icon desktop Chromium shows in the tab for a page's icon links, given in document order. */
export function chromiumTabIcon(links: Link[], icoFrames: number[]): Link | null {
  // Later links come first, and the sort keeps that order among equal scores.
  const candidates = links.toReversed().sort((a, b) => promised(b) - promised(a));
  let best: { link: Link; score: number } | null = null;
  for (const [i, link] of candidates.entries()) {
    const got = delivered(kindOf(link)!, icoFrames);
    if (!best || got > best.score) best = { link, score: got };
    const next = candidates[i + 1];
    if (!next || promised(next) <= best.score) break;
  }
  return best?.link ?? null;
}

const ico = (sizes: string | null): Link => ({ href: '/favicon.ico', type: '', sizes });
const svg = (sizes: string | null): Link => ({ href: '/favicon.svg', type: 'image/svg+xml', sizes });
const OBSERVED: [links: Link[], shown: Kind][] = [
  [[ico('16x16 32x32 48x48'), svg(null)], 'ico'],
  [[svg(null), ico('16x16 32x32 48x48')], 'ico'],
  [[ico('32x32'), svg(null)], 'svg'],
  [[ico('any'), svg(null)], 'ico'],
  [[ico(null), svg(null)], 'ico'],
  [[ico('16x16 32x32 48x48'), svg('any')], 'ico'],
  [[ico('48x48'), svg(null)], 'svg'],
  [[ico('32x32'), svg('any')], 'svg'],
];

const failures: string[] = [];
for (const [links, shown] of OBSERVED) {
  const got = kindOf(chromiumTabIcon(links, ICO_FRAMES)!);
  if (got !== shown) failures.push(`the rule gives the ${got} for ${JSON.stringify(links.map((l) => `${l.href} sizes=${l.sizes}`))}, and Chromium showed the ${shown}`);
}

/** The sizes of the images inside an .ico file (its directory: 6 bytes, then 16 bytes an image). */
function icoSizes(file: Buffer): number[] {
  if (file.readUInt16LE(0) !== 0 || file.readUInt16LE(2) !== 1) throw new Error('not an .ico file');
  return Array.from({ length: file.readUInt16LE(4) }, (_, i) => file[6 + 16 * i] || 256);
}
const attr = (tag: string, name: string) => new RegExp(`\\s${name}="([^"]*)"`).exec(tag)?.[1] ?? null;

const paths = await pages();
const frames = new Map<string, number[]>();
for (const path of paths) {
  const html = await readFile(fileFor(path), 'utf8');
  const links: Link[] = (html.match(/<link\b[^>]*>/g) ?? [])
    .filter((tag) => (attr(tag, 'rel') ?? '').split(/\s+/).includes('icon'))
    .map((tag) => ({ href: attr(tag, 'href') ?? '', type: attr(tag, 'type') ?? '', sizes: attr(tag, 'sizes') }));
  const fail = (what: string) => failures.push(`${path}: ${what}`);
  const unknown = links.filter((l) => !kindOf(l));
  const fallback = links.filter((l) => kindOf(l) === 'ico');
  if (unknown.length) fail(`an icon link this check has no rule for: ${unknown.map((l) => l.href).join(', ')}`);
  if (links.filter((l) => kindOf(l) === 'svg').length !== 1) fail('it should link one SVG icon');
  if (fallback.length !== 1) fail('it should link one .ico fallback');
  if (unknown.length || fallback.length !== 1) continue;

  const href = fallback[0].href;
  if (!frames.has(href)) frames.set(href, await readFile(fileFor(href)).then(icoSizes, () => []));
  const inFile = frames.get(href)!;
  if (!inFile.length) fail(`${href} isn't a built .ico file`);
  const named = (fallback[0].sizes ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  const missing = named.filter((t) => !inFile.includes(Number(t.split('x')[0])));
  if (missing.length) fail(`${href} is linked with sizes it doesn't hold: ${missing.join(' ')} (it holds ${inFile.join(', ')} px)`);

  const shown = chromiumTabIcon(links, inFile);
  if (!shown || kindOf(shown) !== 'svg') fail(`desktop Chromium would show ${shown?.href ?? 'no icon'} in the tab, not the SVG: the fallback's sizes="${fallback[0].sizes ?? ''}" promise its tab sizes (${DESIRED.join(' and ')} px)`);
}

if (failures.length) {
  console.error(`Tab icon check failed:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log(`Tab icon check passed: the SVG is desktop Chromium's tab icon on ${paths.length} pages, and the .ico fallback holds what its link names.`);
