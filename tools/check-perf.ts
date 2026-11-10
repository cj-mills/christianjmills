// Performance check, run after every build; any failure fails the build.
//
// 1. Static budgets on dist/: the bytes each page needs before it is usable, gzip-compressed as
//    served, plus per-file limits on images and interactive modules. Deterministic.
// 2. Runtime metrics in headless Chromium with Lighthouse-like mobile throttling (150 ms RTT,
//    1.6 Mbps down, 4x CPU slowdown): LCP and CLS, and the font bytes the page fetches. Fonts are
//    counted as the browser fetches them, because which faces and subsets a page loads depends on
//    what it sets (unicode ranges, weights, styles). The run scrolls through the page a step at a
//    time and waits at each step for the interactives near the window to mount, so its CLS is that
//    of a reader who never outruns a mount, and doesn't depend on how the mounts fall against the
//    scrolling. What a mount does to a reader who is at its figure is step 4's to find.
//
// 3. Interactives (hub decision 2026-11-08-christianjmills-interactive-budget), in the same
//    throttled run: how long each takes to mount, from when the site starts loading its module to
//    when mount has finished, and everything it loads to get there (the module, what it imports,
//    any WebAssembly), gzip-compressed and summed. Then, at a phone's and a desktop's size, that the
//    part of each interactive its module marks with data-fit (the drawing and the controls that
//    change it) is no taller than --interactive-max-block-size (spec/README.md, Interactives).
//
// 4. The shift of each mount, whatever its timing (hub decision
//    2026-11-10-christianjmills-mount-shift-held-windows). The swap from a fallback to its module
//    moves the page only for a reader who has the figure, or what follows it, in the window when
//    the mount ends, so a run that scrolls sees it or not by chance. Here every interactive module
//    is kept from loading until the window is where the check wants it: at three places in each
//    figure, at a phone's size and a desktop's. The module is then let through, and the layout
//    shift from then until the figure has mounted is added to the page's shift from loading, as a
//    reader who went straight there would get both. Each hold is judged against the CLS budget.
//
// Budgets: the HTML, font and interactive numbers are ruled (hub decisions
// 2026-11-09-christianjmills-interactive-and-html-budgets, 2026-11-07-christianjmills-typefaces).
// The rest are PROVISIONAL, the prototype's proposal: the requirements set "performance targets the
// build checks" but no numbers for them yet.
import { readFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { chromium, type BrowserContext } from 'playwright';
import { EXPECTED_MOUNT_FAILURES, EXPECTED_MOUNT_SHIFTS, fileFor, pages, serve } from './lib/dist.ts';

const KB = 1024;
const BUDGET = {
  // Ruled: one number for every page. The map lab's page, the heaviest planned, measured 35.0 KB.
  html: 50 * KB,
  css: 15 * KB,
  initialJs: 10 * KB,
  initialTotal: 100 * KB,
  image: 200 * KB,
  // Ruled. Time to mount is the main check: the map lab on a stand-in module measured about
  // 1,000 ms, so the limit leaves room for a run's noise. Everything an interactive loads is the
  // backstop against bloat: the lab loads about 41 KB before its real module.
  interactiveMountMs: 1500,
  interactiveLoaded: 100 * KB,
  // Font files a page fetches, as transferred: ruled by the user (hub decision
  // 2026-11-07-christianjmills-typefaces). The text face's roman and italic, the display face's one
  // weight and the code face come to about 89 KB.
  fonts: 100 * KB,
  lcpMs: 2500,
  cls: 0.1,
};
// Pages whose function needs more script: search loads Pagefind's UI (its index loads on demand).
const JS_OVERRIDES: Record<string, number> = { '/search/': 60 * KB };

// The sizes the size rule is checked at: the phone and the desktop the design rounds were shown at.
const FIT_VIEWPORTS = [{ width: 375, height: 740 }, { width: 1280, height: 720 }];

// Where the window is held while a figure mounts, and at what sizes: the size the runtime metrics
// are read at and the desktop the design rounds were shown at. A place is one of the figure's edges
// and how far down the window it sits. With the figure's start in the window, the reader sees the
// swap and whatever follows the figure; with only its end there, the window starts inside the
// figure and the reader is reading on from it.
const HOLD_VIEWPORTS = [{ width: 412, height: 823, deviceScaleFactor: 2 }, { width: 1280, height: 720 }];
const HOLDS = [
  { edge: 'start', at: 1 / 4, where: 'its start a quarter of the way down the window' },
  { edge: 'end', at: 1 / 2, where: 'its end halfway down the window' },
  { edge: 'end', at: 1 / 8, where: 'its end an eighth of the way down the window' },
] as const;

// Published modules known to move the page when they mount, by the path they are served at, each
// with why the build lets it through. A path holds one version of one deliverable, so an entry ends
// with that version. An entry the check no longer finds moving the page, or whose module no page
// mounts, fails the build, so the list can't outlive its reason.
const KNOWN_MOUNT_SHIFTS: Record<string, string> = {};

const failures: string[] = [];
const known: string[] = [];
const gz = async (urlPath: string) => gzipSync(await readFile(fileFor(urlPath))).byteLength;
const attr = (tag: string, name: string) => new RegExp(`\\b${name}="([^"]*)"`).exec(tag)?.[1];
const kb = (n: number) => `${(n / KB).toFixed(1)} KB`;

const rows: string[] = [];
const modules = new Set<string>();
for (const page of await pages()) {
  const html = await readFile(fileFor(page), 'utf8');
  const htmlBytes = gzipSync(html).byteLength;
  let css = 0;
  for (const tag of html.match(/<link\b[^>]*rel="stylesheet"[^>]*>/g) ?? []) css += await gz(attr(tag, 'href')!);
  let js = 0;
  for (const tag of html.match(/<script\b[^>]*\bsrc="[^"]*"[^>]*>/g) ?? []) js += await gz(attr(tag, 'src')!);
  for (const m of html.matchAll(/<script\b(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)) js += gzipSync(m[1]).byteLength;
  let images = 0;
  for (const tag of html.match(/<img\b[^>]*>/g) ?? []) {
    const src = attr(tag, 'src')!;
    if (!attr(tag, 'width') || !attr(tag, 'height')) failures.push(`${page}: <img src="${src}"> has no width and height (layout shift)`);
    const size = (await readFile(fileFor(src))).byteLength;
    if (size > BUDGET.image) failures.push(`${page}: ${src} is ${kb(size)}; images are limited to ${kb(BUDGET.image)}`);
    if (attr(tag, 'loading') !== 'lazy') images += size;
  }
  // A themed figure's dark image is fetched in place of the light one, so it keeps the same limit.
  for (const tag of html.match(/<source\b[^>]*>/g) ?? []) {
    const src = attr(tag, 'srcset')!;
    const size = (await readFile(fileFor(src))).byteLength;
    if (size > BUDGET.image) failures.push(`${page}: ${src} is ${kb(size)}; images are limited to ${kb(BUDGET.image)}`);
  }
  for (const m of html.matchAll(/data-module="([^"]+)"/g)) modules.add(m[1]);

  const jsBudget = JS_OVERRIDES[page] ?? BUDGET.initialJs;
  const total = htmlBytes + css + js + images;
  if (htmlBytes > BUDGET.html) failures.push(`${page}: HTML ${kb(htmlBytes)} > ${kb(BUDGET.html)}`);
  if (css > BUDGET.css) failures.push(`${page}: CSS ${kb(css)} > ${kb(BUDGET.css)}`);
  if (js > jsBudget) failures.push(`${page}: initial JS ${kb(js)} > ${kb(jsBudget)}`);
  if (total > BUDGET.initialTotal) failures.push(`${page}: initial total ${kb(total)} > ${kb(BUDGET.initialTotal)}`);
  rows.push(`${page.padEnd(48)} html ${kb(htmlBytes).padStart(8)}  css ${kb(css).padStart(8)}  js ${kb(js).padStart(8)}  total ${kb(total).padStart(8)}`);
}
for (const m of modules) rows.push(`${('interactive ' + m).padEnd(48)} module ${kb(await gz(m))} (loaded as its figure nears the window)`);

// Runtime metrics.
const { origin, server } = await serve();
const browser = await chromium.launch();

interface Hold {
  /** The figure's place among the page's interactive figures, from 0, and how many there are. */
  figure: number;
  of: number;
  id: string;
  module: string;
  where: string;
  /** The page's layout shift before the module was let through, and from the hold to the figure's mount. */
  load: number;
  shift: number;
  /** The figure's height before and after. */
  from: number;
  to: number;
  viewport?: { width: number; height: number };
}

/** Holds the window at each place in each figure that mounts `module`, and reads the shift of each
 *  mount. A figure can be held once in a page load, and a figure near the held one mounts with it,
 *  so the page is loaded as many times as it takes to hold every figure at every place. Only
 *  `module` is ever let through, so a shift is that module's. */
async function holdWindows(context: BrowserContext, page: string, module: string, pageModules: string[]): Promise<Hold[]> {
  const out: Hold[] = [];
  let pending: { figure: number; hold: (typeof HOLDS)[number] }[] | undefined;
  while (!pending || pending.length) {
    let open!: () => void;
    const opened = new Promise<void>((r) => (open = r));
    let isOpen = false;
    const tab = await context.newPage();
    await tab.addInitScript(() => {
      let total = 0;
      const add = (entries: PerformanceEntryList) => { for (const e of entries as any[]) if (!e.hadRecentInput) total += e.value; };
      const observer = new PerformanceObserver((list) => add(list.getEntries()));
      observer.observe({ type: 'layout-shift', buffered: true });
      const frames = (n: number) => new Promise<void>((done) => { const step = () => (n-- > 0 ? requestAnimationFrame(step) : done()); step(); });
      (window as any).__held = {
        frames,
        // The shift so far. An entry is ready a frame after the change it reports.
        read: () => { add(observer.takeRecords()); return total; },
        settled: (f: HTMLElement) => 'mounted' in f.dataset || 'mountFailed' in f.dataset,
      };
    });
    // Every module waits; this one until the window is in place, the others for good.
    await tab.route((url) => pageModules.includes(url.pathname), async (route) => {
      if (new URL(route.request().url()).pathname === module) await opened;
      else await new Promise(() => {});
      await route.continue().catch(() => {});
    });
    // A module the page is waiting for may hold back its load event, so the page is ready when its fonts are.
    await tab.goto(origin + page, { waitUntil: 'domcontentloaded' });
    const { load, figures } = await tab.evaluate(async () => {
      const held = (window as any).__held;
      await held.frames(2);
      await document.fonts.ready;
      await held.frames(2);
      return {
        load: held.read() as number,
        figures: [...document.querySelectorAll<HTMLElement>('figure[data-interactive]')].map((f) => ({ id: f.dataset.interactive!, module: new URL(f.dataset.module!, location.href).pathname })),
      };
    });
    pending ??= figures.flatMap((f, figure) => (f.module === module ? HOLDS.map((hold) => ({ figure, hold })) : []));
    let progressed = false;
    for (const next of [...pending]) {
      // Put the window in place, unless this figure has already mounted in this load, beside another.
      // The window is drawn there before the module is let through, as a reader's has been: a
      // scroll and a swap in one frame would be read as one move.
      const start = await tab.evaluate(async ({ figure, edge, at }) => {
        const held = (window as any).__held;
        const f = document.querySelectorAll<HTMLElement>('figure[data-interactive]')[figure];
        if (held.settled(f) || 'mounting' in f.dataset) return null;
        const r = f.getBoundingClientRect();
        const mark = held.read() as number;
        scrollTo({ top: scrollY + (edge === 'start' ? r.top : r.bottom) - at * innerHeight, behavior: 'instant' });
        await held.frames(2);
        return { mark, from: r.height };
      }, { figure: next.figure, edge: next.hold.edge, at: next.hold.at });
      if (!start) continue;
      if (!isOpen) { open(); isOpen = true; }
      // Wait for this figure, and for any other the site has started on: those near the window.
      const done = await tab
        .waitForFunction(({ figure, module }) => {
          const held = (window as any).__held;
          const all = [...document.querySelectorAll<HTMLElement>('figure[data-interactive]')];
          if (!held.settled(all[figure])) return false;
          return all.every((f) => {
            if (new URL(f.dataset.module!, location.href).pathname !== module || held.settled(f)) return true;
            if ('mounting' in f.dataset) return false;
            const r = f.getBoundingClientRect();
            return r.bottom < 1 - innerHeight || r.top > 2 * innerHeight - 1;
          });
        }, { figure: next.figure, module }, { timeout: 30_000 })
        .then(() => true, () => false);
      pending.splice(pending.indexOf(next), 1);
      progressed = true;
      if (!done) {
        failures.push(`${page}: interactive ${figures[next.figure].id} neither mounted nor failed within 30 s of its module loading, with the window held at its figure`);
        break;
      }
      const end = await tab.evaluate(async (figure) => {
        const held = (window as any).__held;
        await held.frames(3);
        return { total: held.read() as number, to: document.querySelectorAll<HTMLElement>('figure[data-interactive]')[figure].getBoundingClientRect().height };
      }, next.figure);
      out.push({ figure: next.figure, of: figures.length, id: figures[next.figure].id, module, where: next.hold.where, load, shift: end.total - start.mark, from: start.from, to: end.to });
    }
    await tab.close();
    if (!progressed) {
      failures.push(`${page}: could not hold the window for interactive ${figures.find((f) => f.module === module)?.id}: its figures mounted before ${module} was let through`);
      break;
    }
  }
  return out;
}
try {
  for (const page of await pages()) {
    const context = await browser.newContext({ viewport: { width: 412, height: 823 }, deviceScaleFactor: 2 });
    const tab = await context.newPage();
    const cdp = await context.newCDPSession(tab);
    await cdp.send('Network.enable');
    const fontRequests = new Set<string>();
    const fontFiles = new Map<string, number>();
    const fontUrls = new Map<string, string>();
    cdp.on('Network.responseReceived', (e) => {
      if (e.type === 'Font') {
        fontRequests.add(e.requestId);
        fontUrls.set(e.requestId, new URL(e.response.url).pathname);
      }
    });
    cdp.on('Network.loadingFinished', (e) => {
      if (fontRequests.has(e.requestId)) fontFiles.set(fontUrls.get(e.requestId)!, e.encodedDataLength);
    });
    // What asked for each file: a module's imports and the files it fetches lead back to it.
    const askedBy = new Map<string, string>();
    cdp.on('Network.requestWillBeSent', (e) => {
      const from = e.initiator.url ?? e.initiator.stack?.callFrames[0]?.url;
      if (from && !askedBy.has(e.request.url)) askedBy.set(e.request.url, from);
    });
    await cdp.send('Network.emulateNetworkConditions', {
      offline: false,
      latency: 150,
      downloadThroughput: (1.6 * 1024 * 1024) / 8,
      uploadThroughput: (750 * 1024) / 8,
    });
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
    await tab.addInitScript(() => {
      (window as any).__cls = 0;
      (window as any).__lcp = 0;
      new PerformanceObserver((list) => {
        for (const e of list.getEntries() as any[]) if (!e.hadRecentInput) (window as any).__cls += e.value;
      }).observe({ type: 'layout-shift', buffered: true });
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) (window as any).__lcp = e.startTime;
      }).observe({ type: 'largest-contentful-paint', buffered: true });
    });
    await tab.goto(origin + page, { waitUntil: 'load' });
    await tab.waitForTimeout(500);
    const lcp = await tab.evaluate(() => (window as any).__lcp as number);
    // Scroll through the page so interactives mount, then read the cumulative shift. At each step
    // the window waits for the figures near it to mount (the site starts a mount a window's height
    // ahead), so where the window is when a mount ends doesn't depend on how long the mount took.
    await tab.evaluate(async () => {
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      const figures = [...document.querySelectorAll<HTMLElement>('figure[data-interactive]')];
      const mounting = () =>
        figures.some((f) => {
          if ('mounted' in f.dataset || 'mountFailed' in f.dataset) return false;
          if ('mounting' in f.dataset) return true;
          const r = f.getBoundingClientRect();
          return r.bottom >= 1 - innerHeight && r.top <= 2 * innerHeight - 1;
        });
      for (let y = 0; y < document.body.scrollHeight; y += 300) {
        window.scrollTo(0, y);
        await sleep(100);
        for (const until = performance.now() + 30_000; mounting() && performance.now() < until; ) await sleep(50);
      }
    });
    await tab.waitForTimeout(1000);
    // Interactives: wait for each to mount or fail, then read how long each took and what it loaded.
    const interactive = await tab.locator('figure[data-interactive]').count();
    if (interactive) {
      await tab
        .waitForFunction(() => [...document.querySelectorAll<HTMLElement>('figure[data-interactive]')].every((f) => 'mounted' in f.dataset || 'mountFailed' in f.dataset), null, { timeout: 60_000 })
        .catch(() => failures.push(`${page}: an interactive neither mounted nor failed within 60 s`));
      const state = await tab.evaluate(() => ({
        failed: [...document.querySelectorAll<HTMLElement>('figure[data-mount-failed]')].map((f) => f.dataset.interactive!),
        modules: [...new Set([...document.querySelectorAll<HTMLElement>('figure[data-interactive]')].map((f) => `${f.dataset.interactive} ${new URL(f.dataset.module!, location.href).href}`))],
        mounts: performance.getEntriesByType('measure').filter((m) => m.name.startsWith('interactive:')).map((m) => [m.name.slice(12), m.duration] as [string, number]),
      }));
      for (const id of state.failed) if (!EXPECTED_MOUNT_FAILURES.includes(id)) failures.push(`${page}: interactive ${id} failed to mount`);
      for (const entry of state.modules) {
        const [id, url] = entry.split(' ');
        const times = state.mounts.filter(([name]) => name === id).map(([, ms]) => ms);
        if (!times.length) continue;
        // The first mount waits for the module and whatever it loads; later mounts of it share that.
        const slowest = Math.max(...times);
        const belongs = (u: string, seen = new Set<string>()): boolean => u === url || (!seen.has(u) && askedBy.has(u) && belongs(askedBy.get(u)!, seen.add(u)));
        const loaded = [url, ...[...askedBy.keys()].filter((u) => u !== url && u.startsWith(origin) && belongs(u))];
        let bytes = 0;
        for (const u of loaded) bytes += await gz(new URL(u).pathname);
        rows.push(`${(page + ' interactive ' + id).padEnd(48)} mounts ${times.length}  slowest ${Math.round(slowest)} ms  loads ${kb(bytes)} in ${loaded.length} files`);
        if (slowest > BUDGET.interactiveMountMs) failures.push(`${page}: interactive ${id} took ${Math.round(slowest)} ms to mount > ${BUDGET.interactiveMountMs} ms`);
        if (bytes > BUDGET.interactiveLoaded) failures.push(`${page}: interactive ${id} loads ${kb(bytes)} > ${kb(BUDGET.interactiveLoaded)} (${loaded.map((u) => new URL(u).pathname).join(', ')})`);
      }
    }
    const cls = await tab.evaluate(() => (window as any).__cls as number);
    const fonts = [...fontFiles.values()].reduce((a, b) => a + b, 0);
    rows.push(`${page.padEnd(48)} LCP ${Math.round(lcp)} ms  CLS ${cls.toFixed(3)}  fonts ${kb(fonts)} in ${fontFiles.size} files`);
    if (fonts > BUDGET.fonts) failures.push(`${page}: fonts ${kb(fonts)} > ${kb(BUDGET.fonts)} (${[...fontFiles.keys()].join(', ')})`);
    if (!fontFiles.size) failures.push(`${page}: fetched no font files; the self-hosted faces are not loading`);
    if (lcp > BUDGET.lcpMs) failures.push(`${page}: LCP ${Math.round(lcp)} ms > ${BUDGET.lcpMs} ms`);
    if (cls > BUDGET.cls) failures.push(`${page}: CLS ${cls.toFixed(3)} > ${BUDGET.cls}`);
    await context.close();
  }

  // The shift of each mount, with the window held at each place in each figure (step 4, above).
  const mounted = new Set<string>();
  for (const page of await pages()) {
    const pageModules = [...new Set([...(await readFile(fileFor(page), 'utf8')).matchAll(/data-module="([^"]+)"/g)].map((m) => m[1]))];
    for (const m of pageModules) mounted.add(m);
    if (!pageModules.length) continue;
    const holds: Hold[] = [];
    for (const viewport of HOLD_VIEWPORTS) {
      const context = await browser.newContext({ viewport, deviceScaleFactor: viewport.deviceScaleFactor ?? 1 });
      for (const module of pageModules) holds.push(...(await holdWindows(context, page, module, pageModules)).map((h) => ({ ...h, viewport })));
      await context.close();
    }
    // LIST_HOLDS=1 prints every hold, for whoever is fitting a module to its fallback.
    if (process.env.LIST_HOLDS) for (const h of holds) rows.push(`  ${h.viewport!.width} × ${h.viewport!.height}  ${h.id} figure ${h.figure + 1} of ${h.of}, ${h.where}: shift ${(h.load + h.shift).toFixed(3)} (${h.load.toFixed(3)} from loading), ${Math.round(h.from)} px to ${Math.round(h.to)} px tall`);
    for (const module of pageModules) {
      const mine = holds.filter((h) => h.module === module);
      if (!mine.length) continue;
      const id = mine[0].id;
      const figures = new Set(mine.map((h) => h.figure)).size;
      const over = mine.filter((h) => h.load + h.shift > BUDGET.cls);
      const worst = mine.reduce((a, b) => (b.load + b.shift > a.load + a.shift ? b : a));
      const told = `layout shift ${(worst.load + worst.shift).toFixed(3)} at ${worst.viewport!.width} × ${worst.viewport!.height} with figure ${worst.figure + 1} of ${worst.of} held with ${worst.where}; the figure went from ${Math.round(worst.from)} px to ${Math.round(worst.to)} px tall`;
      rows.push(`${(page + ' interactive ' + id).padEnd(48)} held ${mine.length} times in ${figures} figures  worst shift ${(worst.load + worst.shift).toFixed(3)}  over the limit ${over.length}`);
      const why = KNOWN_MOUNT_SHIFTS[module] ?? (EXPECTED_MOUNT_SHIFTS.includes(id) ? 'the fixture moves the page on purpose' : undefined);
      if (over.length && why) known.push(`${page}: interactive ${id} moves the page when it mounts: ${told}. ${over.length} of ${mine.length} holds are over ${BUDGET.cls}. Let through: ${why}`);
      else if (over.length) failures.push(`${page}: interactive ${id} moves the page when it mounts: ${told}. ${over.length} of ${mine.length} holds are over ${BUDGET.cls}`);
      else if (why) failures.push(`${page}: interactive ${id} is listed as moving the page when it mounts, and no hold found a shift over ${BUDGET.cls} (worst: ${told}); take it off the list`);
    }
  }
  // An entry for a module no page mounts has outlived its version. A run narrowed to some pages can't tell.
  if (!process.env.CHECK_PAGES) for (const m of Object.keys(KNOWN_MOUNT_SHIFTS)) if (!mounted.has(m)) failures.push(`${m} is listed as known to move the page when it mounts, and no page mounts it; take it off the list`);

  // The size rule: what a module marks with data-fit is no taller than the frame allows.
  for (const page of await pages()) {
    if (!/data-interactive=/.test(await readFile(fileFor(page), 'utf8'))) continue;
    for (const viewport of FIT_VIEWPORTS) {
      const context = await browser.newContext({ viewport });
      const tab = await context.newPage();
      await tab.goto(origin + page, { waitUntil: 'load' });
      const figures = tab.locator('figure[data-interactive]');
      for (let i = 0; i < (await figures.count()); i++) {
        await figures.nth(i).scrollIntoViewIfNeeded();
        await figures.nth(i).and(tab.locator('[data-mounted], [data-mount-failed]')).waitFor({ timeout: 30_000 }).catch(() => {});
      }
      const over = await tab.evaluate(() =>
        [...document.querySelectorAll<HTMLElement>('figure[data-interactive] [data-fit]')].flatMap((el) => {
          // The limit is a length in CSS; an element given it as its height says what it comes to here.
          const probe = document.createElement('div');
          probe.style.cssText = 'position:absolute;visibility:hidden;block-size:var(--interactive-max-block-size)';
          el.closest('figure')!.append(probe);
          const limit = probe.getBoundingClientRect().height;
          probe.remove();
          const height = el.getBoundingClientRect().height;
          return height > limit + 0.5 ? [`interactive ${el.closest<HTMLElement>('figure')!.dataset.interactive}: its drawing and controls are ${Math.round(height)} px tall; the frame allows ${Math.round(limit)} px`] : [];
        }),
      );
      for (const o of over) failures.push(`${page} at ${viewport.width} × ${viewport.height}: ${o}`);
      await context.close();
    }
  }
} finally {
  await browser.close();
  server.close();
}

console.log(rows.join('\n'));
if (known.length) console.log(`\nknown, and let through:\n  ${known.join('\n  ')}`);
if (failures.length) {
  console.error(`\nperformance check failed:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('\nperformance check passed');
