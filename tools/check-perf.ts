// Performance check, run after every build; any failure fails the build.
//
// 1. Static budgets on dist/: the bytes each page needs before it is usable, gzip-compressed as
//    served, plus per-file limits on images and interactive modules. Deterministic.
// 2. Runtime metrics in headless Chromium with Lighthouse-like mobile throttling (150 ms RTT,
//    1.6 Mbps down, 4x CPU slowdown): LCP and CLS, including the shift when an interactive mounts,
//    and the font bytes the page fetches. Fonts are counted as the browser fetches them, because
//    which faces and subsets a page loads depends on what it sets (unicode ranges, weights, styles).
//
// 3. Interactives (hub decision 2026-11-08-christianjmills-interactive-budget), in the same
//    throttled run: how long each takes to mount, from when the site starts loading its module to
//    when mount has finished, and everything it loads to get there (the module, what it imports,
//    any WebAssembly), gzip-compressed and summed. Then, at a phone's and a desktop's size, that the
//    part of each interactive its module marks with data-fit (the drawing and the controls that
//    change it) is no taller than --interactive-max-block-size (spec/README.md, Interactives).
//
// Budgets: the HTML, font and interactive numbers are ruled (hub decisions
// 2026-11-09-christianjmills-interactive-and-html-budgets, 2026-11-07-christianjmills-typefaces).
// The rest are PROVISIONAL, the prototype's proposal: the requirements set "performance targets the
// build checks" but no numbers for them yet.
import { readFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { EXPECTED_MOUNT_FAILURES, fileFor, pages, serve } from './lib/dist.ts';

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

const failures: string[] = [];
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
    // Scroll through the page so interactives mount, then read the cumulative shift.
    await tab.evaluate(async () => {
      for (let y = 0; y < document.body.scrollHeight; y += 300) {
        window.scrollTo(0, y);
        await new Promise((r) => setTimeout(r, 100));
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
if (failures.length) {
  console.error(`\nperformance check failed:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('\nperformance check passed');
