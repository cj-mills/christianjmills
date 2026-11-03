// Performance check, run after every build; any failure fails the build.
//
// 1. Static budgets on dist/: the bytes each page needs before it is usable, gzip-compressed as
//    served, plus per-file limits on images and interactive modules. Deterministic.
// 2. Runtime metrics in headless Chromium with Lighthouse-like mobile throttling (150 ms RTT,
//    1.6 Mbps down, 4x CPU slowdown): LCP and CLS, including the shift when an interactive mounts.
//
// PROVISIONAL budgets: the requirements set "performance targets the build checks" but no numbers
// yet. These are the prototype's proposal.
import { readFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { fileFor, pages, serve } from './lib/dist.ts';

const KB = 1024;
const BUDGET = {
  html: 30 * KB,
  css: 15 * KB,
  initialJs: 10 * KB,
  initialTotal: 100 * KB,
  image: 200 * KB,
  interactiveModule: 50 * KB,
  lcpMs: 2500,
  cls: 0.1,
};
// Pages whose function needs more script: search loads Pagefind's UI (its index loads on demand).
const JS_OVERRIDES: Record<string, number> = { '/search/': 60 * KB };

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
  for (const m of html.matchAll(/data-module="([^"]+)"/g)) modules.add(m[1]);

  const jsBudget = JS_OVERRIDES[page] ?? BUDGET.initialJs;
  const total = htmlBytes + css + js + images;
  if (htmlBytes > BUDGET.html) failures.push(`${page}: HTML ${kb(htmlBytes)} > ${kb(BUDGET.html)}`);
  if (css > BUDGET.css) failures.push(`${page}: CSS ${kb(css)} > ${kb(BUDGET.css)}`);
  if (js > jsBudget) failures.push(`${page}: initial JS ${kb(js)} > ${kb(jsBudget)}`);
  if (total > BUDGET.initialTotal) failures.push(`${page}: initial total ${kb(total)} > ${kb(BUDGET.initialTotal)}`);
  rows.push(`${page.padEnd(48)} html ${kb(htmlBytes).padStart(8)}  css ${kb(css).padStart(8)}  js ${kb(js).padStart(8)}  total ${kb(total).padStart(8)}`);
}
for (const m of modules) {
  const size = await gz(m);
  rows.push(`${('interactive ' + m).padEnd(48)} module ${kb(size)} (loaded when scrolled near)`);
  if (size > BUDGET.interactiveModule) failures.push(`${m}: interactive module ${kb(size)} > ${kb(BUDGET.interactiveModule)}`);
}

// Runtime metrics.
const { origin, server } = await serve();
const browser = await chromium.launch();
try {
  for (const page of await pages()) {
    const context = await browser.newContext({ viewport: { width: 412, height: 823 }, deviceScaleFactor: 2 });
    const tab = await context.newPage();
    const cdp = await context.newCDPSession(tab);
    await cdp.send('Network.enable');
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
    const cls = await tab.evaluate(() => (window as any).__cls as number);
    rows.push(`${page.padEnd(48)} LCP ${Math.round(lcp)} ms  CLS ${cls.toFixed(3)}`);
    if (lcp > BUDGET.lcpMs) failures.push(`${page}: LCP ${Math.round(lcp)} ms > ${BUDGET.lcpMs} ms`);
    if (cls > BUDGET.cls) failures.push(`${page}: CLS ${cls.toFixed(3)} > ${BUDGET.cls}`);
    await context.close();
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
