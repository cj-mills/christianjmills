// Accessibility check, run after every build; any violation fails the build.
// axe-core against every built page in headless Chromium, in light and dark, at a phone's width and
// a desktop's (the page frame changes at 64em: the contents are a disclosure below it and a rail
// above), with JavaScript on (interactives mounted, a search run) and off (the fallbacks a reader
// without JavaScript gets).
// axe needs JavaScript to run, so the "off" state is the page served with its scripts removed and
// its <noscript> content unwrapped, which is what a browser without JavaScript renders.
// Rules: WCAG 2.0, 2.1 and 2.2 A and AA, plus axe's best practices.
import AxeBuilder from '@axe-core/playwright';
import { chromium, type Page } from 'playwright';
import { EXPECTED_MOUNT_FAILURES, pages, serve } from './lib/dist.ts';

// Either side of the frame's 64em breakpoint.
const VIEWPORTS = { phone: { width: 375, height: 812 }, desktop: { width: 1280, height: 800 } };
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'];

async function exercise(tab: Page, path: string) {
  // Bring every interactive into view so it mounts, then wait until it has. One that fails to
  // mount fails the check, unless it is the fixture's, which fails on purpose.
  const figures = tab.locator('figure[data-interactive]');
  for (let i = 0; i < (await figures.count()); i++) {
    const figure = figures.nth(i);
    await figure.scrollIntoViewIfNeeded();
    await figure.and(tab.locator('[data-mounted], [data-mount-failed]')).waitFor({ timeout: 30_000 });
    const id = (await figure.getAttribute('data-interactive'))!;
    if ((await figure.getAttribute('data-mount-failed')) !== null && !EXPECTED_MOUNT_FAILURES.includes(id)) throw new Error(`${path}: interactive ${id} failed to mount`);
  }
  if (path === '/search/') {
    await tab.locator('pagefind-input input').fill('caves');
    await tab.locator('pagefind-results a').first().waitFor({ timeout: 5000 });
  }
}

const failures: string[] = [];
let runs = 0;
const { origin, server } = await serve();
const browser = await chromium.launch();
try {
  for (const path of await pages()) {
    for (const [width, viewport] of Object.entries(VIEWPORTS)) {
      for (const colorScheme of ['light', 'dark'] as const) {
        for (const javaScriptEnabled of [true, false]) {
          const context = await browser.newContext({ colorScheme, viewport });
          if (!javaScriptEnabled) {
            await context.route(origin + path, async (route) => {
              const res = await route.fetch();
              const body = (await res.text())
                .replace(/<script\b[\s\S]*?<\/script>/g, '')
                .replace(/<\/?noscript>/g, '');
              await route.fulfill({ response: res, body });
            });
          }
          const tab = await context.newPage();
          const errors: string[] = [];
          tab.on('pageerror', (e) => errors.push(e.message));
          await tab.goto(origin + path, { waitUntil: 'load' });
          if (javaScriptEnabled) await exercise(tab, path);
          const result = await new AxeBuilder({ page: tab }).withTags(TAGS).analyze();
          const mode = `${width}, ${colorScheme}, JS ${javaScriptEnabled ? 'on' : 'off'}`;
          for (const v of result.violations) {
            for (const n of v.nodes) failures.push(`${path} (${mode}): ${v.id}: ${v.help}\n      ${n.target.join(' ')}\n      ${n.failureSummary?.split('\n').join('\n      ')}`);
          }
          for (const e of errors) failures.push(`${path} (${mode}): page error: ${e}`);
          runs++;
          console.log(`  ${path} (${mode}): ${result.violations.length} violations`);
          await context.close();
        }
      }
    }
  }
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  console.error(`accessibility check failed (${failures.length}):\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log(`accessibility check passed: ${runs} page states, axe rules ${TAGS.join(', ')}`);
