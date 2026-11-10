// The interactive contract and the page's scripts, checked in a browser on the fixture page
// (spec/fixtures/parts, built by `npm run check:fixtures`): that a fallback stays until its module
// has finished mounting, that one module mounts many times with each mount's own parameters, that a
// module's reported values reach its own card and no other, that a failing module keeps its
// fallback, that a table sorts and comes back, that the rail marks the card being read, and that a
// themed figure follows the theme. Any failure fails the build.
import { chromium, type Page } from 'playwright';
import { serve } from './lib/dist.ts';

const PATH = '/fixtures/parts/';
const failures: string[] = [];
const expect = (ok: boolean, what: string) => {
  if (!ok) failures.push(what);
};

const { origin, server } = await serve();
const browser = await chromium.launch();

async function open(options: Parameters<typeof browser.newContext>[0]): Promise<Page> {
  const tab = await (await browser.newContext(options)).newPage();
  tab.on('pageerror', (e) => failures.push(`page error: ${e.message}`));
  await tab.goto(origin + PATH, { waitUntil: 'load' });
  return tab;
}

try {
  // ---- Mounting, at a desktop's width.
  // The window starts short, so every figure is more than a window's height below it.
  const tab = await open({ viewport: { width: 1280, height: 320 } });
  const first = tab.locator('figure[data-interactive="demo"]').nth(0);
  const second = tab.locator('figure[data-interactive="demo"]').nth(1);
  const broken = tab.locator('figure[data-interactive="broken"]');

  // Nothing mounts until its figure is near the window.
  expect((await tab.locator('[data-mounted], [data-mounting]').count()) === 0, "no interactive should start to mount before its figure is within a window's height of the window");
  expect((await first.locator('.step').count()) === 2 && (await first.locator('.step:visible').count()) === 1, 'with scripts running, a step list waiting for its module should show its last step alone');
  await tab.setViewportSize({ width: 1280, height: 720 });

  // While the module is getting ready, the fallback is still what the reader sees.
  await first.scrollIntoViewIfNeeded();
  await first.and(tab.locator('[data-mounting]')).waitFor({ timeout: 5000 });
  expect(await first.locator('.interactive-fallback').isVisible(), 'the fallback should stay in view while its module mounts');
  expect(!(await first.locator('.interactive-root').isVisible()), "the module's root should not be seen until it has mounted");
  const widths = await first.evaluate((f) => [f.querySelector<HTMLElement>('.interactive-root')!.clientWidth, f.querySelector<HTMLElement>('.interactive-fallback')!.clientWidth]);
  expect(widths[0] === widths[1] && widths[0] > 0, `while mounting, the root should have the width it will have (${widths[1]} px), not ${widths[0]} px`);

  await first.and(tab.locator('[data-mounted]')).waitFor({ timeout: 5000 });
  expect(await first.locator('.interactive-fallback').isHidden(), 'the fallback should be hidden once its module has mounted');
  expect((await first.locator('.status').textContent()) === 'first: Every tile starts as wall with a 55% chance.', `the first mount should show its own parameters and the first step's text, read from the fallback; it shows "${await first.locator('.status').textContent()}"`);

  await second.scrollIntoViewIfNeeded();
  await second.and(tab.locator('[data-mounted]')).waitFor({ timeout: 5000 });
  expect((await second.locator('.status').textContent())!.startsWith('second: '), 'the second mount of the same module should get its own parameters');
  const mounts = await tab.evaluate(async (url) => (await import(/* @vite-ignore */ url)).mounts, await first.getAttribute('data-module'));
  expect(JSON.stringify([...mounts].sort()) === '["first","second"]', `one module, imported once, should have mounted twice; its record is ${JSON.stringify(mounts)}`);

  // A reported value reaches the marks and the places of the module's own card, and its own series.
  const card = (n: number) => tab.locator('section.card').nth(n);
  expect((await card(0).locator('.trait-now').textContent()) === '0.500', `the first card should show its module's value to three places; it shows "${await card(0).locator('.trait-now').textContent()}"`);
  const marker = () => card(0).locator('.trait-mark').evaluate((m: HTMLElement) => [m.classList.contains('has-now'), m.style.getPropertyValue('--now')]);
  expect(JSON.stringify(await marker()) === '[true,"0.5"]', `the first card's mark should carry the marker at 0.5; it has ${JSON.stringify(await marker())}`);
  await first.locator('input[type=range]').fill('1');
  expect((await card(0).locator('.trait-now').textContent()) === '0.600', 'moving the control should change the value its card shows');
  expect((await tab.locator('.table-block').first().locator('.trait-mark.has-now').count()) === 0, 'a mark outside the card should get no marker');

  // A module that fails keeps its fallback, and leaves nothing of itself.
  await broken.scrollIntoViewIfNeeded();
  await broken.and(tab.locator('[data-mount-failed]')).waitFor({ timeout: 5000 });
  expect(await broken.locator('.interactive-fallback').isVisible(), "a failing module's fallback should stay");
  expect((await broken.locator('.interactive-root').count()) === 0, 'a failing module should leave no root behind');

  // ---- The table sorts by a column of numbers, and comes back as written.
  const table = tab.locator('table[data-sort]');
  const names = () => table.locator('tbody th[scope=row]').evaluateAll((cells) => cells.map((c) => c.textContent!.trim().split(' ')[0] + ' ' + c.textContent!.trim().split(' ')[1]));
  const written = await names();
  await table.getByRole('button', { name: 'Sort by Regions' }).click();
  expect(JSON.stringify(await names()) === '["The first","A third","The second"]', `sorted by Regions the rows should run 17, 3, 1; they run ${JSON.stringify(await names())}`);
  expect((await table.locator('th[aria-sort=descending]').count()) === 1 && (await table.locator('tr.group').first().isHidden()), 'a sorted table should say which column it is sorted by and drop its groups');
  await table.getByRole('button', { name: 'Sort by Regions' }).click();
  expect(JSON.stringify(await names()) === JSON.stringify(written) && (await table.locator('tr.group').first().isVisible()), 'choosing the sorted column again should put the rows back as written');
  expect((await table.getByRole('button', { name: 'Sort by Technique' }).count()) === 0, 'a column with no numbers should have no sort button');

  // ---- The rail lists each section's cards, and marks the card being read.
  const rail = tab.locator('.contents-rail');
  expect((await rail.locator('ol ol a').allTextContents()).join('|') === 'The first card|The second card', 'the rail should list the cards under their section');
  await tab.locator('#the-second-card').evaluate((h) => scrollTo(0, h.getBoundingClientRect().top + scrollY - 40));
  await tab.waitForFunction(() => document.querySelector('.contents-rail a[aria-current]')?.textContent === 'The second card', null, { timeout: 5000 }).catch(() => {});
  expect((await rail.locator('a[aria-current]').allTextContents()).join('|') === 'The second card', `the rail should mark the card being read, and only it; it marks "${(await rail.locator('a[aria-current]').allTextContents()).join('|')}"`);
  await tab.context().close();

  // ---- A themed figure follows the theme; a phone gets the contents as a disclosure, cards and all.
  for (const colorScheme of ['light', 'dark'] as const) {
    const phone = await open({ viewport: { width: 375, height: 740 }, colorScheme });
    const figure = phone.locator('article > figure img');
    await figure.scrollIntoViewIfNeeded();
    await phone.waitForFunction(() => document.querySelector<HTMLImageElement>('article > figure img')!.currentSrc !== '', null, { timeout: 5000 }).catch(() => {});
    const src = await figure.evaluate((img: HTMLImageElement) => img.currentSrc);
    expect(String(src).endsWith(`step-1-${colorScheme}.png`), `in ${colorScheme} the themed figure should show its ${colorScheme} image, not ${src}`);
    expect((await phone.locator('.contents-head ol ol a').count()) === 2 && !(await phone.locator('.contents-rail').isVisible()), "a phone's contents should be the disclosure, with the cards in it");
    expect(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'the page should not scroll sideways on a phone');
    await phone.context().close();
  }

  // ---- Without JavaScript the fallbacks are the page.
  const plain = await open({ viewport: { width: 375, height: 740 }, javaScriptEnabled: false });
  expect((await plain.locator('.interactive-fallback .step').count()) === 4 && (await plain.locator('.interactive-fallback').first().isVisible()), 'without JavaScript each step list should be there to read');
  await plain.context().close();
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  console.error(`fixture check failed (${failures.length}):\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('fixture check passed: mounting, parameters, reported values, a failing module, sorting, the rail and themed figures');
