// The interactive contract and the page's scripts, checked in a browser on the fixture page
// (spec/fixtures/parts, built by `npm run check:fixtures`): that a fallback stays until its module
// has finished mounting, that one module mounts many times with each mount's own parameters, that a
// module's reported values reach its own card and no other, that a failing module keeps its
// fallback, that the swap holds a figure's end for a reader whose window starts inside the figure,
// that a figure whose page reserves its module's room has that room from the first paint, keeps its
// height when the module mounts, and has no such room without scripts or once the module has failed,
// that the page's own scripts move nothing when they run and are one file linked from its head,
// which browsers are told to keep,
// that a table sorts and comes back, that the rail marks the card being read, that a
// themed figure follows the theme, and that a reference marker opens its note in place, leads to
// it on the references page without scripts, and goes when the reader switches references off.
// Any failure fails the build.
import { readFile } from 'node:fs/promises';
import { chromium, type Page } from 'playwright';
import { fileFor, serve } from './lib/dist.ts';

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

  // ---- The page's scripts are one file, linked from its head, that a browser keeps (hub decision
  // 2026-11-10-christianjmills-page-scripts-in-one-file). None is written into the page.
  const modules = await tab.evaluate(() => [...document.querySelectorAll<HTMLScriptElement>('script[type=module]')].map((s) => ({ src: s.getAttribute('src'), head: s.parentElement === document.head })));
  expect(modules.length === 1 && modules[0].head && /^\/_astro\/.+\.js$/.test(modules[0].src ?? ''), `the page's scripts should be one file under /_astro/, linked from the head; the page has ${JSON.stringify(modules)}`);
  const headers = await readFile(fileFor('/_headers'), 'utf8').catch(() => '');
  expect(/^\/_astro\/\*\n {2}Cache-Control: public, max-age=31536000, immutable$/m.test(headers), 'the build should carry a _headers rule that lets browsers keep the hashed files under /_astro/');

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

  // ---- References: a marker opens its note in place, copied from the references page.
  const words = (text: string | null) => (text ?? '').replace(/\s+/g, ' ').trim();
  const note = tab.locator('.ref-note');
  const refMarker = (n: number) => tab.locator('article .refs a', { hasText: `[${n}]` }).first();
  expect((await tab.locator('article > p .refs a').first().getAttribute('href')) === '/fixtures/parts/references/#ref-1', "a marker should link to its note on the page's references page");
  await refMarker(1).scrollIntoViewIfNeeded();
  await refMarker(1).click();
  await note.waitFor({ timeout: 5000 }).catch(() => {});
  const noteText = async () => words(await note.locator('span').first().textContent().catch(() => ''));
  expect((await noteText()) === 'The talk: 05:54.' && (await note.locator('a.ref-note-all').getAttribute('href')) === '/fixtures/parts/references/#ref-1', `marker 1's note should name its source and passage, and link to itself in the list; it reads "${await noteText()}"`);
  expect((await note.locator('a').nth(1).getAttribute('href')) === 'https://example.com/talk#t=354', 'a passage in a note should link to its place in the source');
  expect((await refMarker(1).getAttribute('aria-expanded')) === 'true', 'an open note should be told on its marker');
  const placed = await tab.evaluate(() => {
    const [m, n] = [document.querySelector('article .refs a')!.getBoundingClientRect(), document.querySelector('.ref-note')!.getBoundingClientRect()];
    return n.top >= m.bottom && n.left >= 0 && n.right <= document.documentElement.clientWidth;
  });
  expect(placed, 'a note should open under its marker, inside the window');
  await tab.keyboard.press('Escape');
  expect((await note.count()) === 0 && (await refMarker(1).getAttribute('aria-expanded')) === null, 'the Escape key should close a note and leave nothing of it');
  await refMarker(3).scrollIntoViewIfNeeded();
  await refMarker(3).click();
  await note.waitFor({ timeout: 5000 }).catch(() => {});
  expect((await noteText()) === "The talk's code: src/main.rs, line 43; lines 40–54. The talk: 06:10–06:59.", `marker 3's note should list both sources, the second passage of a file without its path; it reads "${await noteText()}"`);
  await refMarker(3).click();
  expect((await note.count()) === 0, "a click on an open note's marker should close it");
  // A marker a module has copied into its own root opens the same way.
  const copied = first.locator('.interactive-root .step-sources .refs a');
  await copied.scrollIntoViewIfNeeded();
  await copied.click();
  await note.waitFor({ timeout: 5000 }).catch(() => {});
  expect((await noteText()) === 'The talk: 06:10–06:59.', "a marker copied by an interactive's module should open its note");
  await tab.keyboard.press('Escape');

  // ---- The reader's switch: with the contents, kept across pages, applied before the page is drawn.
  const shown = () => tab.locator('.refs:visible').count();
  const boxes = () => tab.locator('.refs-switch input').evaluateAll((all) => all.map((b) => (b as HTMLInputElement).checked).join());
  expect((await tab.locator('.rail .refs-switch').isVisible()) && !(await tab.locator('header .refs-switch').isVisible()), 'where the page has a rail, the switch should be at its foot and nowhere else');
  expect((await shown()) > 0, 'references should show until the reader switches them off');
  await tab.locator('.rail .refs-switch input').uncheck();
  expect((await shown()) === 0, "switched off, no marker should show, nor an interactive's label for them");
  expect((await tab.evaluate(() => localStorage.getItem('refs'))) === 'off' && (await boxes()) === 'false,false', 'the choice should be kept, and both boxes should say it');
  await tab.reload({ waitUntil: 'load' });
  expect((await tab.evaluate(() => document.documentElement.dataset.refs)) === 'off' && (await shown()) === 0 && (await boxes()) === 'false,false', 'a kept "off" should hold on the next page load');
  expect(await tab.evaluate(() => [...document.head.querySelectorAll('script:not([src])')].some((s) => s.textContent!.includes("localStorage.getItem('refs')"))), "the kept choice should be applied by the page's head, before anything is drawn");
  await tab.locator('.rail .refs-switch input').check();
  expect((await shown()) > 0 && (await tab.evaluate(() => localStorage.getItem('refs'))) === null && (await boxes()) === 'true,true', 'switched on again, the markers should show and nothing should be kept');
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
    expect((await phone.locator('header .refs-switch').isVisible()) && !(await phone.locator('.rail').isVisible()), "a phone's switch for references should be under the contents, at the page's head");
    expect(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'the page should not scroll sideways on a phone');
    await phone.context().close();
  }

  // ---- The swap holds the figure's end for a reader whose window starts inside the figure: what
  // follows stays where it was, though the fixture's module opens far taller than its fallback. The
  // module is kept from loading until the window is in place and has been drawn there, and the
  // page's other modules for good, so nothing else changes size.
  const held = await (await browser.newContext({ viewport: { width: 375, height: 400 } })).newPage();
  let letThrough!: () => void;
  const waiting = new Promise<void>((r) => (letThrough = r));
  await held.route('**/_deliverables/**/*.js', async (route) => {
    await (route.request().url().endsWith('/grows.js') ? waiting : new Promise(() => {}));
    await route.continue().catch(() => {});
  });
  await held.goto(origin + PATH, { waitUntil: 'domcontentloaded' });
  const grows = held.locator('figure[data-interactive="grows"]');
  const frames = () => held.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  const place = () =>
    grows.evaluate((f) => {
      const r = f.getBoundingClientRect();
      return { start: r.top, end: r.bottom, height: r.height, next: f.nextElementSibling!.getBoundingClientRect().top };
    });
  await frames();
  await held.evaluate(() => document.fonts.ready);
  // The window starts just inside the figure, with the whole fallback in it: there the fallback is
  // what the browser would keep in place, and the swap hides it, so only the site's hold keeps what
  // follows where it was.
  await grows.evaluate((f) => scrollTo({ top: scrollY + f.getBoundingClientRect().top + 6, behavior: 'instant' }));
  await frames();
  const before = await place();
  letThrough();
  await grows.and(held.locator('[data-mounted]')).waitFor({ timeout: 5000 }).catch(() => {});
  await frames();
  const after = await place();
  expect(before.start < 0 && before.end > 0, `the window should start inside the figure before its module mounts; the figure runs from ${Math.round(before.start)} to ${Math.round(before.end)} px`);
  expect(after.height > before.height + 200, `the fixture's growing module should open far taller than its fallback; the figure went from ${Math.round(before.height)} to ${Math.round(after.height)} px`);
  expect(Math.abs(after.next - before.next) < 1, `with the window starting inside a figure, the swap should leave what follows the figure where it was; it moved ${Math.round(after.next - before.next)} px`);
  await held.context().close();

  // ---- A reserve (hub decision 2026-11-11-christianjmills-interactive-reserve): the fixture's
  // reserved figure declares its module's room on the tag. With scripts running the figure's live
  // area is that tall before the module is there; the module, which fills the room, mounts with
  // the figure's start in the window, and neither the figure's height nor what follows it changes.
  // When the module can't be loaded, the room goes and the fallback is what is left.
  for (const viewport of [{ width: 375, height: 740 }, { width: 1280, height: 720 }]) {
    for (const fails of [false, true]) {
      const tab = await (await browser.newContext({ viewport })).newPage();
      let release!: () => void;
      const released = new Promise<void>((r) => (release = r));
      await tab.route('**/_deliverables/**/*.js', async (route) => {
        if (!route.request().url().endsWith('/reserved.js')) return new Promise(() => {});
        await released;
        await (fails ? route.abort() : route.continue()).catch(() => {});
      });
      await tab.goto(origin + PATH, { waitUntil: 'domcontentloaded' });
      const figure = tab.locator('figure[data-interactive="reserved"]');
      const settle = () => tab.evaluate(async () => { await document.fonts.ready; await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))); });
      const read = () =>
        figure.evaluate((f) => {
          // The reserve is a length in CSS; an element given it as its height says what it comes to here.
          const probe = document.createElement('div');
          probe.style.cssText = 'position:absolute;visibility:hidden;block-size:var(--interactive-reserve)';
          f.append(probe);
          const reserve = probe.getBoundingClientRect().height;
          probe.remove();
          const live = f.querySelector<HTMLElement>('.interactive-root') ?? f.querySelector<HTMLElement>('.interactive-fallback')!;
          return { reserve, live: live.getBoundingClientRect().height, height: f.getBoundingClientRect().height, start: f.getBoundingClientRect().top, next: f.nextElementSibling!.getBoundingClientRect().top, status: f.querySelector('.status')?.textContent ?? '' };
        });
      await settle();
      await figure.evaluate((f) => scrollTo({ top: scrollY + f.getBoundingClientRect().top - innerHeight / 4, behavior: 'instant' }));
      await settle();
      const before = await read();
      const at = `at ${viewport.width} × ${viewport.height}`;
      expect(before.reserve > 200 && Math.abs(before.live - before.reserve) < 0.5, `${at}, with scripts running, a reserved figure's live area should be its reserve (${before.reserve.toFixed(1)} px) before its module is there; it is ${before.live.toFixed(1)} px`);
      expect(before.start > 0 && before.start < viewport.height, `${at} the reserved figure's start should be in the window while its module mounts; it is at ${Math.round(before.start)} px`);
      release();
      await figure.and(tab.locator(fails ? '[data-mount-failed]' : '[data-mounted]')).waitFor({ timeout: 5000 }).catch(() => {});
      await settle();
      const after = await read();
      if (fails) {
        expect((await figure.locator('.interactive-root').count()) === 0 && after.live < before.reserve - 100, `${at}, when a reserved figure's module fails, the room should go and the fallback be what is left; the live area is ${after.live.toFixed(1)} px of a reserve of ${after.reserve.toFixed(1)} px`);
      } else {
        expect(after.status === 'This module opened in the room its page reserved: min(16rem + 50cqi, var(--interactive-max-block-size)).', `a module should be handed its reserve as the tag wrote it; the fixture's shows "${after.status}"`);
        expect(Math.abs(after.live - after.reserve) < 0.5, `${at} a module that fills its reserve should be ${after.reserve.toFixed(1)} px tall once mounted; its root is ${after.live.toFixed(1)} px`);
        expect(Math.abs(after.height - before.height) < 0.5 && Math.abs(after.next - before.next) < 0.5, `${at} a reserved figure should keep its height when its module mounts, and what follows it its place; the figure went from ${before.height.toFixed(1)} to ${after.height.toFixed(1)} px and what follows moved ${(after.next - before.next).toFixed(1)} px`);
      }
      await tab.context().close();
    }
  }

  // ---- The page's own scripts move nothing: the copy buttons, the sort buttons and the rail's mark
  // arrive in room that was already there, so it doesn't matter when the scripts run. The page is
  // loaded without its module scripts and with its interactives' modules held back, every box is
  // read, the scripts are put back, and every box is read again.
  for (const viewport of [{ width: 375, height: 740 }, { width: 1280, height: 720 }]) {
    const still = await (await browser.newContext({ viewport })).newPage();
    const scripts: { src?: string; text: string }[] = [];
    await still.route(origin + PATH, async (route) => {
      const res = await route.fetch();
      const body = (await res.text()).replace(/<script type="module"([^>]*)>([\s\S]*?)<\/script>/g, (_all, attrs: string, text: string) => {
        scripts.push({ src: /\bsrc="([^"]*)"/.exec(attrs)?.[1], text });
        return '';
      });
      await route.fulfill({ response: res, body });
    });
    await still.route('**/_deliverables/**/*.js', () => new Promise(() => {}));
    await still.goto(origin + PATH, { waitUntil: 'load' });
    expect(scripts.length > 0 && (await still.locator('button.copy, button.sort').count()) === 0, "the page should have module scripts to hold back, and no buttons before they run");
    const settle = () => still.evaluate(async () => { await document.fonts.ready; await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))); });
    await settle();
    await still.evaluate(() => {
      const box = (e: Element) => { const r = e.getBoundingClientRect(); return [r.top + scrollY, r.left, r.width, r.height]; };
      (window as any).__boxes = [...document.querySelectorAll('body *')].map((e) => [e, box(e)]);
      (window as any).__box = box;
    });
    await still.evaluate((all) => {
      for (const { src, text } of all) {
        const script = document.createElement('script');
        script.type = 'module';
        if (src) script.src = src;
        else script.textContent = text;
        document.body.append(script);
      }
    }, scripts);
    await still.waitForFunction(() => document.querySelectorAll('button.copy').length === document.querySelectorAll('.code-block').length && document.querySelector('button.sort') !== null, null, { timeout: 5000 }).catch(() => {});
    await settle();
    const changed = await still.evaluate(() => {
      const w = window as any;
      const moved = (w.__boxes as [Element, number[]][]).filter(([e, was]) => (w.__box(e) as number[]).some((v, i) => Math.abs(v - was[i]) > 0.5));
      // The cause is a box that changed size with nothing inside it changing size.
      const resized = moved.filter(([e, was]) => (w.__box(e) as number[]).slice(2).some((v, i) => Math.abs(v - was[i + 2]) > 0.5));
      const causes = resized.filter(([e]) => !resized.some(([inner]) => inner !== e && e.contains(inner)));
      return { count: moved.length, buttons: document.querySelectorAll('button.copy, button.sort').length, first: causes.slice(0, 3).map(([e, was]) => `${e.tagName.toLowerCase()}${e.className ? '.' + String(e.className).split(' ')[0] : ''} (${was[3].toFixed(1)} to ${(w.__box(e) as number[])[3].toFixed(1)} px tall)`) };
    });
    expect(changed.buttons >= 3, `the page's scripts should have added the copy and sort buttons; there are ${changed.buttons}`);
    expect(changed.count === 0, `at ${viewport.width} × ${viewport.height} the page's scripts should move nothing: ${changed.count} boxes changed, starting with ${changed.first.join(', ')}`);
    await still.context().close();
  }

  // ---- Without JavaScript the fallbacks are the page.
  const plain = await open({ viewport: { width: 375, height: 740 }, javaScriptEnabled: false });
  expect((await plain.locator('.interactive-fallback .step').count()) === 4 && (await plain.locator('.interactive-fallback').first().isVisible()), 'without JavaScript each step list should be there to read');
  // A reserve is room for a module, so without scripts there is none: the fallback is as tall as its text.
  const unreserved = await plain.locator('figure[data-interactive="reserved"] .interactive-fallback').boundingBox();
  expect(unreserved !== null && unreserved.height < 120, `without JavaScript a reserved figure's fallback should be its own height, with no room kept under it; it is ${unreserved?.height.toFixed(1)} px tall`);
  // A marker is a plain link to its note on the references page, and the box still hides markers.
  await plain.locator('header .refs-switch input').uncheck();
  expect((await plain.locator('.refs:visible').count()) === 0, 'without JavaScript the switch should still hide the markers');
  await plain.locator('header .refs-switch input').check();
  await plain.locator('article > p .refs a').first().click();
  await plain.waitForURL('**/fixtures/parts/references/#ref-1', { timeout: 5000 }).catch(() => {});
  expect(plain.url().endsWith('/fixtures/parts/references/#ref-1') && (await plain.locator('#ref-1').isVisible()), `without JavaScript a marker should lead to its note on the references page, not ${plain.url()}`);
  expect((await plain.locator('#ref-3').textContent())!.replace(/\s+/g, ' ').trim() === "The talk's code: src/main.rs, line 43; lines 40–54. The talk: 06:10–06:59.", `the references page should list each note in full; note 3 reads "${await plain.locator('#ref-3').textContent()}"`);
  expect((await plain.locator('meta[name=robots][content=noindex]').count()) === 1 && (await plain.locator('#source-code').count()) === 1, 'the references page should be kept out of search engines and list the sources its notes name');
  await plain.context().close();
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  console.error(`fixture check failed (${failures.length}):\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log("fixture check passed: mounting, parameters, reported values, a failing module, a figure's end held, a reserve kept and let go, scripts in one kept file that move nothing, sorting, the rail, themed figures, reference notes and the switch");
