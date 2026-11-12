// The format's own check, run before every build: the fixture page in spec/fixtures/parts uses every
// part of the dialect, and this verifies it as the build verifies a bundle, then checks what was
// rendered, what the Markdown copy says, that each mistake a page can make is refused by name, and
// that the American English check finds what it should. Any failure fails the build.
import { fileURLToPath } from 'node:url';
import { readBundleDir, verify } from '../spec/bundle.ts';
import { checkBundle, checkText } from '../spec/english.ts';

const dir = fileURLToPath(new URL('../spec/fixtures/parts/', import.meta.url));
const files = await readBundleDir(dir);
const assetBase = () => '/_deliverables/fixture-parts/1.0.0/';
const failures: string[] = [];
const expect = (ok: boolean, what: string) => {
  if (!ok) failures.push(what);
};

const page = await verify(files, { assetBase });
const html = page.html;
const has = (fragment: string, what: string) => expect(html.includes(fragment), `the page should have ${what}: ${fragment}`);

// Themed figures.
has('<picture><source media="(prefers-color-scheme: dark)" srcset="/_deliverables/fixture-parts/1.0.0/assets/step-1-dark.png">', 'a dark source for a themed figure');
// Cards and parts.
has('<section class="card"><header class="card-head"><p class="card-kind">Generator</p><h3 id="first">The first card</h3>', "a card's head, with its own id");
has('<h3 id="the-second-card">', 'a card id made from its title');
has('<section class="part"><h4 id="first-the-rule">The rule</h4>', "a part, its id starting with its card's");
has('<h4 id="the-second-card-step-through-it">', "the same part title in a second card, under that card's id");
expect(
  JSON.stringify(page.headings.filter((h) => h.depth < 4)) ===
    JSON.stringify([
      { depth: 2, slug: 'which-one', text: 'Which one' },
      { depth: 2, slug: 'generators', text: 'Generators' },
      { depth: 3, slug: 'first', text: 'The first card', card: true },
      { depth: 3, slug: 'the-second-card', text: 'The second card', card: true },
      { depth: 2, slug: 'shaping', text: 'Shaping' },
    ]),
  `the outline should list the sections and their cards, not ${JSON.stringify(page.headings)}`,
);
// Step lists.
has('<li class="step" data-key="noise" data-at="1" data-of="11"><picture>', 'a step with its key and place');
has('<p class="step-label">Step 1 of 11</p>', "a step's place in a longer run");
has('<p class="step-label">Step 1</p>', 'a step numbered by its place in the list');
has('<span class="step-value" data-value="">55</span>', 'a value an interactive replaces');
// Interactives.
has(`data-params="${JSON.stringify({ name: 'first', value: 0.5, series: 'c01' }).replaceAll('"', '&quot;')}"`, "a mount's parameters");
expect(html.split('data-interactive="demo"').length === 3, 'one module should be mounted twice on the fixture page');
has('data-interactive="reserved" data-module="/_deliverables/fixture-parts/1.0.0/assets/reserved.js" data-reserve="min(16rem + 50cqi, var(--interactive-max-block-size))" style="--interactive-reserve: min(16rem + 50cqi, var(--interactive-max-block-size))"', "a mount's reserve, as written and as the length the site's CSS holds the figure to");
expect(!/data-interactive="(?:demo|broken|grows)"[^>]*(?:data-reserve|style)=/.test(html), 'a mount that declares no reserve should carry none');
// Tables and marks.
has('<div class="table-scroll" tabindex="0" role="region" aria-label="The techniques compared"><table data-sort=""><caption class="visually-hidden">The techniques compared</caption>', 'a named, sortable table in its scrolling region');
has('<tr class="group"><th scope="rowgroup" colspan="4"><span>Generators</span></th></tr>', 'a row that heads a group');
has('<th scope="row"><a href="#first">The first card</a>', 'a row head');
has('<span class="scale">0 to 40</span>', "a column's scale at its head");
has('style="--lo:0.608;--mid:0.637;--hi:0.665"', "a mark's positions on its scale");
has('aria-label="Median 0.637. 8 of 10 seeds fall between 0.608 and 0.665. Lowest 0.559, highest 0.698, over 200 seeds."', 'a mark said in words');
has('<span class="key key-now"></span>the map on screen', "the key's marker, in a table that shows an interactive's value");
expect(html.split('key-now').length === 2, "the key should name the marker only where a table shows an interactive's value");
has('<span class="trait-now" data-trait="floor_share" data-digits="3"></span>', "the place for an interactive's own value");
// References: a block's citations are one numbered marker at its end, linking to its note on the
// page's references page; the same passages get the same number wherever they are cited.
const NBSP = '\u00a0';
const mark = (n: number) => `<span class="refs">${NBSP}<a href="/fixtures/parts/references/#ref-${n}" role="doc-noteref">[${n}]</a></span>`;
has(`cites a source.${mark(1)}</p>`, "a paragraph's marker, after its last word");
has(`neighbors are.${mark(2)}</p>`, "a part's marker");
has(`<li>Count the walls around a tile.${mark(3)}</li>`, 'one marker for a list item that cites three passages of two sources');
has(`<li>Five or more make it wall.${mark(2)}</li>`, 'the number of the note that already holds the same passages');
has(`the caves have settled.${mark(2)}</p>`, "a step's marker, in the text a module reads");
expect(!html.includes('class="ref"') && !html.includes('example.com/talk#'), "the page's HTML should hold no locator: they are on its references page");
expect(
  JSON.stringify(page.notes) ===
    JSON.stringify([
      { n: 1, sources: [{ id: 'talk', label: 'the talk', items: [{ locator: '05:54', href: 'https://example.com/talk#t=354' }] }] },
      { n: 2, sources: [{ id: 'talk', label: 'the talk', items: [{ locator: '06:10–06:59', href: 'https://example.com/talk#t=370,419' }] }] },
      {
        n: 3,
        sources: [
          {
            id: 'code',
            label: "the talk's code",
            items: [
              { locator: 'src/main.rs, line 43', href: 'https://example.com/repo/blob/0123456789abcdef0123456789abcdef01234567/src/main.rs#L43' },
              { locator: 'src/main.rs, lines 40–54', href: 'https://example.com/repo/blob/0123456789abcdef0123456789abcdef01234567/src/main.rs#L40-L54' },
            ],
          },
          { id: 'talk', label: 'the talk', items: [{ locator: '06:10–06:59', href: 'https://example.com/talk#t=370,419' }] },
        ],
      },
    ]),
  `the page's notes should be its three sets of passages, in the order first cited, a single line as "line 43"; they are ${JSON.stringify(page.notes)}`,
);

// The Markdown copy.
const says = (fragment: string, what: string) => expect(page.markdown.includes(fragment), `the Markdown copy should have ${what}: ${fragment}`);
says('### The first card\n\n*Generator*\n\n', "a card's title, then its kind");
says('#### The rule\n\n', 'a part');
says('**Step 1 of 11.**\n\nEvery tile starts as wall with a 55% chance.\n\n![The map at step 1 of 11.](/_deliverables/fixture-parts/1.0.0/assets/step-1-light.png)', 'a step: its place, its text, its figure');
says('**The techniques compared**\n\n| Technique | Floor share | Regions | Seeds failing |\n| --- | --- | --- | --- |\n| **Generators** |  |  |  |', 'a table under its name, with its group rows');
says('| [The first card](#first) `c01` | 0.637 (8 of 10: 0.608 to 0.665) |', 'a mark in words');
says("1. Count the walls around a tile [[the talk's code, src/main.rs, line 43]](https://example.com/repo/blob/0123456789abcdef0123456789abcdef01234567/src/main.rs#L43)[[the talk's code, src/main.rs, lines 40–54]](https://example.com/repo/blob/0123456789abcdef0123456789abcdef01234567/src/main.rs#L40-L54)[[the talk, 06:10–06:59]](https://example.com/talk#t=370,419).", 'every reference in full, one per claim, where the HTML has a number');
expect(!page.markdown.includes('#ref-'), 'the Markdown copy should have no numbered markers');

// Each mistake is refused, by name.
const mdoc = new TextDecoder().decode(files.get('content/index.mdoc')!);
async function refuses(id: string, change: (source: string) => string) {
  const changed = change(mdoc);
  if (changed === mdoc) return failures.push(`[${id}] the fixture no longer has the text this test changes`);
  try {
    await verify(new Map(files).set('content/index.mdoc', new TextEncoder().encode(changed)), { assetBase });
    failures.push(`[${id}] should have been refused`);
  } catch (err) {
    expect(String(err).includes(`[${id}]`) || String(err).includes(id), `[${id}] was refused for another reason: ${String(err).slice(0, 300)}`);
  }
}
await refuses('nesting', (s) => s.replace('{% part title="The rule" %}', '{% /card %}\n{% part title="The rule" %}').replace('{% /part %}\n{% /card %}\n\n{% card title="The second card" %}', '{% /part %}\n\n{% card title="The second card" %}'));
await refuses('nesting', (s) => s.replace('{% steps %}\n{% step src="assets/step-1-light.png"', '{% step src="assets/step-1-light.png"').replace('{% /step %}\n{% /steps %}\n{% /interactive %}\n{% /part %}\n\n{% part title="What it produces', '{% /step %}\n{% /interactive %}\n{% /part %}\n\n{% part title="What it produces'));
await refuses('asset-dark-size', (s) => s.replace('dark="assets/step-1-dark.png" alt="A map', 'dark="assets/odd.png" alt="A map'));
await refuses('asset-missing', (s) => s.replace('dark="assets/step-2-dark.png"', 'dark="assets/nowhere.png"'));
await refuses('step-alt', (s) => s.replace(' alt="The map at step 1 of 11."', ''));
await refuses('step-at', (s) => s.replace('at=1 of=11', 'at=12 of=11'));
await refuses('step-text', (s) => s.replace('{% step %}\nA step can do without a figure.\n{% /step %}', '{% step %}{% /step %}'));
await refuses('mark-order', (s) => s.replace('p10=11 median=17 p90=23', 'p10=18 median=17 p90=23'));
await refuses('different scales', (s) => s.replace('to=1 trait="floor_share" series="c01"', 'to=2 trait="floor_share" series="c01"'));
await refuses('already has that id', (s) => s.replace('{% card title="The second card" %}', '{% card title="The second card" id="first" %}'));
await refuses('card-id', (s) => s.replace('id="first"', 'id="First Card"'));
await refuses('attribute-missing-required', (s) => s.replace('{% table label="The techniques compared" ', '{% table '));
await refuses('interactive-unknown', (s) => s.replace('id="broken"', 'id="absent"'));
// A reserve is a CSS length of a few known parts; anything else could carry more than a length into the page.
for (const bad of ['30em', '20rem; position: fixed', 'url(x)', 'var(--color-ink)', 'min(20rem, 50cqi', 'tall'])
  await refuses('interactive-reserve', (s) => s.replace('reserve="min(16rem + 50cqi, var(--interactive-max-block-size))"', `reserve="${bad}"`));

// No GIFs, by name or by content: one named as a PNG is still refused.
const GIF = new TextEncoder().encode('GIF89a\x01\x00\x01\x00\x00\x00\x00;');
for (const [name, what] of [['assets/clip.gif', 'a .gif'], ['assets/clip.png', 'a GIF named .png']] as const) {
  try {
    await verify(new Map(files).set(name, GIF), { assetBase });
    failures.push(`[asset-gif] ${what} should have been refused`);
  } catch (err) {
    expect(String(err).includes('[asset-gif]'), `[asset-gif] ${what} was refused for another reason: ${String(err).slice(0, 300)}`);
  }
}

// The American English check. The British spellings and the day-first date are put together from
// pieces here, so this file passes the check itself.
const [colorB, neighborsB, centerB] = ['col' + 'our', 'neighb' + 'ours', 'cent' + 're'];
const cap = (w: string) => w[0].toUpperCase() + w.slice(1);
const dayFirst = ['3', 'November'].join(' ');
const place = `${cap(centerB)} Pompidou`;
const found = checkText(`The ${colorB} of rock${cap(neighborsB)},\nat the ${place} on ${dayFirst} 2026. A dialogue, analyzed.`, [place]);
expect(
  JSON.stringify(found.map((f) => [f.line, f.found])) === JSON.stringify([[1, colorB], [1, cap(neighborsB)], [2, dayFirst]]),
  `the American English check found ${JSON.stringify(found)}`,
);
expect(checkBundle(files, page.manifest).length === 0, `the fixture should be in American English: ${checkBundle(files, page.manifest).join('; ')}`);
const british = new Map(files).set(
  'content/index.mdoc',
  new TextEncoder().encode(`${mdoc}\nThe ${colorB} of a tile, with \`${colorB}\` in code.\n\n> A quotation keeps its ${colorB}.\n\n{% figure src="assets/step-1-light.png" alt="The ${centerB}." /%}\n`),
);
const refused = checkBundle(british, page.manifest);
expect(refused.length === 2 && refused[0].startsWith('content/index.mdoc:') && refused[1].includes(centerB), `the British text in a page, outside code and quotations, should be found twice: ${JSON.stringify(refused)}`);

if (failures.length) {
  console.error(`format check failed (${failures.length}):\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('format check passed: the fixture page renders every part, and each mistake is refused');
