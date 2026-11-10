// Builds src/styles/tokens.json, the design system's tokens in a subset of the W3C Design Tokens
// (DTCG) format, for the site and for the user's apps. It is generated and committed: apps read one
// stable file, and `npm run tokens` fails the build when the file is not what this produces.
//
//   node tools/design/tokens.ts          write src/styles/tokens.json
//   node tools/design/tokens.ts --check  exit 1 if it is stale
//
// Sources, each ruled by the user (hub decisions):
//   colors ... a GPUI Kit theme, mapped and corrected by theme.ts
//               (2026-11-07-christianjmills-default-theme: Ayu Light and Ayu Dark)
//   faces ..... fonts.ts (2026-11-07-christianjmills-typefaces)
//   scale ..... a 1.2 ratio and a 4 px radius (2026-11-05-christianjmills-design-system-starting-tokens)
//   space ..... multiples of the body size (2026-11-08-christianjmills-space-scale)
//   measure ... 68ch (2026-11-08-christianjmills-measure)
import { readFileSync, writeFileSync } from 'node:fs';
import { FONTS, packageVersion } from './fonts.ts';
import { contrast, mix } from './color.ts';
import { correctTheme, mapTheme, type GpuiTheme, type Roles } from './theme.ts';

export const THEME = { file: 'ayu.json', light: 'Ayu Light', dark: 'Ayu Dark' };
const RATIO = 1.2;
const RADIUS = '4px';
// The body size runs fluidly from 1rem at a 360 px viewport to 1.125rem at 1200 px; every step is
// the body size times RATIO^n, at both ends. The root font size is never set, so rem stays the
// reader's own default and nothing compounds.
const BODY = { min: 1, max: 1.125, from: 360, to: 1200 };
const STEPS = [-2, -1, 0, 1, 2, 3, 4, 5];

const root = new URL('../../', import.meta.url);
export const TOKENS_FILE = new URL('src/styles/tokens.json', root);
const themeFile = new URL(`src/styles/themes/${THEME.file}`, root);

type Token = { $type: string; $value: unknown; $description?: string; $extensions?: Record<string, unknown> };
type Group = { [k: string]: Token | Group | string | Record<string, unknown> };

const round = (x: number, d = 4) => Number(x.toFixed(d));
/** A length that runs linearly from `min` rem at BODY.from to `max` rem at BODY.to, held at both ends. */
function fluid(min: number, max: number): string {
  const slope = ((max - min) * 16) / (BODY.to - BODY.from); // px per px of viewport
  const intercept = (min * 16 - slope * BODY.from) / 16; // rem
  return `clamp(${round(min)}rem, ${round(intercept)}rem + ${round(slope * 100)}vw, ${round(max)}rem)`;
}
const step = (n: number) => fluid(BODY.min * RATIO ** n, BODY.max * RATIO ** n);

// Space is measured in body sizes, so every gap grows with the text it separates: each step is a
// multiple of the body size at both ends of its range. A pair ("s-l") runs from the smaller step's
// size at 360 px to the larger's at 1200 px, for gaps that should open up faster than the text,
// such as the page's side margin. The apps read min and max from the extension, since clamp() and vw
// are the web's.
const SPACE: Record<string, number> = { '3xs': 0.25, '2xs': 0.5, xs: 0.75, s: 1, m: 1.5, l: 2, xl: 3, '2xl': 4, '3xl': 6 };
const SPACE_PAIRS: [string, string][] = [
  ...Object.keys(SPACE).slice(1).map((k, i) => [Object.keys(SPACE)[i], k] as [string, string]),
  ['s', 'l'],
];
function spaceToken(lo: string, hi = lo): Token {
  const min = BODY.min * SPACE[lo], max = BODY.max * SPACE[hi];
  return {
    $type: 'dimension',
    $value: fluid(min, max),
    $extensions: { 'com.christianjmills.fluid': { min: `${round(min)}rem`, max: `${round(max)}rem`, from: `${BODY.from}px`, to: `${BODY.to}px` } },
  };
}

function colorGroup(light: Roles, dark: Roles, rawLight: Roles, rawDark: Roles): Group {
  const out: Group = { code: {}, syntax: {} };
  for (const role of Object.keys(light)) {
    const token: Token = {
      $type: 'color',
      $value: { light: light[role], dark: dark[role] },
    };
    // The theme's own value before correction; derived roles (the callout fills) have none.
    if (role in rawLight) token.$extensions = { 'com.christianjmills.theme': { light: rawLight[role], dark: rawDark[role] } };
    if (role.startsWith('code-')) (out.code as Group)[role.slice(5)] = token;
    else if (role.startsWith('syntax-')) (out.syntax as Group)[role.slice(7)] = token;
    else out[role] = token;
  }
  return out;
}

/** A pair that keeps its sense in both themes, for a drawing where light and dark mean something
 *  (on a map, floor is the light one and wall the dark one): paper and a softened ink, whichever
 *  way round the theme has them. A figure drawn in ink and paper inverts with the theme; one drawn
 *  in this pair doesn't. */
function figurePair(light: Roles, dark: Roles): Group {
  const pair = {
    light: { light: light.paper, dark: mix(dark.ink, dark.paper, 0.2) },
    dark: { light: mix(light.ink, light.paper, 0.12), dark: dark.paper },
  };
  for (const mode of ['light', 'dark'] as const)
    if (contrast(pair.dark[mode], pair.light[mode]) < 4.5) throw new Error(`figure pair: ${pair.dark[mode]} on ${pair.light[mode]} (${mode}) is below 4.5:1`);
  return {
    $description: 'A light and a dark that keep their sense in both themes, for drawings where they mean something: on a map, floor is the light one and wall the dark one.',
    light: { $type: 'color', $value: pair.light },
    dark: { $type: 'color', $value: pair.dark },
  };
}

export function buildTokens(): Group {
  const fam = JSON.parse(readFileSync(themeFile, 'utf8')) as { name: string; themes: GpuiTheme[] };
  const variant = (name: string) => {
    const t = fam.themes.find((x) => x.name === name);
    if (!t) throw new Error(`${THEME.file} has no variant "${name}"`);
    return t;
  };
  const [L, D] = [variant(THEME.light), variant(THEME.dark)];
  if (L.mode !== 'light' || D.mode !== 'dark') throw new Error(`${THEME.light} must be light and ${THEME.dark} dark`);
  const light = correctTheme(L).roles, dark = correctTheme(D).roles;

  return {
    $description:
      'GENERATED by tools/design/tokens.ts; do not edit. The design system of christianjmills.com and the apps that share its identity. ' +
      'Colors: GPUI Kit theme roles, AA-, hierarchy- and accent-corrected, with neutral rules and text; each color token has a value per scheme, ' +
      'and its com.christianjmills.theme extension holds the theme value before correction.',
    $extensions: {
      'com.christianjmills.source': {
        theme: `src/styles/themes/${THEME.file}`,
        light: THEME.light,
        dark: THEME.dark,
        fonts: Object.fromEntries(FONTS.map((f) => [f.package, packageVersion(f.package)])),
      },
    },
    color: colorGroup(light, dark, mapTheme(L), mapTheme(D)),
    interactive: {
      $description: 'What an interactive module reads to draw (spec/README.md): ink and surface.',
      ink: { $type: 'color', $value: '{color.ink}' },
      paper: { $type: 'color', $value: '{color.surface}' },
    },
    figure: figurePair(light, dark),
    font: Object.fromEntries(
      FONTS.map((f) => [
        f.role,
        {
          $type: 'fontFamily',
          $value: [f.family, ...f.fallbacks],
          $extensions: {
            'com.christianjmills.font': { package: f.package, weights: f.weights.map((w) => w.weight), styles: f.styles },
          },
        },
      ]),
    ),
    size: {
      $description: `A ${RATIO} ratio on a body size that runs from ${BODY.min}rem at ${BODY.from} px to ${BODY.max}rem at ${BODY.to} px.`,
      ...Object.fromEntries(STEPS.map((n) => [`step-${n}`, { $type: 'dimension', $value: step(n) }])),
    },
    space: {
      $description: `Multiples of the body size (${Object.entries(SPACE).map(([k, v]) => `${k} ${v}`).join(', ')}), fluid as the body size is; a pair runs from its first step's size at ${BODY.from} px to its second's at ${BODY.to} px.`,
      ...Object.fromEntries(Object.keys(SPACE).map((k) => [k, spaceToken(k)])),
      ...Object.fromEntries(SPACE_PAIRS.map(([lo, hi]) => [`${lo}-${hi}`, spaceToken(lo, hi)])),
    },
    measure: { $type: 'dimension', $value: '68ch', $description: 'The longest line of running text, in the text face at body size.' },
    radius: { $type: 'dimension', $value: RADIUS },
  };
}

/** JSON with each token on one line, so a diff shows one changed token as one changed line. */
export function render(node: unknown, indent = ''): string {
  if (node && typeof node === 'object' && !Array.isArray(node) && !('$type' in node)) {
    const inner = indent + '  ';
    const entries = Object.entries(node as Record<string, unknown>).map(([k, v]) => `${inner}${JSON.stringify(k)}: ${k.startsWith('$') ? JSON.stringify(v) : render(v, inner)}`);
    return `{\n${entries.join(',\n')}\n${indent}}`;
  }
  return JSON.stringify(node);
}
export const expected = () => render(buildTokens()) + '\n';

if (import.meta.url === `file://${process.argv[1]}`) {
  const want = expected();
  if (process.argv.includes('--check')) {
    const have = readFileSync(TOKENS_FILE, 'utf8');
    if (have !== want) {
      console.error('src/styles/tokens.json is stale: run `npm run tokens:write` and commit it.');
      process.exit(1);
    }
    console.log('tokens.json is current');
  } else {
    writeFileSync(TOKENS_FILE, want);
    console.log(`wrote ${TOKENS_FILE.pathname}`);
  }
}
