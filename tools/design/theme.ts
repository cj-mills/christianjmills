// Maps a GPUI Kit theme variant to the site's colour roles and applies the ruled corrections. Every
// rule here was ruled by the user; the hub decision that holds each is named beside it:
//   mapping .................. 2026-11-07-christianjmills-theme-role-mapping
//   AA, hierarchy, accents ... 2026-11-05-christianjmills-design-system-starting-tokens
//   neutral rules and text ... 2026-11-07-christianjmills-neutral-rules-and-text
// The corrections were developed in the design lab's tools/themes.mjs, which this ports.
import { contrast, distance, fromOklch, isHex, luminance, mix, over, parseHex, toOklch } from './color.ts';

export type Mode = 'light' | 'dark';
export interface GpuiTheme {
  name: string;
  mode: Mode;
  colors: Record<string, string>;
  highlight?: Record<string, unknown> & { syntax?: Record<string, { color?: string }> };
}
export type Roles = Record<string, string>;

// ---- mapping: role <- the first GPUI Kit key the theme has -----------------------------------------
const MAP: Record<string, string[]> = {
  paper: ['background'],
  surface: ['secondary.background', 'muted.background'],
  ink: ['foreground'],
  muted: ['muted.foreground'],
  rule: ['border'],
  link: ['link.foreground', 'primary.background'],
  primary: ['primary.background'],
  'primary-ink': ['primary.foreground'],
  tint: ['accent.background'],
  'tint-ink': ['accent.foreground'],
  focus: ['ring', 'primary.background'],
  red: ['base.red'],
  green: ['base.green'],
  blue: ['base.blue'],
  yellow: ['base.yellow'],
  cyan: ['base.cyan'],
  magenta: ['base.magenta'],
};
// Code: highlight's editor keys, else surface and ink.
const CODE: Record<string, [key: string, fallback: string]> = {
  'code-bg': ['editor.background', 'surface'],
  'code-ink': ['editor.foreground', 'ink'],
  'code-line-number': ['editor.line_number', 'muted'],
};
// Syntax tokens <- highlight.syntax keys (first present wins; else the code ink).
const SYNTAX: Record<string, string[]> = {
  constant: ['constant', 'number'],
  string: ['string'],
  comment: ['comment'],
  keyword: ['keyword'],
  parameter: ['variable', 'property'],
  function: ['function'],
  'string-expression': ['string.special', 'string'],
  punctuation: ['punctuation'],
  link: ['link_uri'],
};
/** Callout fills: a base colour mixed into paper (the theme-role-mapping decision). */
const CALLOUT: Record<string, [base: string, share: number]> = { note: ['blue', 0.12], warning: ['yellow', 0.12] };

// ---- the pairs every corrected theme must pass (WCAG 2; text 4.5:1, focus ring 3:1) ------------------
export const PAIRS: [fg: string, bg: string, min: number][] = [
  ['ink', 'paper', 4.5], ['ink', 'surface', 4.5], ['ink', 'note', 4.5], ['ink', 'warning', 4.5],
  ['muted', 'paper', 4.5], ['muted', 'surface', 4.5],
  ['link', 'paper', 4.5], ['link', 'surface', 4.5], ['link', 'note', 4.5], ['link', 'warning', 4.5],
  ['primary-ink', 'primary', 4.5], ['tint-ink', 'tint', 4.5],
  ['code-ink', 'code-bg', 4.5],
  ...Object.keys(SYNTAX).map((t): [string, string, number] => ['syntax-' + t, 'code-bg', 4.5]),
  ['focus', 'paper', 3],
];
const MARGIN = 0.05; // every pair passes with this margin over its minimum

// ---- corrections ------------------------------------------------------------------------------------
const clamp = (x: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, x));
const L = (c: string) => toOklch(c)[0];

/** AA: a failing foreground's lightness moves away from its background, hue (and chroma where the
 *  gamut allows) kept, until every pair it appears in passes. Backgrounds never change. */
function correctAA(r: Roles): Roles {
  const c = { ...r };
  for (const fg of new Set(PAIRS.map((p) => p[0]))) {
    const mine = PAIRS.filter((p) => p[0] === fg);
    const ok = () => mine.every(([f, b, min]) => contrast(c[f], c[b], c.paper) >= min + MARGIN);
    if (ok() || parseHex(c[fg])!.a < 1) continue;
    const [L0, C, H] = toOklch(c[fg]);
    // Away from the background first; if that runs out of range before passing, towards and past it.
    const bg = luminance(over(c[mine[0][1]], c.paper)), self = luminance(over(c[fg], c.paper));
    const away = self === bg ? (bg > 0.18 ? -1 : 1) : Math.sign(self - bg);
    for (const dir of [away, -away]) {
      let passed = false;
      for (let l = L0; l >= 0 && l <= 1; l += dir * 0.005) {
        c[fg] = fromOklch([l, C, H]);
        if ((passed = ok())) break;
      }
      if (passed) break;
    }
  }
  return c;
}

// Hierarchy: minimum OKLCH lightness gaps, so every theme keeps the same hierarchy. Surface moves
// away from paper (paper never moves); ink moves away from paper until muted sits clearly apart.
const GAP: Record<Mode, { surface: number; inkMuted: number }> = {
  light: { surface: 0.035, inkMuted: 0.15 },
  dark: { surface: 0.07, inkMuted: 0.15 },
};
function moveAway(c: string, from: string, min: number): string {
  const [l, C, H] = toOklch(c), lf = L(from);
  if (Math.abs(l - lf) >= min) return c;
  const dir = l === lf ? (lf > 0.5 ? -1 : 1) : Math.sign(l - lf);
  return fromOklch([Math.min(1, Math.max(0, lf + dir * min)), C, H]);
}
function inkApart(c: Roles, mode: Mode): Roles {
  const lm = L(c.muted), lp = L(c.paper);
  let [l, C, H] = toOklch(c.ink);
  const away = Math.sign(lm - lp) || (lp > 0.5 ? -1 : 1);
  while (Math.abs(l - lm) < GAP[mode].inkMuted && l > 0 && l < 1) l += away * 0.005;
  return { ...c, ink: fromOklch([Math.min(1, Math.max(0, l)), C, H]) };
}

// Accents: primary and link keep their hue, with lightness and chroma held to one band per mode.
const ACCENT: Record<Mode, { L: [number, number]; C: [number, number] }> = {
  light: { L: [0.42, 0.56], C: [0.07, 0.13] },
  dark: { L: [0.7, 0.8], C: [0.07, 0.13] },
};
function normaliseAccent(c: string, mode: Mode): string {
  const [l, C, H] = toOklch(c);
  const chroma = C < 0.02 ? C : clamp(C, ACCENT[mode].C); // a near-grey accent stays grey
  return fromOklch([clamp(l, ACCENT[mode].L), chroma, H]);
}
// Links stand apart from body text: if the OKLab distance is under LINK_MIN, ink moves away from
// paper until it clears, and failing that the link takes the band's full chroma.
const LINK_MIN = 0.14;
function linkApart(c: Roles, mode: Mode): Roles {
  const out = { ...c };
  let [l, C, H] = toOklch(out.ink);
  const away = L(out.paper) > 0.5 ? -1 : 1;
  while (distance(out.link, out.ink) < LINK_MIN && l > 0 && l < 1) {
    l += away * 0.005;
    out.ink = fromOklch([Math.min(1, Math.max(0, l)), C, H]);
  }
  if (distance(out.link, out.ink) < LINK_MIN) {
    const [ll, , lh] = toOklch(out.link);
    out.link = fromOklch([ll, ACCENT[mode].C[1], lh]);
  }
  return out;
}

// Neutral rules and text (2026-11-07-christianjmills-neutral-rules-and-text): every rule at one
// WCAG contrast against paper in both modes, near neutral; ink and muted near grey.
const RULE_CONTRAST = 1.4;
const RULE_CHROMA = 0.02;
const TEXT_CHROMA = { ink: 0.02, muted: 0.025 };
function fixedRule(rule: string, paper: string): string {
  const [, C, H] = toOklch(rule);
  const lp = L(paper), dir = lp > 0.5 ? -1 : 1;
  let out = rule;
  for (let l = lp; l >= 0 && l <= 1; l += dir * 0.002) {
    out = fromOklch([l, Math.min(C, RULE_CHROMA), H]);
    if (contrast(out, paper) >= RULE_CONTRAST) break;
  }
  return out;
}
function capChroma(c: string, max: number): string {
  const [l, C, H] = toOklch(c);
  return C <= max ? c : fromOklch([l, max, H]);
}

// ---- the pipeline -----------------------------------------------------------------------------------
/** The theme's roles as GPUI Kit gives them, before any correction. */
export function mapTheme(t: GpuiTheme): Roles {
  const r: Roles = {};
  for (const [role, keys] of Object.entries(MAP)) {
    const v = keys.map((k) => t.colors[k]).find(isHex);
    if (!v) throw new Error(`${t.name}: no colour for role "${role}" (keys ${keys.join(', ')})`);
    r[role] = v;
  }
  const h = (t.highlight ?? {}) as Record<string, unknown>;
  for (const [role, [key, fallback]] of Object.entries(CODE)) r[role] = isHex(h[key] as string) ? (h[key] as string) : r[fallback];
  const syn = t.highlight?.syntax ?? {};
  for (const [tok, keys] of Object.entries(SYNTAX)) r['syntax-' + tok] = keys.map((k) => syn[k]?.color).find(isHex) ?? r['code-ink'];
  // Tokens are opaque lowercase #rrggbb: translucent theme values are composited over paper.
  const paper = r.paper;
  for (const k of Object.keys(r)) r[k] = '#' + over(r[k], paper).map((v) => v.toString(16).padStart(2, '0')).join('');
  return r;
}

/** The corrected roles, and every pair's contrast. Throws if a pair still fails. */
export function correctTheme(t: GpuiTheme): { roles: Roles; pairs: [string, string, number, number][] } {
  const raw = mapTheme(t);
  let c: Roles = { ...raw };
  // Hierarchy and accents first, then the neutral rules and text.
  c.primary = normaliseAccent(c.primary, t.mode);
  c.link = normaliseAccent(c.link, t.mode);
  c.surface = moveAway(c.surface, c.paper, GAP[t.mode].surface);
  c.rule = fixedRule(c.rule, c.paper);
  c.ink = capChroma(c.ink, TEXT_CHROMA.ink);
  c.muted = capChroma(c.muted, TEXT_CHROMA.muted);
  for (const [role, [base, share]] of Object.entries(CALLOUT)) c[role] = mix(c.paper, c[base], share);
  // Then AA, ink apart from the corrected muted, links apart from ink, and AA once more.
  c = correctAA(linkApart(inkApart(correctAA(c), t.mode), t.mode));
  const pairs = PAIRS.map(([f, b, min]): [string, string, number, number] => [f, b, contrast(c[f], c[b], c.paper), min]);
  const failing = pairs.filter(([, , x, min]) => x < min);
  if (failing.length) throw new Error(`${t.name}: still failing after correction: ${failing.map(([f, b, x]) => `${f} on ${b} ${x.toFixed(2)}`).join('; ')}`);
  return { roles: c, pairs };
}
