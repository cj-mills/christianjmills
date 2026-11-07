// Colour maths for the theme corrections: hex parsing, WCAG 2 contrast, OKLCH and OKLab.
// Ported from the design lab's tools/themes.mjs, which the user ruled on (hub decision
// 2026-11-05-christianjmills-design-system-starting-tokens).

export type Rgb = [number, number, number];
export type Oklch = [l: number, c: number, h: number];

export function parseHex(c: string | undefined): { rgb: Rgb; a: number } | null {
  const m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(c ?? '');
  if (!m) return null;
  const rgb = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16)) as Rgb;
  return { rgb, a: m[2] ? parseInt(m[2], 16) / 255 : 1 };
}
export const isHex = (c: string | undefined) => parseHex(c) !== null;
const toHex = (rgb: number[]) => '#' + rgb.map((v) => v.toString(16).padStart(2, '0')).join('');

/** A colour composited over a background, as opaque 8-bit sRGB. */
export function over(fg: string, bg: string): Rgb {
  const f = parseHex(fg)!, b = parseHex(bg)!;
  return f.rgb.map((v, i) => Math.round(v * f.a + b.rgb[i] * (1 - f.a))) as Rgb;
}
export function luminance(rgb: Rgb): number {
  const c = rgb.map((v) => v / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
/** WCAG 2 contrast of fg on bg, both composited over the page's paper. */
export function contrast(fg: string, bg: string, paper: string = bg): number {
  const b = over(bg, paper);
  const f = over(fg, toHex(b));
  const [x, y] = [luminance(f), luminance(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

const toLin = (v: number) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const fromLin = (v: number) => Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));

export function toOklch(c: string): Oklch {
  const [r, g, b] = parseHex(c)!.rgb.map(toLin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  return [L, Math.hypot(A, B), Math.atan2(B, A)];
}
/** OKLCH to hex, reducing chroma until the colour is inside sRGB (hue and lightness kept). */
export function fromOklch([L, C, H]: Oklch): string {
  for (let c = C; c >= 0; c -= 0.005) {
    const A = c * Math.cos(H), B = c * Math.sin(H);
    const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
    const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
    const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
    const rgb = [
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ];
    if (rgb.every((v) => v >= -1e-4 && v <= 1 + 1e-4)) return toHex(rgb.map((v) => fromLin(Math.min(1, Math.max(0, v)))));
  }
  return L > 0.5 ? '#ffffff' : '#000000';
}
export function oklab(c: string): [number, number, number] {
  const [L, C, H] = toOklch(c);
  return [L, C * Math.cos(H), C * Math.sin(H)];
}
export const distance = (x: string, y: string) => {
  const a = oklab(x), b = oklab(y);
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
};
/** Mix `share` of colour b into colour a, in OKLab. */
export function mix(a: string, b: string, share: number): string {
  const x = oklab(a), y = oklab(b);
  const [L, A, B] = x.map((v, i) => v + (y[i] - v) * share);
  return fromOklch([L, Math.hypot(A, B), Math.atan2(B, A)]);
}
