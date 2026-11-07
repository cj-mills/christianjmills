// The site's typefaces (hub decision 2026-11-07-christianjmills-typefaces): Source Sans 3 for
// text, Literata for heads, Source Code Pro for code. Self-hosted from pinned Fontsource packages
// through Astro's Fonts API, which writes the @font-face rules, the metric-matched fallbacks and the
// preload links. Each role loads only the weights and styles it sets. The Latin and Latin Extended
// subsets are declared with their unicode ranges, so a page fetches Latin Extended only when it
// uses one of its characters.
import type { FontProvider } from 'astro';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

export interface FontRole {
  /** Token name: `font.<role>`, CSS variable `--font-<role>`. */
  role: 'text' | 'head' | 'code';
  family: string;
  package: string;
  /** File-name stem in the package's `files/`. */
  stem: string;
  /** A variable font's weight range ("200 900"), or the static weights the role sets. */
  weights: { weight: string; variable: boolean }[];
  styles: ('normal' | 'italic')[];
  fallbacks: string[];
  /** Whether the Latin roman is preloaded: only the text face's, which every page sets first. */
  preload: boolean;
}

export const FONTS: FontRole[] = [
  {
    role: 'text',
    family: 'Source Sans 3',
    package: '@fontsource-variable/source-sans-3',
    stem: 'source-sans-3',
    weights: [{ weight: '200 900', variable: true }],
    styles: ['normal', 'italic'],
    fallbacks: ['sans-serif'],
    preload: true,
  },
  {
    role: 'head',
    family: 'Literata',
    package: '@fontsource/literata',
    stem: 'literata',
    weights: [{ weight: '600', variable: false }],
    styles: ['normal'],
    fallbacks: ['serif'],
    preload: false,
  },
  {
    role: 'code',
    family: 'Source Code Pro',
    package: '@fontsource/source-code-pro',
    stem: 'source-code-pro',
    // 600 is declared for any bold a syntax theme asks for; it is fetched only when used.
    weights: [{ weight: '400', variable: false }, { weight: '600', variable: false }],
    styles: ['normal'],
    fallbacks: ['monospace'],
    preload: false,
  },
];
export const SUBSETS = ['latin', 'latin-ext'] as const;

const require = createRequire(import.meta.url);
const pkgFile = (pkg: string, file: string) => require.resolve(`${pkg}/package.json`).replace(/package\.json$/, file);
const unicode = (pkg: string): Record<string, string> => JSON.parse(readFileSync(pkgFile(pkg, 'unicode.json'), 'utf8'));
export const packageVersion = (pkg: string): string => JSON.parse(readFileSync(pkgFile(pkg, 'package.json'), 'utf8')).version;
export const packageLicense = (pkg: string): string => readFileSync(pkgFile(pkg, 'LICENSE'), 'utf8');

/** One @font-face per subset, weight and style: the package's file and the subset's unicode range. */
export function faces(f: FontRole) {
  const ranges = unicode(f.package);
  return SUBSETS.flatMap((subset) =>
    f.weights.flatMap(({ weight, variable }) =>
      f.styles.map((style) => ({
        subset,
        weight,
        style,
        unicodeRange: ranges[subset].split(','),
        file: pkgFile(f.package, `files/${f.stem}-${subset}-${variable ? 'wght' : weight}-${style}.woff2`),
      })),
    ),
  );
}

/** An Astro font provider for the pinned packages. Unlike Astro's local provider it records each
 *  face's subset, so a page can preload the Latin roman without the Latin Extended one. */
export function fontsourcePackages(): FontProvider {
  return {
    name: 'fontsource-packages',
    resolveFont({ familyName, weights, styles, subsets }) {
      const f = FONTS.find((x) => x.family === familyName);
      if (!f) return undefined;
      const fonts = faces(f)
        .filter((x) => (!subsets || subsets.includes(x.subset)) && (!styles || styles.includes(x.style)) && (!weights || weights.includes(x.weight)))
        .map((x) => ({
          src: [{ url: x.file, format: 'woff2' }],
          weight: x.weight,
          style: x.style,
          unicodeRange: x.unicodeRange,
          display: 'swap' as const,
          meta: { subset: x.subset },
        }));
      return { fonts };
    },
    listFonts: () => FONTS.map((x) => x.family),
  };
}
