// Deliverable manifest, format 0.1 (draft). The site repo owns this schema; content projects
// write `manifest.json` at the root of each bundle to match it. Each field group serves a
// requirement in the deliverable contract (hub decision 2026-11-03-site-deliverable-contract).
import { z } from 'zod';
import categories from './categories.json' with { type: 'json' };

export const FORMAT_VERSION = '0.1';

const slug = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'lowercase words joined by hyphens');
const isoDate = z.iso.date();
const semver = z.string().regex(/^\d+\.\d+\.\d+$/, 'MAJOR.MINOR.PATCH');
const sitePath = z.string().regex(/^\/(?:[a-z0-9-]+\/)+$/, 'an absolute path with a trailing slash, e.g. /procgen/caves/');
const bundlePath = z.string().regex(/^(?!\/)(?!.*\.\.)[\w./-]+$/, 'a relative path inside the bundle');
const gitCommit = z.string().regex(/^[0-9a-f]{40}$/, 'a full 40-character commit hash');

const allCategories: string[] = Object.values(categories.groups).flatMap((g) => g.terms);
const category = z.string().refine((c) => allCategories.includes(c), {
  message: `not in the site's category vocabulary (spec/categories.json)`,
});

// A source a page can cite. `kind` decides which locators a reference may use (see dialect.ts).
const sourceBase = {
  id: slug.refine((id) => id !== 'this', '"this" is reserved for the bundle’s own files'),
  label: z.string().min(1).max(40).describe('Short name shown inline in references'),
  citation: z.string().min(1),
  rights: z.string().min(1).describe('What the page may reproduce from the source, and under which license'),
};
export const source = z.discriminatedUnion('kind', [
  z.object({ ...sourceBase, kind: z.literal('video'), url: z.url() }),
  z.object({ ...sourceBase, kind: z.literal('document'), url: z.url(), pages: z.int().positive() }),
  z.object({
    ...sourceBase,
    kind: z.literal('code'),
    repo: z.url(),
    commit: gitCommit,
  }),
  z.object({ ...sourceBase, kind: z.literal('web'), url: z.url() }),
  z.object({ ...sourceBase, kind: z.literal('image'), path: bundlePath, width: z.int().positive(), height: z.int().positive() }),
]);
export type Source = z.infer<typeof source>;

const license = z.object({ spdx: z.string().min(1), url: z.url().optional() });

export const manifest = z.strictObject({
  format: z.literal(FORMAT_VERSION),

  // identity
  id: slug.describe('Permanent; never reused'),
  version: semver,
  title: z.string().min(1).max(120),
  summary: z.string().min(1).max(300),
  pageType: z.enum(['post', 'tutorial', 'report', 'paper', 'project-log', 'case-study', 'lecture-resource']),
  project: z.string().min(1).describe('The source project, as named in the hub'),

  // URLs
  urls: z.strictObject({
    canonical: sitePath,
    supersedes: z.array(z.string().startsWith('/')).default([]),
  }),

  // dates
  published: isoDate,
  revisions: z
    .array(z.strictObject({ date: isoDate, version: semver, note: z.string().min(1) }))
    .min(1)
    .describe('Newest last; the last date is the page’s “updated” date'),

  // tested against
  testedAgainst: z
    .array(z.strictObject({ name: z.string(), version: z.string(), date: isoDate }))
    .default([]),
  hardware: z.array(z.string()).default([]),

  categories: z.array(category).min(1),

  sources: z.array(source).default([]),

  // How a traced page was checked, for the "how this was checked" note under its Sources (hub
  // decision 2026-11-04-traced-deliverables-reader-view). Written by the project's build from its
  // evidence; a snapshot of when the bundle was built.
  checked: z
    .strictObject({
      claims: z.int().nonnegative().describe('Claims that cite a source; the verifier checks each against the passages it cites'),
      supported: z.int().nonnegative().describe('Of those, how many the verifier found supported'),
      measured: z.int().nonnegative().describe('Statements of the deliverable’s own measurements, regenerated and checked by its build'),
      verifiers: z
        .array(z.strictObject({ name: z.string().min(1), version: z.string().min(1), model: z.string().min(1).optional() }))
        .min(1),
      lastVerified: isoDate,
      openFlags: z.int().nonnegative().describe('Flags open when the bundle was built: a changed source, a missing or failed verdict'),
    })
    .refine((c) => c.supported <= c.claims, 'more claims supported than checked')
    .optional(),

  // The names and quotations that keep their own spelling. The publish command refuses British
  // spellings and day-first dates in everything else a reader sees (spec/english.ts).
  english: z.strictObject({ allow: z.array(z.string().min(1)).default([]) }).optional(),

  // provenance of generated media
  provenance: z
    .array(
      z.strictObject({
        asset: bundlePath,
        generator: z.string().describe('A model, or the deliverable’s own code'),
        prompt: z.string().optional(),
        handEdits: z.string().optional(),
      }),
    )
    .default([]),

  // Comments: a GitHub Discussion in the site repo's Comments category, titled with the id, unless
  // the page continues an older thread, named here by its number (src/lib/comments.ts).
  comments: z.strictObject({ discussion: z.int().positive().optional() }).optional(),

  tier: z.strictObject({
    access: z.enum(['free', 'paid']),
    agents: z.enum(['allowed', 'paid', 'denied']),
  }),

  licenses: z.strictObject({ prose: license, code: license }),

  code: z.strictObject({ repo: z.url(), commit: gitCommit.optional() }).optional(),

  formats: z
    .array(z.enum(['html', 'markdown', 'pdf']))
    .refine((f) => f.includes('html') && f.includes('markdown'), 'html and markdown are always produced'),

  content: z.strictObject({ entry: bundlePath.regex(/\.mdoc$/, 'a .mdoc file') }),
  interactives: z
    .array(
      z.strictObject({
        id: slug,
        module: bundlePath.regex(/\.m?js$/, 'an ES module'),
        description: z.string().min(1),
      }),
    )
    .default([]),
});
export type Manifest = z.infer<typeof manifest>;

// The site's lockfile: which deliverable versions the build includes, where their bundles are,
// and the hash each must match.
export const lockfile = z.strictObject({
  lockfileVersion: z.literal(1),
  deliverables: z.record(
    slug,
    z.strictObject({
      version: semver,
      format: z.literal(FORMAT_VERSION),
      bucket: z.enum(['public', 'private']),
      key: z.string().regex(/^sha256\/[0-9a-f]{64}\.tar\.gz$/),
      sha256: z.string().regex(/^[0-9a-f]{64}$/),
      size: z.int().positive(),
    }),
  ),
});
export type Lockfile = z.infer<typeof lockfile>;

// Bundle limits, from the performance targets (provisional; see tools/check-perf.ts for the page budgets).
export const BUNDLE_LIMITS = {
  maxCompressedBytes: 25 * 1024 * 1024, // Cloudflare static assets: 25 MiB per file
  maxFiles: 500,
  maxFileBytes: 5 * 1024 * 1024, // larger media is referenced by hash, not shipped in the bundle
};
