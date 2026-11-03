# christianjmills.com, Astro prototype

A prototype of the relaunched site: Astro with Markdoc, built from deliverable bundles that
content projects publish (format spec: `spec/README.md`). It sits beside the Quarto site in this
repo and doesn't touch it. The design tokens are placeholders.

## Build from a clean checkout

Needs Node 26.13.0 (`engines`, enforced by `.npmrc`).

```bash
cd site
npm ci
npm run setup:browser   # the headless Chromium pinned by Playwright, for the checks
npm run build
```

`npm run build` generates the CSS tokens, builds the site with Astro, indexes it with Pagefind,
then fails if a page breaks the performance budgets (`tools/check-perf.ts`) or has an axe
accessibility violation in light or dark, with JavaScript on or off (`tools/check-a11y.ts`).

**Bundles.** The build fetches the bundles pinned in `deliverables.lock.json`. No R2 bucket
exists yet, so the `public` bucket is a local stand-in folder (`deliverables.config.ts`); set
`DELIVERABLES_PUBLIC_URL` to point elsewhere. If the stand-in is missing, recreate it with
`node ../deliverables-prototype/cellular-automata-caves/build.mjs --publish`. The bundle is
deterministic, so it gets the same hash the lockfile pins.

## Write and preview

```bash
npm run dev
```

With `deliverables.local.json` mapping a deliverable to its working tree, the dev server renders
that tree instead of the pinned bundle (spec README, "Publishing and previews").
