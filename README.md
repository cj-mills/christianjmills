# christianjmills.com

The source of the relaunched site: Astro with Markdoc, built from deliverable bundles that
content projects publish (format spec: `spec/README.md`). The relaunch is in progress, and the
design tokens are placeholders.

Until the relaunch goes live, christianjmills.com still serves the last Quarto build from the
`gh-pages` branch. The Quarto sources, the old posts among them, are at the tag `quarto-final`.

## Build from a clean checkout

Needs Node 26.13.0 (`engines`, enforced by `.npmrc`).

```bash
npm ci
npm run setup:browser   # the headless Chromium pinned by Playwright, for the checks
npm run build
```

`npm run build` generates the CSS tokens, builds the site with Astro, indexes it with Pagefind,
then fails if a page breaks the performance budgets (`tools/check-perf.ts`) or has an axe
accessibility violation in light or dark, with JavaScript on or off (`tools/check-a11y.ts`).

**Bundles.** The build fetches the bundles pinned in `deliverables.lock.json` from the public R2
bucket at `https://deliverables.christianjmills.com/` and checks each hash. For an offline build,
set `DELIVERABLES_PUBLIC_URL` to a `file:` URL of a folder laid out like the bucket.

## Cloudflare

Commands that touch Cloudflare run through `tools/cloudflare.sh`, which has 1Password inject a
scoped API token and the account ID for the length of one command (see the script's header for
the env file it reads). Wrangler is pinned in `package.json`, and its telemetry is off.

```bash
npm run cf -- npx wrangler whoami                       # check the credentials
npm run deploy:preview                                  # build, check, deploy dist/ to workers.dev
sh tools/cloudflare.sh node tools/publish-deliverable.ts <deliverable-dir> --r2
```

The site deploys as an assets-only Worker (`wrangler.jsonc`): no Worker script runs, and no route
or custom domain points at it.

## Write and preview

```bash
npm run dev
```

With `deliverables.local.json` mapping a deliverable to its working tree, the dev server renders
that tree instead of the pinned bundle (spec README, "Publishing and previews").

## License

The code in this repo is under the MIT License (`LICENSE`). Each deliverable carries its own
prose and code licenses in its manifest, and the site shows them on the page.
