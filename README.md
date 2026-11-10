# christianjmills.com

The source of the relaunched site: Astro with Markdoc, built from deliverable bundles that
content projects publish (format spec: `spec/README.md`). The relaunch is in progress: the design
system's tokens, typefaces, mark and page frame are in; the dialect's parts and the page templates
are not designed yet.

Until the relaunch goes live, christianjmills.com still serves the last Quarto build from the
`gh-pages` branch. The Quarto sources, the old posts among them, are at the tag `quarto-final`.

## Build from a clean checkout

Needs Node 26.13.0 (`engines`, enforced by `.npmrc`).

```bash
npm ci
npm run setup:browser   # the headless Chromium pinned by Playwright, for the checks
npm run build
```

`npm run build` first checks the repo itself: that its own text is in American English
(`tools/check-english.ts`), that the format's fixture page renders every part of the dialect and
each mistake is refused (`tools/check-spec.ts`), and, on a build of that fixture page alone, that
interactives mount as the contract says (`tools/check-fixtures.ts`). Then it generates the CSS
tokens, builds the site with Astro, indexes it with Pagefind, and fails if desktop Chromium would
show the fallback `.ico` as a page's tab icon in place of the SVG (`tools/check-icons.ts`), if a
page breaks the performance budgets (`tools/check-perf.ts`, fonts and interactives included), or
if it has an axe accessibility violation in light or dark, with JavaScript on or off
(`tools/check-a11y.ts`).

**Design tokens.** `src/styles/tokens.json` is generated and committed, so the user's apps can
read the same tokens. `tools/design/tokens.ts` builds it from a GPUI Kit theme in
`src/styles/themes/` (mapped and corrected by `tools/design/theme.ts`), the typefaces in
`tools/design/fonts.ts`, and the type scale. After changing any of those, run
`npm run tokens:write` and commit the result: the build fails while `tokens.json` is stale. The
fonts are self-hosted from pinned Fontsource packages through Astro's Fonts API. Their licenses and
the theme's are served at `/licenses.txt`.

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
