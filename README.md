# christianjmills.com

The source of the relaunched site: Astro with Markdoc, built from deliverable bundles that
content projects publish (format spec: `spec/README.md`). The relaunch is in progress: the design
system's tokens, typefaces, mark and page frame are in; the dialect's parts and the page templates
are not designed yet.

Until the relaunch goes live, christianjmills.com still serves the last Quarto build from the
`gh-pages` branch through GitHub Pages; the relaunched site serves the same build as its frozen copy
(The old site, below). The Quarto sources, the old posts among them, are at the tag `quarto-final`.

## Build from a clean checkout

Needs Node 26.13.0 (`engines`, enforced by `.npmrc`).

```bash
npm ci
npm run setup:browser   # the headless Chromium pinned by Playwright, for the checks
npm run build
```

`npm run build` first checks the repo itself: its TypeScript and Astro types (`astro check`), that
its own text is in American English (`tools/check-english.ts`), that the format's fixture page
renders every part of the dialect and each mistake is refused (`tools/check-spec.ts`), that the
frozen copy's routing does what it should on a made-up old site (`tools/check-legacy.ts`), and, on a
build of that fixture page alone, that interactives mount as the contract says
(`tools/check-fixtures.ts`). Then it generates the CSS tokens, builds the site with Astro, indexes it
with Pagefind, and fails if desktop Chromium would show the fallback `.ico` as a page's tab icon in
place of the SVG (`tools/check-icons.ts`), if a page breaks the performance budgets
(`tools/check-perf.ts`, fonts and interactives included), or if it has an axe accessibility
violation in light or dark, with JavaScript on or off (`tools/check-a11y.ts`). Last, it puts the
frozen copy of the old site around the new one and writes the routing (`tools/archive-merge.ts`),
then serves `dist/` with `wrangler dev` and fails if any address of the old site or the new one
answers other than it should (`tools/check-urls.ts`).

**Design tokens.** `src/styles/tokens.json` is generated and committed, so the user's apps can
read the same tokens. `tools/design/tokens.ts` builds it from a GPUI Kit theme in
`src/styles/themes/` (mapped and corrected by `tools/design/theme.ts`), the typefaces in
`tools/design/fonts.ts`, and the type scale. After changing any of those, run
`npm run tokens:write` and commit the result: the build fails while `tokens.json` is stale. The
fonts are self-hosted from pinned Fontsource packages through Astro's Fonts API. Their licenses and
the theme's are served at `/licenses.txt`.

**Bundles.** The build fetches the bundles pinned in `deliverables.lock.json` from the public R2
bucket at `https://deliverables.christianjmills.com/` and checks each hash. For an offline build,
set `DELIVERABLES_PUBLIC_URL` to a `file:` URL of a folder laid out like the bucket, and
`CHECK_URLS_EXTERNAL=0` to leave out the redirects to GitHub.

**The old site.** christianjmills.com keeps every address the Quarto site had (hub decision
`2026-11-03-christianjmills-legacy-urls`). Its last build, `gh-pages` at `2ddf70cf`, every file of
it, is in the public bucket as an archive in two parts (each under the 315 MB wrangler can upload),
pinned in `legacy/archive.lock.json` and fetched once into `node_modules/.cache/`. At build time:

- each old page is served at its old address with a banner saying it is archived;
- an old page leaves the copy when a deliverable takes its address (`urls.canonical`) or replaces it
  (`urls.supersedes`, a 301);
- Quarto's alias pages become 301s, and the 71 posts retired on October 9, 2026 send their readers to
  their new homes (`legacy/retired.json`); their images and videos stay where they were;
- a file the new site builds at an old file's path fails the build, unless the deliverable names it.

The Worker serves files only by their names (`html_handling: "none"` in `wrangler.jsonc`), so
`/about.html` and `/posts/x/index.html` answer 200 where they did on GitHub Pages; the folders and
slashes come from the rules the build writes into `dist/_redirects`. The archive was packed once by
`tools/archive-pack.ts` and is never packed again. An offline build needs its parts in the local
folder `DELIVERABLES_PUBLIC_URL` names, beside the bundles.

**Feed and sitemaps.** `/blog.xml` (the old feed's address), `/sitemap.xml` and `/robots.txt` are
generated from the manifests; `/sitemap-archive.xml` lists the old pages the copy still serves.

**Comments.** Each post ends with a "Show comments" button that loads giscus from giscus.app only when
pressed. The thread is a GitHub Discussion in this repo's Comments category, titled with the
deliverable's id, or the older discussion its manifest names (`comments.discussion`). Without
scripts, the post links to the thread on GitHub.

## Cloudflare

Commands that touch Cloudflare run through `tools/cloudflare.sh`, which has 1Password inject a
scoped API token and the account ID for the length of one command (see the script's header for
the env file it reads). Wrangler is pinned in `package.json`, and its telemetry is off.

```bash
npm run cf -- npx wrangler whoami                       # check the credentials
npm run deploy:upload                                   # build, check, upload a version, check its preview URL
npm run deploy:promote -- <version id>                  # send the Worker's traffic to that version
sh tools/cloudflare.sh node tools/publish-deliverable.ts <deliverable-dir> --r2
```

The site deploys as an assets-only Worker (`wrangler.jsonc`): no Worker script runs. Deploys go
through Worker versions (`tools/deploy.ts`): `deploy:upload` uploads the committed build as a new
version, which changes nothing anyone is served, and runs the URL check at the version's own preview
URL; `deploy:promote` then sends the Worker's traffic to a version this checkout uploaded and
checked. Until christianjmills.com is attached to the Worker, that traffic is only the preview's on
workers.dev. The workers.dev hosts answer with `X-Robots-Tag: noindex` (`public/_headers`); the live
domain doesn't match that rule.

A GitHub workflow (`.github/workflows/build.yml`) runs `npm run build` on every push to `main` and
every pull request, with no secrets.

## Write and preview

```bash
npm run dev
```

With `deliverables.local.json` mapping a deliverable to its working tree, the dev server renders
that tree instead of the pinned bundle (spec README, "Publishing and previews").

## License

The code in this repo is under the MIT License (`LICENSE`). Each deliverable carries its own
prose and code licenses in its manifest, and the site shows them on the page.
