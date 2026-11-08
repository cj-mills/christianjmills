# Deliverable format 0.1 (draft)

The contract between content projects and the site (hub decision
`2026-11-03-site-deliverable-contract`). A project hands the site a versioned bundle; the site
pins it in a lockfile, verifies it and renders it. This folder is the spec, and its code is
what enforces it:

| File | Defines |
|---|---|
| `manifest.ts` | the manifest schema, the lockfile schema, bundle limits |
| `categories.json` | the category vocabulary (placeholder) |
| `dialect.ts` | the Markdoc dialect: tags, node rules, validation |
| `bundle.ts` | bundle layout, packing, unpacking, verification |
| `markdown.ts` | the Markdown copy for agents |

Draft status: written for the Astro prototype. Field names and tags may change before 1.0.

## Bundle

A deterministic `tar.gz` (entries sorted, mtime 0, no owner) containing only:

```
manifest.json      the manifest
content/…          Markdoc; manifest.content.entry names the page
assets/…           images, interactive modules, files the page serves
```

Limits: 25 MiB compressed, 500 files, 5 MiB per file (`BUNDLE_LIMITS`). Bundles are stored
content-addressed: `sha256/<hex>.tar.gz` in the `public` bucket (free deliverables) or the
`private` bucket (paid).

## Manifest

See `manifest.ts` for types and constraints. Groups: identity (`id`, `version`, `title`,
`summary`, `pageType`, `project`); `urls` (`canonical`, `supersedes`); dates (`published`,
`revisions` with a note each, the last matching `version`); `testedAgainst` and `hardware`;
`categories` from the vocabulary; `sources`; `provenance` of generated media; `tier`;
`licenses` (prose and code apart); `code`; `formats`; `content`; `interactives`.

Source kinds and the locators a reference to each may use:

| kind | fields | locator attributes |
|---|---|---|
| `video` | `url` | `t="mm:ss"` or `t="mm:ss-mm:ss"` (required) |
| `document` | `url`, `pages` | `page="9"` or `page="9-10"` (required, checked against `pages`) |
| `code` | `repo`, `commit` | `path`, `lines="40-54"` (required) |
| `web` | `url` | `anchor` (optional) |
| `image` | bundle `path`, `width`, `height` | `region="x,y,w,h"` (required) |
| `this` (reserved, not declared) | the bundle's own files | `path`, `lines` (required, checked against the file) |

## Dialect

Markdoc, plus:

- `{% ref source="id" …locator /%}`: inline reference to a manifest source. Unknown source, a
  locator the kind doesn't take, a malformed or out-of-range locator: build error. Write one per
  claim. The HTML merges a block's references (paragraph, list item, table cell) into one marker
  at its end; the Markdown copy keeps them one per claim.
- `{% interactive id="id" caption="…" %}fallback{% /interactive %}`: a mount for a declared
  interactive. The children are the fallback shown without JavaScript and are required.
- `{% figure src="assets/…" alt="…" caption="…" /%}` and `![alt](assets/…)`: bundle images
  only, PNG in 0.1, alt text required. Width and height are read from the file.
- `{% callout type="note|tip|warning" title="…" %}…{% /callout %}`
- Headings start at `##`; the page title is the `h1`. Code fences name a language the site
  highlights (`LANGUAGES` in `dialect.ts`). The site highlights them into syntax roles (Shiki's
  CSS-variables theme) that its theme colours, and heads each block with the language's name.
  Highlighting runs when a bundle is verified, from its Markdoc, so how code looks is the site's
  and never needs a republish.

Any validation error fails the publish command and the site build, with file and line.

## Interactives

An interactive is an ES module in `assets/` that exports `mount(root: HTMLElement)`. The site
loads it when its figure nears the viewport, calls `mount` with an empty element inside the
figure, and hides the fallback only after `mount` returns. Modules take colours from the site's
CSS custom properties (`--interactive-ink`, `--interactive-paper`, and the `--color-*` tokens),
so they follow light and dark mode. Gzipped size limit: 50 KB (`tools/check-perf.ts`).

**Size.** The module gets the main column's width: `root`'s inline size. It keeps its drawing and
controls within the block size the site sets as `--interactive-max-block-size` (read it with
`getComputedStyle(root)`), so a reader sees what a control does while moving it. On a short
viewport that means drawing smaller than the column allows.

## Lockfile

`deliverables.lock.json`:

```json
{
  "lockfileVersion": 1,
  "deliverables": {
    "<id>": { "version": "0.1.0", "format": "0.1", "bucket": "public",
              "key": "sha256/<hex>.tar.gz", "sha256": "<hex>", "size": 9352 }
  }
}
```

The build fetches each bundle, checks size and SHA-256, and checks that the manifest's `id`
and `version` match the entry. Republishing a pinned version with different contents is
refused: bump the version and add a revision.

## Publishing and previews

`node tools/publish-deliverable.ts <deliverable-dir>` verifies the working tree, packs it,
writes it to the bucket and pins it. With `--r2`, run through `tools/cloudflare.sh`, it
uploads to the R2 bucket with wrangler; without it, it writes to the local folder that
`DELIVERABLES_PUBLIC_URL` names as a `file:` URL. Every check that can refuse, the pin
included, runs before it uploads or writes anything. A project wraps it in its own build, run
with a checkout of this repo beside it: see
[`build.mjs`](https://github.com/cj-mills/procedural-map-generation-notes/blob/8736c5d368f915a8412199198f7b5210262358c4/build.mjs)
in `cj-mills/procedural-map-generation-notes`.

For previews, `deliverables.local.json` (not committed) maps an id to a local working tree:
`{ "<id>": "../path/to/deliverable/" }`. It is honoured only with `DELIVERABLES_ALLOW_LOCAL=1`,
which `npm run dev` sets; the tree is validated like a bundle but not hash-checked.
