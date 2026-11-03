# Prototype deliverables

Stand-ins for content projects, kept here only while the Astro prototype is evaluated. Under the
deliverable contract each would live in its own project and publish bundles to the site; this
one belongs to the `procedural-worlds` idea in the hub.

- `cellular-automata-caves/`: cave generation by cellular automata, with an interactive, traced to
  Herbert Wolverson's 2020 talk. `node build.mjs` renders the no-JavaScript fallback from the
  interactive's own code. `sh ../../site/tools/cloudflare.sh node build.mjs --publish --r2` also
  runs the site's publish command, uploading the bundle to R2 and pinning it.
