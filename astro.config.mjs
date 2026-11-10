import { defineConfig } from 'astro/config';
import { FONTS, SUBSETS, fontsourcePackages } from './tools/design/fonts.ts';

export default defineConfig({
  site: 'https://christianjmills.com',
  trailingSlash: 'always',
  build: { format: 'directory' },
  // Static output only for now. Paid paths would be the only routes rendered on demand (hub
  // decision 2026-11-03-site-deliverable-contract), which needs the Cloudflare adapter.
  output: 'static',
  // Self-hosted fonts from pinned packages (tools/design/fonts.ts): Astro writes the @font-face
  // rules, metric-matched fallbacks for each generic family, and the preload links.
  fonts: FONTS.map((f) => ({
    provider: fontsourcePackages(),
    name: f.family,
    cssVariable: `--font-${f.role}`,
    weights: f.weights.map((w) => w.weight),
    styles: f.styles,
    subsets: [...SUBSETS],
    fallbacks: f.fallbacks,
  })),
  vite: {
    // Pin the tsconfig. Left to find one itself, Rolldown's resolver walks up from any file this
    // tsconfig doesn't include (Vite's own `<root>/index.html` importer, say) to the next
    // tsconfig.json above. Inside a worktree in `.claude/worktrees/` that is the main checkout's,
    // whose `extends` fails where the main checkout has no node_modules.
    tsconfig: './tsconfig.json',
    // A page's scripts are files, never written into its HTML, however small they get: the same
    // file serves every post and is kept by the browser (hub decision
    // 2026-11-10-christianjmills-page-scripts-in-one-file). Everything else keeps Vite's rule.
    build: { assetsInlineLimit: (file) => (file.endsWith('.js') ? false : undefined) },
    // The site is the repo root, so other worktrees of the repo sit inside it: keep the dev server
    // from watching them.
    server: { watch: { ignored: ['**/.claude/worktrees/**'] } },
  },
});
