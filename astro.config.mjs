import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://christianjmills.com',
  trailingSlash: 'always',
  build: { format: 'directory' },
  // Static output only for now. Paid paths would be the only routes rendered on demand (hub
  // decision 2026-11-03-site-deliverable-contract), which needs the Cloudflare adapter.
  output: 'static',
  vite: {
    // Pin the tsconfig. Left to find one itself, Rolldown's resolver walks up from any file this
    // tsconfig doesn't include (Vite's own `<root>/index.html` importer, say) to the next
    // tsconfig.json above. Inside a worktree in `.claude/worktrees/` that is the main checkout's,
    // whose `extends` fails where the main checkout has no node_modules.
    tsconfig: './tsconfig.json',
    // The site is the repo root, so other worktrees of the repo sit inside it: keep the dev server
    // from watching them.
    server: { watch: { ignored: ['**/.claude/worktrees/**'] } },
  },
});
