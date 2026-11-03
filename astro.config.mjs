import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://christianjmills.com',
  trailingSlash: 'always',
  build: { format: 'directory' },
  // Static output only for now. Paid paths would be the only routes rendered on demand (hub
  // decision 2026-11-03-site-deliverable-contract), which needs the Cloudflare adapter.
  output: 'static',
  // The site is the repo root, so other worktrees of the repo sit inside it: keep the dev server
  // from watching them.
  vite: { server: { watch: { ignored: ['**/.claude/worktrees/**'] } } },
});
