import type { APIRoute } from 'astro';
import { ARCHIVE_SITEMAP_PATH, SITEMAP_PATH } from '../lib/feeds.ts';

// Crawlers may read everything; the preview's hosts are kept out of search by a header instead
// (public/_headers), since the same files are served on workers.dev and on the live domain.
export const GET: APIRoute = ({ site }) =>
  new Response(
    ['User-agent: *', 'Allow: /', '', `Sitemap: ${new URL(SITEMAP_PATH, site)}`, `Sitemap: ${new URL(ARCHIVE_SITEMAP_PATH, site)}`, ''].join('\n'),
    { headers: { 'Content-Type': 'text/plain; charset=utf-8' } },
  );
