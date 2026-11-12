// The site's machine-readable lists, the parts that need no content collection: their addresses,
// XML escaping and the sitemap's form. Build tools that run outside Astro use them too
// (tools/archive-merge.ts).
export const FEED_PATH = '/blog.xml';
/** The new site's pages. The frozen copy of the old site lists its own (tools/archive-merge.ts). */
export const SITEMAP_PATH = '/sitemap.xml';
export const ARCHIVE_SITEMAP_PATH = '/sitemap-archive.xml';

const ENTITIES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' };
export const xml = (s: string) => s.replace(/[&<>"']/g, (c) => ENTITIES[c]);

export function sitemap(site: URL, entries: { path: string; lastmod?: string }[]): string {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...entries.map(
      (e) => `  <url><loc>${xml(new URL(e.path, site).href)}</loc>${e.lastmod ? `<lastmod>${e.lastmod}</lastmod>` : ''}</url>`,
    ),
    '</urlset>',
    '',
  ].join('\n');
}
