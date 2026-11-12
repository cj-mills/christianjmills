// The machine-readable lists of the site, generated from the manifests: the RSS feed at /blog.xml
// (the old site's feed URL, kept), the sitemap, and robots.txt (hub decision
// 2026-11-03-christianjmills-launch-gate).
import { deliverables, slugOf, updatedOf, type Deliverable } from './site.ts';
import { FEED_PATH, xml } from './lists.ts';

export { ARCHIVE_SITEMAP_PATH, SITEMAP_PATH, sitemap } from './lists.ts';
export const SITE_TITLE = 'Christian Mills';
export const SITE_DESCRIPTION = 'Writing, tutorials and interactive demos.';

/** A date of the manifest's, YYYY-MM-DD, as RSS writes it: midnight UTC. */
const rfc822 = (iso: string) => new Date(`${iso}T00:00:00Z`).toUTCString().replace('GMT', '+0000');

/**
 * A feed item's guid is the deliverable's permanent id, not its URL, so a page that moves (through
 * `supersedes`) stays one item in a reader's feed. It is not a link, so isPermaLink="false".
 */
export const guidOf = (d: Deliverable) => `christianjmills.com/deliverables/${d.data.id}`;

export async function feed(site: URL): Promise<string> {
  const all = await deliverables();
  // The channel's date is the newest change of any item, so the same manifests give the same feed.
  const newest = all.map(updatedOf).sort().at(-1);
  const items = all.map((d) => {
    const link = new URL(d.data.urls.canonical, site).href;
    return [
      '    <item>',
      `      <title>${xml(d.data.title)}</title>`,
      `      <link>${xml(link)}</link>`,
      `      <guid isPermaLink="false">${xml(guidOf(d))}</guid>`,
      `      <pubDate>${rfc822(d.data.published)}</pubDate>`,
      `      <description>${xml(d.data.summary)}</description>`,
      ...d.data.categories.map((c) => `      <category>${xml(c)}</category>`),
      '    </item>',
    ].join('\n');
  });
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
    '  <channel>',
    `    <title>${xml(SITE_TITLE)}</title>`,
    `    <link>${xml(new URL('/', site).href)}</link>`,
    `    <description>${xml(SITE_DESCRIPTION)}</description>`,
    '    <language>en-us</language>',
    `    <atom:link href="${xml(new URL(FEED_PATH, site).href)}" rel="self" type="application/rss+xml" />`,
    ...(newest ? [`    <lastBuildDate>${rfc822(newest)}</lastBuildDate>`] : []),
    ...items,
    '  </channel>',
    '</rss>',
    '',
  ].join('\n');
}

/** The new site's pages a search engine may index: home, the listings, every deliverable. */
export async function sitemapEntries(): Promise<{ path: string; lastmod?: string }[]> {
  const all = await deliverables();
  const terms = [...new Set(all.flatMap((d) => d.data.categories))].sort();
  return [
    { path: '/', lastmod: all.map(updatedOf).sort().at(-1) },
    { path: '/categories/' },
    ...terms.map((t) => ({ path: `/categories/${t}/` })),
    ...all.map((d) => ({ path: `/${slugOf(d)}/`, lastmod: updatedOf(d) })),
  ];
}
