import type { APIRoute } from 'astro';
import { sitemap, sitemapEntries } from '../lib/feeds.ts';

// The new site's pages, from the manifests. The frozen copy's pages are in /sitemap-archive.xml.
export const GET: APIRoute = async ({ site }) =>
  new Response(sitemap(site!, await sitemapEntries()), { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
