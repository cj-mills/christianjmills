import type { APIRoute } from 'astro';
import { feed } from '../lib/feeds.ts';

// The RSS feed, at the old site's feed URL so its subscribers keep it (src/lib/feeds.ts).
export const GET: APIRoute = async ({ site }) =>
  new Response(await feed(site!), { headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' } });
