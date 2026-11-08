// iOS and iPadOS ask for /apple-touch-icon.png when a page is added to the home screen.
import type { APIRoute } from 'astro';
import { homeIcon } from '../lib/home-icon.ts';

export const GET: APIRoute = () => new Response(homeIcon(180), { headers: { 'Content-Type': 'image/png' } });
