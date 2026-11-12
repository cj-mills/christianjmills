import type { APIRoute } from 'astro';
import { caveIco } from '../lib/cave-favicon.ts';

export const GET: APIRoute = () => new Response(caveIco() as Uint8Array<ArrayBuffer>, { headers: { 'Content-Type': 'image/x-icon' } });
