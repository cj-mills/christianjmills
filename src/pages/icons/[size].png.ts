// The web app manifest's home-screen icons: /icons/192.png and /icons/512.png.
import type { APIRoute, GetStaticPaths } from 'astro';
import { homeIcon } from '../../lib/home-icon.ts';

export const getStaticPaths = (() => [{ params: { size: '192' } }, { params: { size: '512' } }]) satisfies GetStaticPaths;
export const GET: APIRoute = ({ params }) => new Response(homeIcon(Number(params.size)) as Uint8Array<ArrayBuffer>, { headers: { 'Content-Type': 'image/png' } });
