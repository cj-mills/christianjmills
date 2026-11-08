// The web app manifest, for Android's "Add to Home screen": the site's name and its home-screen
// icons. The site stays a website (display "browser"); the manifest exists for the icon.
import type { APIRoute } from 'astro';
import tokens from '../styles/tokens.json';

export const GET: APIRoute = () => {
  const icons = [192, 512].flatMap((s) =>
    (['any', 'maskable'] as const).map((purpose) => ({ src: `/icons/${s}.png`, sizes: `${s}x${s}`, type: 'image/png', purpose })),
  );
  const manifest = {
    name: 'Christian Mills',
    short_name: 'Christian Mills',
    start_url: '/',
    display: 'browser',
    background_color: tokens.color.paper.$value.dark,
    icons,
  };
  return new Response(JSON.stringify(manifest, null, 2) + '\n', { headers: { 'Content-Type': 'application/manifest+json' } });
};
