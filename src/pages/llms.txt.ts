import type { APIRoute } from 'astro';
import { deliverables } from '../lib/site.ts';

export const GET: APIRoute = async ({ site }) => {
  const all = (await deliverables()).filter((d) => d.data.tier.agents === 'allowed');
  const lines = [
    '# Christian Mills',
    '',
    '> Writing, tutorials and interactive demos. Each page below has a Markdown copy.',
    '',
    '## Pages',
    '',
    ...all.map((d) => `- [${d.data.title}](${new URL(`${d.data.urls.canonical}index.md`, site)}): ${d.data.summary}`),
    '',
  ];
  return new Response(lines.join('\n'), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
