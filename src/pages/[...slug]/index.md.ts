import type { APIRoute } from 'astro';
import { deliverables, slugOf, type Deliverable } from '../../lib/site.ts';

// The Markdown copy of each page, for agents. Rendered by the loader from the validated content.
export async function getStaticPaths() {
  return (await deliverables()).map((entry) => ({ params: { slug: slugOf(entry) }, props: { entry } }));
}

export const GET: APIRoute = ({ props }) =>
  new Response((props as { entry: Deliverable }).entry.data.markdown, {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
