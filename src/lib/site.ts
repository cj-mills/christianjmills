import { getCollection, type CollectionEntry } from 'astro:content';
import categories from '../../spec/categories.json';

export type Deliverable = CollectionEntry<'deliverables'>;

export const slugOf = (d: Deliverable) => d.data.urls.canonical.replace(/^\/|\/$/g, '');
export const updatedOf = (d: Deliverable) => d.data.revisions.at(-1)!.date;

export async function deliverables(): Promise<Deliverable[]> {
  const all = await getCollection('deliverables');
  return all.sort((a, b) => b.data.published.localeCompare(a.data.published));
}

export const categoryGroups = Object.entries(categories.groups).map(([key, g]) => ({ key, ...g }));

export function formatDate(iso: string) {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
}
