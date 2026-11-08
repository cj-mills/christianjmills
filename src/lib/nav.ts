// The site's sections, in the order the masthead and the footer list them. Only pages that exist go
// here; a section is added by adding its line.
export type Section = 'writing' | 'categories' | 'search';
export const NAV: { id: Section; label: string; href: string }[] = [
  { id: 'writing', label: 'Writing', href: '/' },
  { id: 'categories', label: 'Categories', href: '/categories/' },
  { id: 'search', label: 'Search', href: '/search/' },
];

/** aria-current for a nav link: "page" on the section's own page, "true" anywhere inside it. */
export function current(item: (typeof NAV)[number], path: string, section?: Section) {
  if (path === item.href) return 'page';
  if (section === item.id) return 'true';
  return undefined;
}
