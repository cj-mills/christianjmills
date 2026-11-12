// Comments are GitHub Discussions in this repo's Comments category, shown through giscus, which
// loads only when the reader asks (hub decision 2026-11-03-christianjmills-launch-gate). A thread
// belongs to the deliverable's permanent id, not its URL, so a page that moves keeps its comments;
// a manifest may name an older discussion to continue instead (`comments.discussion`).
//
// The ids are GitHub's for cj-mills/christianjmills and its Comments category (observed
// 2026-11-12: the GraphQL API's repository.id and discussionCategories).
export const GISCUS = {
  repo: 'cj-mills/christianjmills',
  repoId: 'MDEwOlJlcG9zaXRvcnkyOTM2MTIwMjI=',
  category: 'Comments',
  categoryId: 'DIC_kwDOEYAp9s4DGyfc',
  origin: 'https://giscus.app',
} as const;

export interface Thread {
  /** giscus's mapping: a discussion titled with the deliverable's id, or one by its number. */
  mapping: 'specific' | 'number';
  term: string;
  /** Where the thread is without scripts: the discussion, or the category's search for the id. */
  href: string;
}

export function threadOf(id: string, discussion?: number): Thread {
  const base = `https://github.com/${GISCUS.repo}/discussions`;
  if (discussion) return { mapping: 'number', term: String(discussion), href: `${base}/${discussion}` };
  return { mapping: 'specific', term: id, href: `${base}/categories/comments?discussions_q=${encodeURIComponent(`in:title ${id}`)}` };
}

/** The attributes giscus's client reads from its script element. */
export function giscusAttributes(thread: Pick<Thread, 'mapping' | 'term'>): Record<string, string> {
  return {
    'data-repo': GISCUS.repo,
    'data-repo-id': GISCUS.repoId,
    'data-category': GISCUS.category,
    'data-category-id': GISCUS.categoryId,
    'data-mapping': thread.mapping,
    'data-term': thread.term,
    // A discussion is matched by a hash of the whole term, so one id never picks up another's.
    'data-strict': '1',
    'data-reactions-enabled': '1',
    'data-emit-metadata': '0',
    'data-input-position': 'top',
    // Follows the reader's light or dark setting, as the site does. Provisional: a theme drawn
    // from the site's tokens is the design system's to make.
    'data-theme': 'preferred_color_scheme',
    'data-lang': 'en',
  };
}
