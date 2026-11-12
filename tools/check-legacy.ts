// The frozen copy's routing, tried on a made-up old site before every build (tools/lib/legacy-plan.ts):
// the cases the real copy has (alias pages, retired posts, an old page a deliverable takes over) and
// those it will have (a superseded page, a chain of redirects), and each mistake it must refuse.
import { planMerge } from './lib/legacy-plan.ts';

const failures: string[] = [];
const expect = (ok: boolean, what: string) => {
  if (!ok) failures.push(what);
};
const enc = (s: string) => new TextEncoder().encode(s);
const page = (title: string) => enc(`<!doctype html><html><body><main class="content">${title}</main></body></html>`);
const alias = (to: string) =>
  enc(`<html xmlns="http://www.w3.org/1999/xhtml">\n<head>\n  <title>Redirect</title>\n  <script type="text/javascript">\n    var redirects = {"":"${to}"};\n    var hash = 1;\n  </script>\n</head>\n</html>\n`);
const oldSitemap = (paths: string[]) =>
  enc(`<?xml version="1.0"?><urlset>${paths.map((p) => `<url><loc>https://christianjmills.com${p}</loc><lastmod>2025-10-15T00:00:00Z</lastmod></url>`).join('')}</urlset>`);

const archive = () =>
  new Map<string, Uint8Array>([
    ['index.html', page('old home')],
    ['404.html', page('old 404')],
    ['about.html', page('about')],
    ['blog.xml', enc('<rss/>')],
    ['sitemap.xml', oldSitemap(['/about.html', '/posts/old/index.html', '/posts/kept/index.html', '/posts/taken/index.html', '/posts/retired/index.html'])],
    ['robots.txt', enc('')],
    ['CNAME', enc('christianjmills.com')],
    ['posts/old/index.html', page('old')],
    ['posts/old/images/a.png', enc('png')],
    ['posts/kept/index.html', page('kept')],
    ['posts/taken/index.html', page('taken')],
    ['posts/retired/index.html', page('retired')],
    ['posts/retired/video/demo.mp4', enc('mp4')],
    ['services.html', alias('about.html')],
    ['series/a.html', alias('../posts/old/index.html')],
    ['Notes on Old/index.html', alias('../posts/kept/index.html')],
  ]);
const built = () => new Set(['index.html', '404.html', 'blog.xml', 'sitemap.xml', 'robots.txt', 'posts/new/index.html', 'posts/taken/index.html']);
const manifests = [
  { id: 'new', urls: { canonical: '/posts/new/', supersedes: ['/posts/old/'] } },
  { id: 'taken', urls: { canonical: '/posts/taken/', supersedes: [] } },
];
const retired = { '/posts/retired/': 'https://github.com/cj-mills/retired-posts/blob/main/posts/retired/README.md' };
const input = () => ({ archive: archive(), builtOn: '2026-09-07', retired, manifests: [...manifests], built: built() });

const plan = planMerge(input());
const to = (from: string) => plan.redirects.get(from)?.to;
expect(['/posts/old/', '/posts/old/index.html', '/posts/old'].every((a) => to(a) === '/posts/new/'), 'a superseded page should send each of its addresses to the new page');
expect(plan.leaving.get('posts/old/index.html') === 'superseded by new' && !plan.writes.has('posts/old/index.html'), 'a superseded page should leave the copy');
expect(plan.writes.has('posts/old/images/a.png'), "a superseded page's other files stay, at their addresses");
expect(to('/series/a.html') === '/posts/new/', 'an alias to a superseded page should go to the new page in one hop');
expect(to('/services.html') === '/about.html' && to('/services') === '/about.html', 'an alias page should be a 301 to its target, by name and without .html');
expect(to('/Notes on Old/') === '/posts/kept/index.html' && plan.rules.includes('/Notes%20on%20Old/ /posts/kept/index.html 301'), 'an alias in a folder with spaces should be written encoded');
expect(['/posts/retired/', '/posts/retired/index.html', '/posts/retired'].every((a) => to(a) === retired['/posts/retired/']), 'a retired post should go to its new home');
expect(plan.leaving.get('posts/retired/index.html') === 'retired' && plan.writes.has('posts/retired/video/demo.mp4'), "a retired post's page should leave the copy, and its files stay");
expect(plan.expectations.some((e) => e.path === '/posts/retired/video/demo.mp4' && e.expect === 200), 'the URL check should ask for every file the copy still serves');
expect(!plan.writes.has('posts/taken/index.html') && plan.leaving.get('posts/taken/index.html') === 'taken over by taken', 'an old page a deliverable takes over should leave the copy');
expect(!plan.writes.has('CNAME') && !plan.writes.has('index.html') && !plan.writes.has('robots.txt'), "GitHub Pages' files and the replaced root files should not be written");
const kept = String(plan.writes.get('posts/kept/index.html'));
expect(kept.startsWith('<!doctype html><html><body><main class="content"><div role="note" data-archived') && kept.includes('archived as it was on September 7, 2026') && kept.endsWith('kept</main></body></html>'), 'an old page should get the banner at the start of its <main>, and nothing else');
expect(to('/posts/kept') === '/posts/kept/' && to('/about') === '/about.html', 'a folder without its slash, and an old page without .html, should be sent to the address that serves it');
expect(plan.rules.trimEnd().endsWith('/ /index.html 200\n/*/ /:splat/index.html 200'), 'the rewrites that serve a folder should come last');
expect(plan.archiveSitemap.includes('/posts/kept/index.html') && plan.archiveSitemap.includes('/about.html') && !plan.archiveSitemap.includes('/posts/old/') && !plan.archiveSitemap.includes('/posts/taken/'), 'the archive sitemap should list only the old pages the copy still serves');
const ex = new Map(plan.expectations.map((e) => [e.path, e]));
expect(ex.get('/posts/old/index.html')?.expect === '/posts/new/' && ex.get('/posts/kept/index.html')?.expect === 200 && ex.get('/posts/taken/index.html')?.why === 'old sitemap, now a page of the new site', 'the URL check should ask each old address for what it must answer');

const refuses = (what: string, needle: string, change: (i: ReturnType<typeof input>) => void) => {
  const i = input();
  change(i);
  try {
    planMerge(i);
    failures.push(`${what} should have been refused`);
  } catch (err) {
    expect(String(err).includes(needle), `${what} was refused for another reason: ${String(err).slice(0, 200)}`);
  }
};
refuses('a file the new site and the copy both have', 'the frozen copy also has', (i) => i.built.add('posts/kept/index.html'));
refuses('a root file the new site should replace and does not', 'should serve /blog.xml', (i) => i.built.delete('blog.xml'));
refuses('a redirect to a page the site does not serve', "doesn't serve", (i) => i.manifests.push({ id: 'gone', urls: { canonical: '/posts/gone/', supersedes: ['/posts/kept/'] } }));
refuses('a loop of alias pages', 'redirect loop', (i) => {
  i.archive.set('x.html', alias('y.html'));
  i.archive.set('y.html', alias('x.html'));
});
refuses('one address sent to two places', 'would redirect both', (i) => i.manifests.push({ id: 'other', urls: { canonical: '/posts/taken/', supersedes: ['/posts/old/'] } }));
refuses('an old page with nowhere for the banner', 'no <main>', (i) => i.archive.set('posts/kept/index.html', enc('<html><body>no main</body></html>')));
refuses('a retired post the copy has no page for', 'no page for', (i) => i.archive.delete('posts/retired/index.html'));
refuses('a second merge into the same dist/', 'already has a _redirects', (i) => i.built.add('_redirects'));

if (failures.length) {
  console.error(`frozen copy check failed (${failures.length}):\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('frozen copy check passed: aliases, retired posts, a superseded page, a chain of redirects and a takeover route as they should, and each mistake is refused');
