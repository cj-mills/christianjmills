// References, with scripts (hub decision 2026-11-10-christianjmills-numbered-reference-markers).
//
// The switch: the page's "Show references" boxes are one choice, kept in this browser for every
// page. The stylesheet hides the markers; the page's head applies a kept "off" before first paint.
//
// The note: a marker is a link to its note on the page's references page. Clicked, tapped or
// entered, it shows that note under itself instead (reference-note.ts), read from the references
// page, which is fetched on the reader's first move toward a marker.
import { prepare, toggle } from './reference-note.ts';

const root = document.documentElement;

// ---- The switch.
const boxes = [...document.querySelectorAll<HTMLInputElement>('.refs-switch input')];
function show(on: boolean) {
  if (on) delete root.dataset.refs;
  else root.dataset.refs = 'off';
  for (const box of boxes) box.checked = on;
  try {
    if (on) localStorage.removeItem('refs');
    else localStorage.setItem('refs', 'off');
  } catch {
    // Storage is closed to this page: the choice lasts until the page is left.
  }
}
for (const box of boxes) box.addEventListener('change', () => show(box.checked));
// A browser may bring back a box's state on reload; the kept choice is the one that counts.
for (const box of boxes) box.checked = root.dataset.refs !== 'off';

// ---- The note. Markers are found wherever they are when the reader reaches for one, so one an
// interactive's module has copied into its own root opens the same way.
const markerOf = (target: EventTarget | null) => (target instanceof Element ? target.closest<HTMLAnchorElement>('.refs a[href*="#ref-"]') : null);

for (const type of ['pointerover', 'focusin', 'touchstart'] as const) {
  document.addEventListener(
    type,
    (e) => {
      const a = markerOf(e.target);
      if (a) prepare(a);
    },
    { passive: true },
  );
}
document.addEventListener('click', (e) => {
  const a = markerOf(e.target);
  if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  // If the note can't be shown, the link is followed as it would be without scripts.
  toggle(a).catch(() => location.assign(a.href));
});
