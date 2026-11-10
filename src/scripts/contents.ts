// Marks the section being read in the contents rail: the last section whose heading has passed a
// line a third of the way down the window, or the last section once the page is scrolled to its end.
// Where a section lists its cards, the card being read is the one marked. A rail longer than the
// window keeps the marked entry in view. Without the script the rail is a plain list of links.
const rail = document.querySelector<HTMLElement>('.contents-rail');
const links = [...document.querySelectorAll<HTMLAnchorElement>('.contents-rail a[href^="#"]')];
const heads = links.map((a) => document.getElementById(decodeURIComponent(a.hash.slice(1))));
let queued = false;

function mark() {
  queued = false;
  const line = window.innerHeight / 3;
  let at = -1;
  heads.forEach((h, i) => {
    if (h && h.getBoundingClientRect().top <= line) at = i;
  });
  const end = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
  if (end && at >= 0) at = links.length - 1;
  links.forEach((a, i) => (i === at ? a.setAttribute('aria-current', 'location') : a.removeAttribute('aria-current')));
  const now = links[at];
  if (!now || !rail || rail.scrollHeight <= rail.clientHeight) return;
  // Scroll the rail itself, never the page, and only as far as brings the entry in.
  const [box, entry] = [rail.getBoundingClientRect(), now.getBoundingClientRect()];
  if (entry.top < box.top) rail.scrollTop -= box.top - entry.top;
  else if (entry.bottom > box.bottom) rail.scrollTop += entry.bottom - box.bottom;
}

if (links.length) {
  const queue = () => {
    if (!queued) requestAnimationFrame(mark);
    queued = true;
  };
  addEventListener('scroll', queue, { passive: true });
  addEventListener('resize', queue);
  mark();
}
