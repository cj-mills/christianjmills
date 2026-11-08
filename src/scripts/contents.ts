// Marks the section being read in the contents rail: the last section whose heading has passed a
// line a third of the way down the window, or the last section once the page is scrolled to its end.
// Without the script the rail is a plain list of links.
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
