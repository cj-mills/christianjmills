// The fixture's interactive that breaks the rule for a mount: it opens much taller than its
// fallback, so the swap moves whatever follows the figure. The build's check has to find that
// wherever the window is while the module mounts (tools/check-perf.ts, EXPECT_MOUNT_SHIFTS), and
// the site has to hold the figure's end for a reader whose window starts inside the figure
// (tools/check-fixtures.ts).
export async function mount(root, { fallback }) {
  await new Promise((done) => setTimeout(done, 150));
  root.style.minBlockSize = `calc(${fallback.getBoundingClientRect().height}px + 30rem)`;
  const out = document.createElement('p');
  out.className = 'status';
  out.textContent = 'This module opened taller than its fallback.';
  root.append(out);
}
