// The fixture's interactive that opens taller than its fallback and moves nothing, because its
// page reserves the room on the tag (spec/README.md, Interactives). The site holds the figure's
// live area to the reserve from the first paint, where scripts are running, and hands the module
// the reserve: as `reserve`, the length as the tag wrote it, and as --interactive-reserve, which
// the module uses in its own CSS. Here the drawing fills it. The build's checks have to find the
// figure the same height before and after the mount, wherever the window is and at every width
// (tools/check-perf.ts), and the room gone without scripts and when the module fails
// (tools/check-fixtures.ts).
export async function mount(root, { reserve }) {
  await new Promise((done) => setTimeout(done, 150));
  const drawing = document.createElement('div');
  drawing.dataset.fit = '';
  drawing.style.cssText = 'display: grid; place-items: center; block-size: var(--interactive-reserve); border: 1px dashed var(--color-muted); box-sizing: border-box';
  const out = document.createElement('p');
  out.className = 'status';
  out.textContent = `This module opened in the room its page reserved: ${reserve}.`;
  drawing.append(out);
  root.append(drawing);
}
