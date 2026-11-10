// The fixture's interactive: it stands in for a module that loads WebAssembly. It waits before it
// is ready, as that load does, and shares the wait between its mounts, as a module shares one
// WebAssembly instance. Then it shows what it was given and reports a value to its card.
let ready;
export const mounts = [];

export async function mount(root, { params, fallback, report }) {
  ready ??= new Promise((done) => setTimeout(done, 150));
  await ready;
  const steps = [...fallback.querySelectorAll('.step')];
  // A step's text is written once, in the page. Its reference marker is a `span.refs` at its end:
  // the text is read without it, and the marker is shown under the drawing, where it opens its
  // note as it does in the page.
  const written = (i) => steps[i]?.querySelector('p:not(.step-label)');
  const text = (i) => {
    const p = written(i)?.cloneNode(true);
    for (const marker of p?.querySelectorAll('.refs') ?? []) marker.remove();
    return p?.textContent ?? '';
  };

  // The drawing and the control that changes it are what must fit the screen.
  const fit = document.createElement('div');
  fit.dataset.fit = '';
  const out = document.createElement('p');
  out.className = 'status';
  const range = document.createElement('input');
  range.type = 'range';
  range.min = '0';
  range.max = String(Math.max(0, steps.length - 1));
  range.value = '0';
  range.setAttribute('aria-label', `Step, for ${params.name}`);
  // The label has the markers' class, so it goes with them when the reader hides references.
  const sources = document.createElement('p');
  sources.className = 'refs step-sources';
  const show = () => {
    out.textContent = `${params.name}: ${text(Number(range.value))}`;
    const markers = [...(written(Number(range.value))?.querySelectorAll('.refs') ?? [])].map((m) => m.cloneNode(true));
    sources.replaceChildren(...(markers.length ? ['This step’s sources:', ...markers] : []));
    report({ floor_share: params.value + Number(range.value) / 10 }, params.series);
  };
  range.addEventListener('input', show);
  fit.append(out, range);
  // The swap from the fallback to the module must not move the page under the reader, so this
  // module opens no shorter than its fallback and within a line or two of its height. The build's
  // check holds the window at several places in the figure while it mounts and reads the shift.
  root.style.minBlockSize = `${fallback.getBoundingClientRect().height}px`;
  root.append(fit, sources);
  mounts.push(params.name);
  show();
}
