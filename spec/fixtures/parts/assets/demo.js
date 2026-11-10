// The fixture's interactive: it stands in for a module that loads WebAssembly. It waits before it
// is ready, as that load does, and shares the wait between its mounts, as a module shares one
// WebAssembly instance. Then it shows what it was given and reports a value to its card.
let ready;
export const mounts = [];

export async function mount(root, { params, fallback, report }) {
  ready ??= new Promise((done) => setTimeout(done, 150));
  await ready;
  const steps = [...fallback.querySelectorAll('.step')];
  const text = (i) => steps[i]?.querySelector('p:not(.step-label)')?.textContent ?? '';

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
  const show = () => {
    out.textContent = `${params.name}: ${text(Number(range.value))}`;
    report({ floor_share: params.value + Number(range.value) / 10 }, params.series);
  };
  range.addEventListener('input', show);
  fit.append(out, range);
  // The swap from the fallback to the module must not move the page under the reader (the build's
  // layout-shift check covers it), so this module starts at the size its fallback had.
  root.style.minBlockSize = `${fallback.getBoundingClientRect().height}px`;
  root.append(fit);
  mounts.push(params.name);
  show();
}
