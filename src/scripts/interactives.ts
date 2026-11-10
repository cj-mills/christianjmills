// Mounts interactives declared with {% interactive %} (spec/README.md, Interactives). Each figure
// carries its module's URL and its own parameters; the module exports mount(root, context). One
// module may be mounted in many figures on a page: it is imported once and mount is called once for
// each. The fallback stays in place until mount has finished, which a module that loads
// WebAssembly does later, and stays if the module fails, so the page works without JavaScript and
// when a module breaks.
interface MountContext {
  /** The tag's `params`, parsed: whatever this mount is given. */
  params: Record<string, unknown>;
  /** The fallback, still in the page. A module may read it (a step list holds each step's text); it doesn't change it. */
  fallback: HTMLElement;
  /** Shows the module's own values beside the measured ones in the card the figure is in: each
   *  {% now %} of a trait gets the number, and each {% mark %} of it a marker. With `series`, only
   *  the marks of that series. A value of null clears it. Outside a card it does nothing. */
  report(values: Record<string, number | null>, series?: string): void;
}
interface InteractiveModule {
  mount(root: HTMLElement, context: MountContext): void | Promise<void>;
}

function reporter(figure: HTMLElement): MountContext['report'] {
  const card = figure.closest<HTMLElement>('.card');
  return (values, series) => {
    if (!card) return;
    for (const [trait, value] of Object.entries(values)) {
      const named = `[data-trait="${CSS.escape(trait)}"]`;
      for (const now of card.querySelectorAll<HTMLElement>(`.trait-now${named}`))
        now.textContent = value === null ? '' : value.toFixed(Number(now.dataset.digits ?? 0));
      for (const mark of card.querySelectorAll<HTMLElement>(`.trait-mark${named}`)) {
        const mine = series === undefined || mark.dataset.series === undefined || mark.dataset.series === series;
        const [from, to] = [Number(mark.dataset.from), Number(mark.dataset.to)];
        mark.classList.toggle('has-now', mine && value !== null);
        if (mine && value !== null) mark.style.setProperty('--now', String(Math.min(1, Math.max(0, (value - from) / (to - from)))));
      }
    }
  };
}

async function activate(figure: HTMLElement) {
  const url = figure.dataset.module;
  const fallback = figure.querySelector<HTMLElement>('.interactive-fallback');
  if (!url || !fallback) return;
  const name = `interactive:${figure.dataset.interactive}`;
  const root = document.createElement('div');
  root.className = 'interactive-root';
  // The build's performance check reads how long each mount took, from here to mount's end.
  const started = performance.now();
  try {
    const mod = (await import(/* @vite-ignore */ url)) as InteractiveModule;
    figure.dataset.mounting = '';
    figure.prepend(root);
    await mod.mount(root, {
      params: JSON.parse(figure.dataset.params ?? '{}'),
      fallback,
      report: reporter(figure),
    });
    fallback.hidden = true;
    delete figure.dataset.mounting;
    figure.dataset.mounted = '';
    performance.measure(name, { start: started, end: performance.now() });
  } catch (err) {
    root.remove();
    delete figure.dataset.mounting;
    figure.dataset.mountFailed = '';
    console.error(`interactive ${figure.dataset.interactive} failed to mount; keeping its fallback`, err);
  }
}

const figures = document.querySelectorAll<HTMLElement>('figure[data-interactive]');
const observer = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      observer.unobserve(e.target);
      activate(e.target as HTMLElement);
    }
  },
  { rootMargin: '100% 0px' },
);
figures.forEach((f) => observer.observe(f));
