// Mounts interactives declared with {% interactive %}. Each figure carries its module URL; the module
// exports mount(root). The fallback stays in place until the module mounts, and stays if it fails,
// so the page works without JavaScript and when a module breaks.
interface InteractiveModule {
  mount(root: HTMLElement): void;
}

async function activate(figure: HTMLElement) {
  const url = figure.dataset.module;
  if (!url) return;
  try {
    const mod = (await import(/* @vite-ignore */ url)) as InteractiveModule;
    const root = document.createElement('div');
    root.className = 'interactive-root';
    figure.prepend(root);
    mod.mount(root);
    figure.querySelector<HTMLElement>('.interactive-fallback')?.setAttribute('hidden', '');
  } catch (err) {
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
  { rootMargin: '200px' },
);
figures.forEach((f) => observer.observe(f));
