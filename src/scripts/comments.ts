// Loads a post's comments from giscus.app when the reader presses "Show comments", and not before
// (src/components/Comments.astro): no third-party script runs unasked.
import { GISCUS, giscusAttributes, type Thread } from '../lib/comments.ts';

for (const section of document.querySelectorAll<HTMLElement>('.comments[data-term]')) {
  const button = section.querySelector<HTMLButtonElement>('.comments-load');
  if (!button) continue;
  button.addEventListener(
    'click',
    () => {
      const host = document.createElement('div');
      host.className = 'giscus';
      // Focus stays with the comments once the button is gone.
      host.tabIndex = -1;
      const script = document.createElement('script');
      script.src = `${GISCUS.origin}/client.js`;
      script.async = true;
      script.crossOrigin = 'anonymous';
      const thread = { mapping: section.dataset.mapping as Thread['mapping'], term: section.dataset.term! };
      for (const [name, value] of Object.entries(giscusAttributes(thread))) script.setAttribute(name, value);
      button.closest('p')!.replaceWith(host);
      section.append(script);
      host.focus();
    },
    { once: true },
  );
}
