// Adds a copy button to each code block's head (hub decision 2026-11-08-christianjmills-code-blocks).
// The button needs JavaScript, so it is made here: without JavaScript the block has no dead control.
// The head already has the button's height (global.css, Code), so adding it moves nothing.
for (const block of document.querySelectorAll<HTMLElement>('.code-block')) {
  const head = block.querySelector<HTMLElement>('.code-head');
  const code = block.querySelector<HTMLElement>('pre code') ?? block.querySelector<HTMLElement>('pre');
  if (!head || !code) continue;
  if (!navigator.clipboard) {
    // No way to copy here: a head that was only waiting for the button goes.
    head.hidden = head.childElementCount === 0;
    continue;
  }

  const language = block.querySelector('.code-language')?.textContent;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'copy';
  button.textContent = 'Copy';
  // The accessible name starts with the visible label, and says which block.
  button.setAttribute('aria-label', language ? `Copy the ${language} code` : 'Copy the code');

  const status = document.createElement('span');
  status.className = 'copy-status';
  status.setAttribute('role', 'status');

  let timer: ReturnType<typeof setTimeout> | undefined;
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code.textContent ?? '');
      status.textContent = 'Copied';
    } catch {
      status.textContent = 'Couldn’t copy';
    }
    clearTimeout(timer);
    timer = setTimeout(() => (status.textContent = ''), 2000);
  });

  head.append(status, button);
}
