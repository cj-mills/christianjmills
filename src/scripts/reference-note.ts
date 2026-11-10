// The note a reference marker opens in place (references.ts decides when). The page's references
// page is fetched once and the note is copied from it: the
// element whose id the marker's link names. The note is a popover: a click outside it or the
// Escape key closes it.
const root = document.documentElement;
const pages = new Map<string, Promise<Document>>();

function load(path: string): Promise<Document> {
  let page = pages.get(path);
  if (!page) {
    page = fetch(path).then(async (res) => {
      if (!res.ok) throw new Error(`${path}: ${res.status}`);
      return new DOMParser().parseFromString(await res.text(), 'text/html');
    });
    page.catch(() => pages.delete(path));
    pages.set(path, page);
  }
  return page;
}

/** Starts fetching the references page a marker links to, so its note is there by the click. */
export function prepare(marker: HTMLAnchorElement) {
  load(marker.pathname).catch(() => {});
}

let open: { marker: HTMLAnchorElement; note: HTMLElement } | undefined;
// A click on the open note's own marker first closes the note, as any click outside it does; the
// click then mustn't open it again.
let closed: { marker: HTMLAnchorElement; at: number } | undefined;

function place(marker: HTMLElement, note: HTMLElement) {
  const m = marker.getBoundingClientRect();
  const [width, height] = [note.offsetWidth, note.offsetHeight];
  const left = Math.max(8, Math.min(m.left, root.clientWidth - width - 8));
  // Under the marker, or over it where the window has no room below.
  const below = m.bottom + 4 + height <= innerHeight || m.top - 4 - height < 0;
  note.style.left = `${left + scrollX}px`;
  note.style.top = `${(below ? m.bottom + 4 : m.top - 4 - height) + scrollY}px`;
}

/** Opens a marker's note under it, or closes it if it is the one open. Rejects when the note
 *  can't be had, and the caller follows the link instead. */
export async function toggle(marker: HTMLAnchorElement) {
  if (open?.marker === marker) return open.note.hidePopover();
  if (closed?.marker === marker && performance.now() - closed.at < 400) return;
  const source = (await load(marker.pathname)).getElementById(decodeURIComponent(marker.hash.slice(1)));
  if (!source || !('showPopover' in HTMLElement.prototype)) throw new Error(`no note for ${marker.href}`);

  open?.note.hidePopover();
  const note = document.createElement('span');
  note.className = 'ref-note';
  note.id = 'ref-note';
  note.popover = 'auto';
  note.setAttribute('role', 'note');
  note.setAttribute('aria-label', `Reference ${marker.textContent!.replace(/[[\]]/g, '')}`);
  const text = document.createElement('span');
  text.append(...source.cloneNode(true).childNodes);
  const all = document.createElement('a');
  all.className = 'ref-note-all';
  all.setAttribute('href', marker.getAttribute('href')!);
  all.textContent = 'In the list of references';
  note.append(text, all);
  // Right after its marker, so the Tab key goes from the marker into the note.
  (marker.closest('.refs') ?? marker).after(note);
  marker.setAttribute('aria-expanded', 'true');
  marker.setAttribute('aria-details', note.id);
  note.addEventListener('beforetoggle', (event) => {
    if ((event as ToggleEvent).newState !== 'closed') return;
    closed = { marker, at: performance.now() };
    if (note.contains(document.activeElement)) marker.focus();
    marker.removeAttribute('aria-expanded');
    marker.removeAttribute('aria-details');
    if (open?.note === note) open = undefined;
    queueMicrotask(() => note.remove());
  });
  note.showPopover();
  place(marker, note);
  open = { marker, note };
}
