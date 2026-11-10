// Lets the reader order a table's rows by a column of numbers, where the table asks for it
// ({% table sort=true %}). A sorted table drops its row groups; choosing the sorted column again
// puts the rows back as written. Without the script the table is as written.
for (const table of document.querySelectorAll<HTMLTableElement>('table[data-sort]')) {
  const head = table.tHead?.rows[0];
  if (!head) continue;
  const written = [...table.tBodies].map((body) => [...body.rows]);
  const rows = written.flat().filter((r) => !r.classList.contains('group'));
  // A cell's number is the first thing in it: "0.637", or "–" where there is none.
  const number = (row: HTMLTableRowElement, col: number) => Number.parseFloat(row.cells[col]?.textContent ?? '');
  let sorted = -1;

  [...head.cells].forEach((th, col) => {
    if (!rows.some((r) => !Number.isNaN(number(r, col)))) return;
    const label = (th.firstChild?.textContent ?? th.textContent ?? '').trim();
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'sort';
    button.textContent = '↕';
    button.setAttribute('aria-label', `Sort by ${label}`);
    button.addEventListener('click', () => {
      for (const h of head.cells) h.removeAttribute('aria-sort');
      if (sorted === col) {
        sorted = -1;
        delete table.dataset.sorted;
        written.forEach((group, i) => table.tBodies[i].append(...group));
        return;
      }
      sorted = col;
      table.dataset.sorted = '';
      th.setAttribute('aria-sort', 'descending');
      const value = (r: HTMLTableRowElement) => (Number.isNaN(number(r, col)) ? -Infinity : number(r, col));
      table.tBodies[0].append(...[...rows].sort((a, b) => value(b) - value(a)));
    });
    th.append(' ', button);
  });
}
