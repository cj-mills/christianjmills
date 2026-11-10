// A post's scripts, as the one file every post links from its head (hub decision
// 2026-11-10-christianjmills-page-scripts-in-one-file). The browser fetches it while it reads the
// page, runs it once the page is parsed, and keeps it: it is the same for every post, so it is not
// sent again with each page or counted in a page's HTML. Each part does nothing on a page without
// its elements, and none of them moves the page when it runs (tools/check-fixtures.ts), so a file
// that arrives late changes only when the buttons, the notes and the interactives start to work.
import './interactives.ts';
import './references.ts';
import './code.ts';
import './tables.ts';
import './contents.ts';
