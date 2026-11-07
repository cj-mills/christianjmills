// The mark: the "cjm" monogram, variant E (hub decision 2026-11-07-christianjmills-mark). One stroke
// weight on an 8 × 8 lattice: the j is the m's first stem carried down, and its dot is the start that
// needs nothing first. Every stroke, counter and gap is one lattice unit, so the mark is whole pixels
// at any multiple of 8 px. Drawn in the design lab (concepts/m2-monogram); this path is its output.

/** Lattice units per side. */
export const UNITS = 8;
/** The letters' baseline, in units from the top: the j hangs below it. */
export const BASELINE = 6;
export const PATH =
  'M3 0L4 0L4 1L3 1ZM0 2L2 2L2 3L1 3L1 5L2 5L2 6L0 6ZM3 2L8 2L8 6L7 6L7 3L6 3L6 6L5 6L5 3L4 3L4 8L1 8L1 7L3 7Z';

/** The favicon: the mark on transparent, black in the browser's light mode and white in dark. */
export const faviconSvg = () =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${UNITS} ${UNITS}" shape-rendering="crispEdges">` +
  `<style>path{fill:#000}@media (prefers-color-scheme:dark){path{fill:#fff}}</style><path d="${PATH}"/></svg>\n`;
