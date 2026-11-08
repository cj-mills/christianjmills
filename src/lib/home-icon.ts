// The home-screen icon (hub decision 2026-11-07-christianjmills-home-screen-icon): monogram E in
// the light theme's paper on a full tile of the dark theme's paper, both from the tokens. Each
// size has a whole number of pixels per lattice unit, and the mark stays inside the maskable safe
// zone (the centre circle of 80% of the tile).
import tokens from '../styles/tokens.json';
import { cells, UNITS } from './mark.ts';
import { png, rgb } from './png.ts';

/** Tile size -> pixels per lattice unit. */
export const SIZES: Record<number, number> = { 180: 14, 192: 15, 512: 40 };

export function homeIcon(size: number): Buffer {
  const unit = SIZES[size];
  if (!unit) throw new Error(`no home-screen icon at ${size} px`);
  const ink = rgb(tokens.color.paper.$value.light), tile = rgb(tokens.color.paper.$value.dark);
  const mark = cells(), o = (size - UNITS * unit) / 2;
  if (!Number.isInteger(o)) throw new Error(`home-screen icon ${size}: the mark doesn't centre on whole pixels`);
  return png(size, (x, y) => {
    const cx = Math.floor((x - o) / unit), cy = Math.floor((y - o) / unit);
    return cx >= 0 && cy >= 0 && cx < UNITS && cy < UNITS && mark[cy * UNITS + cx] ? ink : tile;
  });
}
