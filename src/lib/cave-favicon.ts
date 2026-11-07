// The fallback favicon, for browsers without SVG favicons: the cellular-automata cave
// christianjmills.com#107, "the pillar" (hub decision 2026-11-07-christianjmills-mark), as an opaque
// tile of black rock on a white room. A 16 × 16 cave from the design lab's mark rounds
// (concepts/m1-procedural, m2-monogram/fallback.mjs), ported unchanged so the same seed gives the
// same cave. Deterministic: string seeds through FNV-1a and mulberry32, no clock, no Math.random.
import { deflateSync } from 'node:zlib';

const N = 16;
const SEED = 107;
const ROCK = [0, 0, 0];
const ROOM = [255, 255, 255];

const fnv = (s: string) => {
  let h = 0x811c9dc5;
  for (const c of s) {
    h ^= c.codePointAt(0)!;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
};
const mulberry32 = (a: number) => () => {
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
type Grid = Uint8Array; // 1 = rock
const gridOf = (f: (x: number, y: number) => boolean): Grid => {
  const g = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) g[y * N + x] = f(x, y) ? 1 : 0;
  return g;
};
const rock = (g: Grid, x: number, y: number) => (x < 0 || y < 0 || x >= N || y >= N ? 1 : g[y * N + x]);
// Whether (x, y) lies in a 2 × 2 block of its own kind (no feature thinner than two cells).
const in2 = (g: Grid, x: number, y: number, v: number) =>
  [[0, 0], [-1, 0], [0, -1], [-1, -1]].some(([ox, oy]) => [[0, 0], [1, 0], [0, 1], [1, 1]].every(([a, b]) => rock(g, x + ox + a, y + oy + b) === v));

/** A cave from a seed, or null if it fails the round's test (room share, no thin features). */
function caveTry(seed: number, fill = 0.5): Grid | null {
  const rnd = mulberry32(seed);
  let g = gridOf((x, y) => x === 0 || y === 0 || x === N - 1 || y === N - 1 || rnd() < fill);
  for (let it = 0; it < 4; it++) {
    const out = g.slice();
    for (let y = 1; y < N - 1; y++)
      for (let x = 1; x < N - 1; x++) {
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dx || dy) n += rock(g, x + dx, y + dy);
        out[y * N + x] = n > 4 || n === 0 ? 1 : 0;
      }
    g = out;
  }
  for (let pass = 0, changed = true; changed && pass < 20; pass++) {
    changed = false;
    for (const v of [0, 1]) {
      const out = g.slice();
      for (let y = 1; y < N - 1; y++)
        for (let x = 1; x < N - 1; x++)
          if (g[y * N + x] === v && !in2(g, x, y, v)) {
            out[y * N + x] = 1 - v;
            changed = true;
          }
      g = out;
    }
  }
  // Keep only the largest room.
  const label = new Int32Array(N * N).fill(-1);
  let best = -1, bestN = 0, id = 0;
  for (let i = 0; i < N * N; i++) {
    if (g[i] || label[i] >= 0) continue;
    const stack = [i];
    label[i] = id;
    let count = 0;
    while (stack.length) {
      const j = stack.pop()!;
      count++;
      const x = j % N, y = (j / N) | 0;
      for (const [ax, ay] of [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]) {
        const k = ay * N + ax;
        if (ax >= 0 && ay >= 0 && ax < N && ay < N && !g[k] && label[k] < 0) {
          label[k] = id;
          stack.push(k);
        }
      }
    }
    if (count > bestN) [bestN, best] = [count, id];
    id++;
  }
  g = gridOf((x, y) => label[y * N + x] !== best);
  const interior = (N - 2) * (N - 2);
  if (bestN < 0.35 * interior || bestN > 0.6 * interior) return null;
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) if (!in2(g, x, y, g[y * N + x])) return null;
  return g;
}

/** The cave the user picked: loop index SEED of christianjmills.com#i, which must pass the test. */
export function pickedCave(): Grid {
  const g = caveTry(fnv(`christianjmills.com#${SEED}`));
  if (!g) throw new Error(`christianjmills.com#${SEED} no longer passes the cave test`);
  return g;
}

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf: Uint8Array) => {
  let c = 0xffffffff;
  for (const b of buf) c = CRC[(c ^ b) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type: string, data: Buffer) => {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
};
/** An RGB PNG of the cave at S × S pixels (S a multiple of 16: every cell whole pixels). */
export function cavePng(S: number, g: Grid = pickedCave()): Buffer {
  const raw = Buffer.alloc((S * 3 + 1) * S);
  for (let y = 0; y < S; y++) {
    raw[y * (S * 3 + 1)] = 0;
    for (let x = 0; x < S; x++) {
      const on = g[Math.floor((y * N) / S) * N + Math.floor((x * N) / S)];
      (on ? ROCK : ROOM).forEach((v, i) => (raw[y * (S * 3 + 1) + 1 + x * 3 + i] = v));
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(S, 0);
  ihdr.writeUInt32BE(S, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolour
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}
/** favicon.ico: 16, 32 and 48 px PNGs inside one ICO. */
export function caveIco(): Buffer {
  const sizes = [16, 32, 48], g = pickedCave(), pngs = sizes.map((S) => cavePng(S, g));
  const head = Buffer.alloc(6);
  head.writeUInt16LE(0, 0);
  head.writeUInt16LE(1, 2);
  head.writeUInt16LE(sizes.length, 4);
  let off = 6 + 16 * sizes.length;
  const dir = sizes.map((S, k) => {
    const e = Buffer.alloc(16);
    e[0] = S;
    e[1] = S;
    e.writeUInt16LE(1, 4);
    e.writeUInt16LE(24, 6);
    e.writeUInt32LE(pngs[k].length, 8);
    e.writeUInt32LE(off, 12);
    off += pngs[k].length;
    return e;
  });
  return Buffer.concat([head, ...dir, ...pngs]);
}
