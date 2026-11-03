// The deliverable's own build: render the no-JavaScript fallback from the same code the
// interactive runs, then hand the bundle to the site's publish command.
//
//   node build.mjs                     render assets only
//   node build.mjs --publish <args>    render, then run the site's publish command with <args>
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { crc32, deflateSync } from 'node:zlib';
import { generate, openShare } from './assets/caves.js';

const here = dirname(fileURLToPath(import.meta.url));
const WIDTH = 80;
const HEIGHT = 50;
const SCALE = 4;
const GAP = 16;
const GENERATIONS = [0, 3, 10];
const ROCK = [0x3a, 0x3f, 0x47];
const OPEN = [0xe9, 0xe4, 0xd8];
const GAP_COLOUR = [0xff, 0xff, 0xff];

function png(width, height, rgb) {
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const out = Buffer.alloc(body.length + 8);
    out.writeUInt32BE(data.length, 0);
    body.copy(out, 4);
    out.writeUInt32BE(crc32(body), body.length + 4);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.set([8, 2, 0, 0, 0], 8); // 8-bit RGB
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) rgb.copy(raw, y * (width * 3 + 1) + 1, y * width * 3, (y + 1) * width * 3);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const panelW = WIDTH * SCALE;
const outW = GENERATIONS.length * panelW + (GENERATIONS.length - 1) * GAP;
const outH = HEIGHT * SCALE;
const rgb = Buffer.alloc(outW * outH * 3);
for (let p = 0; p < outW * outH; p++) rgb.set(GAP_COLOUR, p * 3);

GENERATIONS.forEach((iterations, panel) => {
  const grid = generate({ width: WIDTH, height: HEIGHT, rockPercent: 55, seed: 1, iterations });
  console.log(`seed 1, 55% rock, generation ${iterations}: ${(openShare(grid) * 100).toFixed(1)}% open`);
  const x0 = panel * (panelW + GAP);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < panelW; x++) {
      const cell = grid[Math.floor(y / SCALE) * WIDTH + Math.floor(x / SCALE)];
      rgb.set(cell ? ROCK : OPEN, (y * outW + x0 + x) * 3);
    }
  }
});

writeFileSync(join(here, 'assets/caves-fallback.png'), png(outW, outH, rgb));

const i = process.argv.indexOf('--publish');
if (i !== -1) {
  const site = resolve(here, '../../site');
  execFileSync('node', [join(site, 'tools/publish-deliverable.ts'), here, ...process.argv.slice(i + 1)], {
    stdio: 'inherit',
    cwd: site,
  });
}
