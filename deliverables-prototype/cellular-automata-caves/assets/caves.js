// Cellular-automata caves: random rock, then a smoothing rule applied a few times.
// Self-contained ES module. The page mounts it through `mount(root)`; build.mjs imports the
// same functions to render the no-JavaScript fallback image.

/** Seeded PRNG (mulberry32), so a seed always gives the same map. */
export function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 1 is rock, 0 is open floor. The outer ring is always rock, so caves stay closed. */
export function noise(width, height, rockPercent, seed) {
  const next = random(seed);
  const grid = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const edge = x === 0 || y === 0 || x === width - 1 || y === height - 1;
      grid[y * width + x] = edge || next() * 100 < rockPercent ? 1 : 0;
    }
  }
  return grid;
}

export function rockNeighbours(grid, width, x, y) {
  let n = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dx !== 0 || dy !== 0) n += grid[(y + dy) * width + (x + dx)];
    }
  }
  return n;
}

/** One generation: rock where more than four of the eight neighbours are rock, or none are. */
export function step(grid, width, height) {
  const out = grid.slice();
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const n = rockNeighbours(grid, width, x, y);
      out[y * width + x] = n > 4 || n === 0 ? 1 : 0;
    }
  }
  return out;
}

export function generate({ width, height, rockPercent, seed, iterations }) {
  let grid = noise(width, height, rockPercent, seed);
  for (let i = 0; i < iterations; i++) grid = step(grid, width, height);
  return grid;
}

export function openShare(grid) {
  let open = 0;
  for (const cell of grid) open += 1 - cell;
  return open / grid.length;
}

const WIDTH = 80;
const HEIGHT = 50;
const MAX_ITERATIONS = 10;

function field(label, input, output) {
  const wrap = document.createElement('label');
  wrap.className = 'field';
  const name = document.createElement('span');
  name.textContent = label;
  wrap.append(name, input);
  if (output) wrap.append(output);
  return wrap;
}

function range(min, max, value) {
  const input = document.createElement('input');
  Object.assign(input, { type: 'range', min, max, value, step: 1 });
  return input;
}

/** Mounts the interactive into `root`. Colours come from the site's design tokens. */
export function mount(root) {
  const state = { rockPercent: 55, seed: 1, iterations: 0 };
  let frames = [];

  const canvas = document.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  canvas.setAttribute('role', 'img');
  canvas.style.cssText = 'width:100%;height:auto;aspect-ratio:80/50;image-rendering:pixelated;display:block';

  const rock = range(40, 65, state.rockPercent);
  const rockOut = document.createElement('output');
  const generation = range(0, MAX_ITERATIONS, state.iterations);
  const generationOut = document.createElement('output');
  const seed = document.createElement('input');
  Object.assign(seed, { type: 'number', min: 1, max: 999999, value: state.seed, inputMode: 'numeric' });
  const reroll = document.createElement('button');
  reroll.type = 'button';
  reroll.textContent = 'New seed';
  const status = document.createElement('p');
  status.className = 'status';
  status.setAttribute('aria-live', 'polite');

  const controls = document.createElement('div');
  controls.className = 'controls';
  controls.append(
    field('Starting rock', rock, rockOut),
    field('Generation', generation, generationOut),
    field('Seed', seed),
    reroll,
  );
  root.append(canvas, controls, status);

  function colours() {
    const css = getComputedStyle(root);
    return {
      rock: css.getPropertyValue('--interactive-ink').trim() || '#333',
      open: css.getPropertyValue('--interactive-paper').trim() || '#eee',
    };
  }

  function recompute() {
    frames = [noise(WIDTH, HEIGHT, state.rockPercent, state.seed)];
    for (let i = 0; i < MAX_ITERATIONS; i++) frames.push(step(frames[i], WIDTH, HEIGHT));
  }

  function draw() {
    const grid = frames[state.iterations];
    const { rock: ink, open: paper } = colours();
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = paper;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = ink;
    for (let i = 0; i < grid.length; i++) if (grid[i]) ctx.fillRect(i % WIDTH, Math.floor(i / WIDTH), 1, 1);
    const open = Math.round(openShare(grid) * 100);
    rockOut.value = `${state.rockPercent}%`;
    generationOut.value = String(state.iterations);
    const text = `Seed ${state.seed}, ${state.rockPercent}% starting rock, generation ${state.iterations} of ${MAX_ITERATIONS}: ${open}% of the map is open floor.`;
    canvas.setAttribute('aria-label', `Cave map, ${WIDTH} by ${HEIGHT} tiles. ${text}`);
    status.textContent = text;
  }

  rock.addEventListener('input', () => {
    state.rockPercent = Number(rock.value);
    recompute();
    draw();
  });
  generation.addEventListener('input', () => {
    state.iterations = Number(generation.value);
    draw();
  });
  seed.addEventListener('change', () => {
    const value = Math.trunc(Number(seed.value));
    if (!(value >= 1 && value <= 999999)) return;
    state.seed = value;
    recompute();
    draw();
  });
  reroll.addEventListener('click', () => {
    state.seed = 1 + Math.floor(Math.random() * 999999);
    seed.value = String(state.seed);
    recompute();
    draw();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  recompute();
  draw();
}
