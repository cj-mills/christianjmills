// The CPU the performance check measures on (hub decision
// 2026-11-12-christianjmills-calibrated-cpu-slowdown): four times slower than the reference machine,
// the user's workstation, wherever the check runs. Chromium can only slow its CPU by a factor of
// the machine it runs on, so the check first times a fixed piece of work in the browser and sets
// the factor from how that compares with the reference's time:
//
//   rate = 4 × the reference's time ÷ this machine's time
//
// A runner twice as slow as the reference gets a rate of 2; a machine twice as fast, 8. The
// reference is read on the reference machine with `npm run perf:calibrate` and kept in
// tools/perf-reference.json; it is read again when Playwright's Chromium changes.
import { readFile, writeFile } from 'node:fs/promises';
import type { Browser } from 'playwright';

export const SLOWDOWN = 4;
const REFERENCE = new URL('../perf-reference.json', import.meta.url);

export interface Reference {
  /** The median time of the work, in ms, on the reference machine. */
  benchmarkMs: number;
  /** The Chromium it was read in: a different engine may run the work at another speed. */
  chromium: string;
  measured: string;
}

/**
 * The median time of a fixed piece of work in a fresh page, unthrottled: the kinds of work a
 * module's mount does (numbers, arrays, objects, strings and a little DOM), about 150 ms on the
 * reference machine so a timer's resolution and a moment's noise matter little, run 7 times after
 * one to warm up.
 */
async function benchmarkOnce(browser: Browser): Promise<number> {
  const context = await browser.newContext();
  const tab = await context.newPage();
  await tab.setContent('<!doctype html><main></main>');
  const times = await tab.evaluate(() => {
    const work = () => {
      let acc = 0;
      for (let round = 0; round < 6; round++) acc += unit();
      return acc;
    };
    const unit = () => {
      let acc = 0;
      const grid = new Float64Array(256 * 256);
      for (let pass = 0; pass < 12; pass++) {
        for (let i = 0; i < grid.length; i++) grid[i] = Math.sin(i * 0.37 + pass) * Math.sqrt(i + 1);
        acc += grid.reduce((a, b) => a + b, 0);
      }
      const objects = Array.from({ length: 60000 }, (_, i) => ({ id: i, key: `k${i % 977}`, value: i * 1.5 }));
      const byKey = new Map<string, number>();
      for (const o of objects) byKey.set(o.key, (byKey.get(o.key) ?? 0) + o.value);
      objects.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : a.id - b.id));
      let text = '';
      for (let i = 0; i < 20000; i++) text += String.fromCharCode(97 + (i % 26));
      acc += text.split('').reverse().join('').length + byKey.size;
      const main = document.querySelector('main')!;
      for (let i = 0; i < 400; i++) {
        const div = document.createElement('div');
        div.textContent = `cell ${i}`;
        main.append(div);
      }
      acc += main.getBoundingClientRect().height;
      main.replaceChildren();
      return acc;
    };
    work();
    const out: number[] = [];
    for (let i = 0; i < 7; i++) {
      const t = performance.now();
      work();
      out.push(performance.now() - t);
    }
    return out;
  });
  await context.close();
  return times.sort((a, b) => a - b)[Math.floor(times.length / 2)];
}

/** The median of three benchmarks, each in a fresh page: the reference and the check read it alike. */
export async function benchmark(browser: Browser): Promise<number> {
  const readings: number[] = [];
  for (let i = 0; i < 3; i++) readings.push(await benchmarkOnce(browser));
  return readings.sort((a, b) => a - b)[1];
}

export async function readReference(): Promise<Reference> {
  return JSON.parse(await readFile(REFERENCE, 'utf8'));
}

export async function writeReference(ref: Reference) {
  await writeFile(REFERENCE, JSON.stringify(ref, null, 2) + '\n');
}

/** The throttling rate that makes this machine's CPU four times slower than the reference's. */
export async function calibratedRate(browser: Browser): Promise<{ rate: number; hostMs: number; reference: Reference }> {
  const reference = await readReference();
  if (reference.chromium !== browser.version()) {
    throw new Error(
      `tools/perf-reference.json was read in Chromium ${reference.chromium}, and the check runs ${browser.version()}: ` +
        'read the reference again on the reference machine with npm run perf:calibrate',
    );
  }
  const hostMs = await benchmark(browser);
  const rate = (SLOWDOWN * reference.benchmarkMs) / hostMs;
  // Chromium can slow a CPU, not speed it up: a machine over four times slower than the reference
  // can't stand in for it.
  if (rate < 1) {
    throw new Error(`this machine runs the benchmark in ${hostMs.toFixed(0)} ms, over ${SLOWDOWN} times the reference's ${reference.benchmarkMs.toFixed(0)} ms, so it can't measure the budgets`);
  }
  return { rate, hostMs, reference };
}
