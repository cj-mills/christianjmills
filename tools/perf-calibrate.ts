// Reads the reference for the performance check's CPU (tools/lib/cpu.ts) on this machine and
// writes tools/perf-reference.json. Run it only on the reference machine, the user's workstation,
// and again when Playwright's Chromium changes; commit the result.
import { chromium } from 'playwright';
import { benchmark, writeReference } from './lib/cpu.ts';

const browser = await chromium.launch();
try {
  // The check's own measure (a median of three benchmarks), read three times: the median is the reference.
  const readings: number[] = [];
  for (let i = 0; i < 3; i++) readings.push(await benchmark(browser));
  readings.sort((a, b) => a - b);
  const benchmarkMs = Math.round(readings[1] * 10) / 10;
  await writeReference({ benchmarkMs, chromium: browser.version(), measured: new Date().toISOString().slice(0, 10) });
  console.log(`reference: ${benchmarkMs} ms in Chromium ${browser.version()} (readings ${readings.map((r) => r.toFixed(1)).join(', ')} ms)`);
} finally {
  await browser.close();
}
