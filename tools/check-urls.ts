// The URL check (hub decision 2026-11-03-christianjmills-legacy-urls): every address of the old
// site, every redirect and every page of the new site answers as tools/archive-merge.ts says it
// must, asked of a real server, not of a model of one:
//
//   node tools/check-urls.ts               serves dist/ with `wrangler dev`, locally, with no account
//   node tools/check-urls.ts <base URL>    asks a deployed version, e.g. its preview URL, or the live
//                                          domain at cutover
//
// An address answers 200 itself, or a 301 to where it should go, which must answer 200: in the site,
// or, for a post retired to GitHub, there (CHECK_URLS_EXTERNAL=0 leaves those out, offline). The
// preview's hosts must answer with `X-Robots-Tag: noindex` and the live domain without it.
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { request } from 'node:http';
import { createServer } from 'node:net';
import { join } from 'node:path';
import { ROOT } from './lib/archive.ts';
import type { Expectation } from './lib/legacy-plan.ts';

const arg = process.argv[2];
const external = process.env.CHECK_URLS_EXTERNAL !== '0';
const expectations: Expectation[] = JSON.parse(await readFile(join(ROOT, '.build/urls.json'), 'utf8'));
const failures: string[] = [];

let base: URL;
let stop = () => {};
if (arg) base = new URL(arg);
else {
  const port = await new Promise<number>((resolve) => {
    const s = createServer().listen(0, '127.0.0.1', () => {
      const { port } = s.address() as { port: number };
      s.close(() => resolve(port));
    });
  });
  const dev = spawn(join(ROOT, 'node_modules/.bin/wrangler'), ['dev', '--ip', '127.0.0.1', '--port', String(port), '--log-level', 'warn'], {
    cwd: ROOT,
    env: { ...process.env, WRANGLER_SEND_METRICS: 'false' },
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  stop = () => dev.kill();
  base = new URL(`http://127.0.0.1:${port}/`);
  for (let i = 0; ; i++) {
    try {
      await fetch(new URL('/robots.txt', base));
      break;
    } catch {
      if (i > 120) throw new Error('wrangler dev did not start');
      await new Promise((r) => setTimeout(r, 500));
    }
  }
}

try {
  const finals = new Map<string, Promise<string | null>>();
  /** Where a redirect lands must answer 200: null when it does, or what it answered. */
  const landing = (to: string) => {
    if (!finals.has(to)) {
      finals.set(
        to,
        (async () => {
          if (to.startsWith('/')) {
            const res = await fetch(new URL(to, base), { redirect: 'manual' });
            return res.status === 200 ? null : `${res.status}`;
          }
          for (let attempt = 0; ; attempt++) {
            const res = await fetch(to, { redirect: 'follow' }).catch((e) => ({ status: String(e) }));
            if (res.status === 200) return null;
            if (attempt === 2) return `${res.status}`;
            await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
          }
        })(),
      );
    }
    return finals.get(to)!;
  };

  const one = async (e: Expectation) => {
    const res = await fetch(new URL(e.path, base), { redirect: 'manual' });
    // A page of the old site carries the archive banner; a page of the new site doesn't.
    const page = e.expect === 200 && res.status === 200 && (res.headers.get('content-type') ?? '').startsWith('text/html');
    const body = page ? await res.text() : (await res.body?.cancel(), '');
    if (e.expect === 200) {
      if (res.status !== 200) failures.push(`${e.path} answered ${res.status}, not 200 (${e.why})`);
      else if (page && body.includes('data-archived') !== e.why.endsWith('in the copy')) {
        failures.push(`${e.path} ${e.why.endsWith('in the copy') ? 'is an old page without' : 'is a new page with'} the archive banner`);
      }
      return;
    }
    const location = res.headers.get('location');
    const wanted = e.expect.startsWith('/') ? new URL(e.expect, base).href : e.expect;
    if (res.status !== 301 || !location || new URL(location, base).href !== wanted) {
      failures.push(`${e.path} answered ${res.status}${location ? ` to ${location}` : ''}, not 301 to ${e.expect} (${e.why})`);
      return;
    }
    if (!e.expect.startsWith('/') && !external) return;
    const landed = await landing(e.expect);
    if (landed) failures.push(`${e.path} redirects to ${e.expect}, which answered ${landed} (${e.why})`);
  };
  // A few at a time: wrangler dev and GitHub both prefer it.
  const queue = [...expectations];
  await Promise.all(Array.from({ length: 8 }, async () => { for (let e; (e = queue.shift()); ) await one(e); }));

  // The preview's hosts are kept out of search engines; the live domain isn't.
  const robots = async (host?: string) =>
    host
      ? new Promise<string | undefined>((resolve, reject) => {
          // fetch() won't send another Host, so this asks wrangler dev as each host would.
          request(new URL('/', base), { headers: { host } }, (res) => {
            res.resume();
            resolve(res.headers['x-robots-tag'] as string | undefined);
          }).on('error', reject).end();
        })
      : (await fetch(new URL('/', base))).headers.get('x-robots-tag') ?? undefined;
  const hosts: [string | undefined, boolean][] = arg
    ? [[undefined, base.hostname.endsWith('.workers.dev')]]
    : [['christianjmills-site.millscj.workers.dev', true], ['0123abcd-christianjmills-site.millscj.workers.dev', true], ['christianjmills.com', false]];
  for (const [host, noindex] of hosts) {
    const tag = await robots(host);
    if ((tag === 'noindex') !== noindex) failures.push(`${host ?? base.host} answered X-Robots-Tag ${tag ?? '(none)'}; it should ${noindex ? '' : 'not '}be noindex`);
  }
} finally {
  stop();
}

const redirects = expectations.filter((e) => e.expect !== 200).length;
if (failures.length) {
  console.error(`URL check failed on ${base.origin} (${failures.length} of ${expectations.length} addresses):\n  ${failures.slice(0, 60).join('\n  ')}${failures.length > 60 ? `\n  …and ${failures.length - 60} more` : ''}`);
  process.exit(1);
}
console.log(
  `URL check passed on ${base.origin}: ${expectations.length - redirects} addresses answer 200, ${redirects} redirect with a 301 to a page that does${external ? '' : ' (GitHub targets not followed)'}, and noindex is on the preview's hosts only`,
);
