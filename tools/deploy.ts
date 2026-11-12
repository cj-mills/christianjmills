// Deploys through Worker versions (hub decision 2026-11-03-christianjmills-launch-gate): upload the
// build as a new version, check it at its own preview URL, then promote it. Uploading changes
// nothing anyone is served; promoting sends all of the Worker's traffic to the version. Run from this
// machine, under the token 1Password injects:
//
//   npm run deploy:upload                 builds, checks, uploads a version, runs the URL check on its preview URL
//   npm run deploy:promote -- <version>   sends the Worker's traffic to that version
//
// Until christianjmills.com is attached to the Worker, its traffic is only workers.dev's, the preview.
// Once it is, promoting is the live site's deploy.
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ROOT } from './lib/archive.ts';

const [command, versionArg] = process.argv.slice(2);
const wrangler = join(ROOT, 'node_modules/.bin/wrangler');
const recordPath = join(ROOT, '.build/version.json');
const git = (...a: string[]) => execFileSync('git', a, { cwd: ROOT }).toString().trim();
if (!process.env.CLOUDFLARE_API_TOKEN) throw new Error('deploys need Cloudflare credentials; run through tools/cloudflare.sh');

interface Uploaded {
  version: string;
  preview: string;
  commit: string;
  checked: boolean;
}

if (command === 'upload') {
  // A version is a commit: what is uploaded can be found again.
  if (git('status', '--porcelain')) throw new Error('the checkout has changes that are not committed; a version is uploaded from a commit');
  const commit = git('rev-parse', 'HEAD');
  const tmp = await mkdtemp(join(tmpdir(), 'deploy-'));
  let entry: { version_id: string; preview_url?: string | null } | undefined;
  try {
    // Wrangler writes what it did, one JSON object a line, to this file: the version's id and its preview URL.
    const output = join(tmp, 'wrangler-output.ndjson');
    execFileSync(wrangler, ['versions', 'upload', '--message', `${commit.slice(0, 12)} ${git('log', '-1', '--format=%s').slice(0, 80)}`, '--tag', commit.slice(0, 12)], {
      cwd: ROOT,
      stdio: 'inherit',
      env: { ...process.env, WRANGLER_OUTPUT_FILE_PATH: output, WRANGLER_SEND_METRICS: 'false' },
    });
    entry = (await readFile(output, 'utf8'))
      .split('\n')
      .filter(Boolean)
      .map((l) => JSON.parse(l))
      .find((e) => e.type === 'version-upload');
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
  if (!entry?.version_id) throw new Error('wrangler did not report the version it uploaded');
  if (!entry.preview_url) throw new Error(`version ${entry.version_id} has no preview URL; wrangler.jsonc sets "preview_urls": true`);
  const record: Uploaded = { version: entry.version_id, preview: entry.preview_url, commit, checked: false };
  await writeFile(recordPath, JSON.stringify(record, null, 2) + '\n');

  // A new version's URL can take a moment to answer.
  for (let i = 0; ; i++) {
    const res = await fetch(new URL('/robots.txt', record.preview)).catch(() => undefined);
    if (res?.ok) break;
    if (i === 60) throw new Error(`${record.preview} did not answer within a minute`);
    await new Promise((r) => setTimeout(r, 1000));
  }
  const check = spawnSync(process.execPath, [join(ROOT, 'tools/check-urls.ts'), record.preview], { cwd: ROOT, stdio: 'inherit' });
  if (check.status !== 0) {
    console.error(`version ${record.version} is uploaded but failed the URL check at ${record.preview}; it is not promoted`);
    process.exit(1);
  }
  record.checked = true;
  await writeFile(recordPath, JSON.stringify(record, null, 2) + '\n');
  console.log(`version ${record.version} of ${commit.slice(0, 12)} passed the URL check at ${record.preview}`);
  console.log(`look at it there, then promote it: npm run deploy:promote -- ${record.version}`);
} else if (command === 'promote') {
  if (!versionArg) throw new Error('usage: node tools/deploy.ts promote <version id>');
  // Only a version this checkout uploaded and checked is promoted.
  const record: Uploaded = JSON.parse(await readFile(recordPath, 'utf8').catch(() => '{}'));
  if (record.version !== versionArg || !record.checked) {
    throw new Error(`${versionArg} is not the version this checkout last uploaded and checked (${record.version ?? 'none'}); upload it with npm run deploy:upload`);
  }
  execFileSync(wrangler, ['versions', 'deploy', `${versionArg}@100%`, '--yes', '--message', `promote ${record.commit.slice(0, 12)}`], {
    cwd: ROOT,
    stdio: 'inherit',
    env: { ...process.env, WRANGLER_SEND_METRICS: 'false' },
  });
  console.log(`promoted ${versionArg} (${record.commit.slice(0, 12)}): the Worker's traffic now goes to it`);
} else {
  console.error('usage: node tools/deploy.ts upload | promote <version id>');
  process.exit(2);
}
