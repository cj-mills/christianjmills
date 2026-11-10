// The site's own text is in American English, with dates in American order (hub decision
// 2026-11-09-american-english-written-from-the-start). Run before every build; any British
// spelling or day-first date in the files below fails it, with the file and line. Names that keep
// their own spelling are listed in spec/english/allow.txt. Deliverables are checked by the publish
// command (spec/english.ts).
import { readdir, readFile } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkText } from '../spec/english.ts';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const INCLUDE = ['README.md', 'astro.config.mjs', 'deliverables.config.ts', 'wrangler.jsonc', 'spec', 'src', 'tools'];
// Vendored files keep their authors' spelling, and the allow list holds the names that keep theirs.
const EXCLUDE = ['spec/english/', 'src/styles/themes/', 'src/styles/tokens.css'];
const TEXT = /\.(ts|mjs|js|astro|css|json|jsonc|md|mdoc|sh|txt)$/;

const allow = (await readFile(join(ROOT, 'spec/english/allow.txt'), 'utf8'))
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l && !l.startsWith('#'));

const files: string[] = [];
for (const name of INCLUDE) {
  const full = join(ROOT, name);
  if (TEXT.test(name)) files.push(full);
  else for (const e of await readdir(full, { recursive: true, withFileTypes: true })) if (e.isFile()) files.push(join(e.parentPath, e.name));
}

const problems: string[] = [];
let checked = 0;
for (const file of files.sort()) {
  const rel = relative(ROOT, file).split(sep).join('/');
  if (!TEXT.test(rel) || EXCLUDE.some((x) => rel.startsWith(x))) continue;
  checked++;
  for (const f of checkText(await readFile(file, 'utf8'), allow)) problems.push(`${rel}:${f.line}: ${f.message}`);
}

if (problems.length) {
  console.error(`American English check failed (${problems.length}):\n  ${problems.join('\n  ')}`);
  process.exit(1);
}
console.log(`American English check passed: ${checked} files`);
