// The Markdown copy of a page, for agents (manifest `formats`). Rendered from the same validated
// Markdoc tree as the HTML, so references keep their resolved links and locators.
import Markdoc, { type RenderableTreeNode, type RenderableTreeNodes } from '@markdoc/markdoc';
import type { Manifest } from './manifest.ts';

const { Tag } = Markdoc;

function text(node: RenderableTreeNodes): string {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(text).join('');
  if (!Tag.isTag(node)) return '';
  return node.children.map(text).join('');
}

function inline(node: RenderableTreeNodes): string {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(inline).join('');
  if (!Tag.isTag(node)) return '';
  const inner = () => node.children.map(inline).join('');
  switch (node.name) {
    case 'strong':
      return `**${inner()}**`;
    case 'em':
      return `*${inner()}*`;
    case 'code':
      return '`' + text(node) + '`';
    case 'a':
      return `[${inner()}](${node.attributes.href})`;
    case 'img':
      return `![${node.attributes.alt}](${node.attributes.src})`;
    case 'picture':
      // The light image stands for the pair.
      return node.children.filter((c) => Tag.isTag(c) && c.name === 'img').map(inline).join('');
    case 'span':
      // A mark is drawn; in words it is the range the middle 8 of 10 fall in. The scale a table
      // prints at a column's head, and the place for an interactive's own value, are left out.
      if (node.attributes.class === 'trait-mark') return `(8 of 10: ${node.attributes['data-p10']} to ${node.attributes['data-p90']})`;
      if (node.attributes.class === 'scale' || node.attributes.class === 'trait-now') return '';
      return inner();
    case 'br':
      return '\n';
    default:
      return inner();
  }
}

function block(node: RenderableTreeNodes, indent = ''): string {
  if (!Tag.isTag(node)) return Array.isArray(node) ? node.map((n) => block(n, indent)).join('') : inline(node);
  const kids = (sep = '') => node.children.map((c) => block(c, indent)).join(sep);
  const m = /^h([1-6])$/.exec(node.name);
  if (m) return `${'#'.repeat(Number(m[1]))} ${inline(node.children)}\n\n`;
  switch (node.name) {
    case 'article':
    case 'section':
    case 'div':
      // A code block's head (its language name) is chrome: the fence names the language.
      if (node.attributes.class === 'code-head') return '';
      return kids();
    case 'header': {
      // A card's head: its title first, then its kind and its opening line.
      const title = node.children.filter((c) => Tag.isTag(c) && /^h[1-6]$/.test(c.name));
      return [...title, ...node.children.filter((c) => !title.includes(c))].map((c) => block(c, indent)).join('');
    }
    case 'table':
      return table(node);
    case 'p':
      if (node.attributes.class === 'trait-key') return '';
      if (node.attributes.class === 'card-kind') return `*${inline(node.children)}*\n\n`;
      if (node.attributes.class === 'step-label') return `**${inline(node.children)}.**\n\n`;
      return `${indent}${inline(node.children)}\n\n`;
    case 'pre': {
      const lang = String(node.attributes['data-language'] ?? '');
      return `\`\`\`${lang}\n${text(node.children)}\n\`\`\`\n\n`;
    }
    case 'ul':
    case 'ol':
      // A step list: each step's label, its text, then its figure.
      if (node.attributes.class === 'steps')
        return node.children
          .map((li) => (Tag.isTag(li) ? [...li.children.filter((c) => !isImage(c)), ...li.children.filter(isImage)].map((c) => block(c, indent)).join('') : ''))
          .join('');
      return (
        node.children
          .map((li, i) => {
            const marker = node.name === 'ol' ? `${i + 1}. ` : '- ';
            const body = Tag.isTag(li) ? li.children.map((c) => (Tag.isTag(c) && ['ul', 'ol'].includes(c.name) ? '\n' + block(c, indent + '  ').trimEnd() : inline(c))).join('') : '';
            return `${indent}${marker}${body.trim()}`;
          })
          .join('\n') + '\n\n'
      );
    case 'blockquote':
      return kids().trimEnd().split('\n').map((l) => `> ${l}`).join('\n') + '\n\n';
    case 'hr':
      return '---\n\n';
    case 'aside':
      return `> **${node.attributes['aria-label']}:** ` + node.children.slice(1).map((c) => block(c).trim()).join('\n> ') + '\n\n';
    case 'figure': {
      if (node.attributes['data-interactive']) {
        const [fallback, caption] = node.children;
        return `*Interactive: ${text(caption)}* (needs JavaScript; [open the page](${'{canonical}'}) to use it). Without it:\n\n${block(fallback)}`;
      }
      return node.children.map((c) => (Tag.isTag(c) && c.name === 'figcaption' ? `*${inline(c.children)}*\n\n` : inline(c) + '\n\n')).join('');
    }
    default:
      return inline(node) + '\n\n';
  }
}

const isImage = (n: RenderableTreeNode) => Tag.isTag(n) && (n.name === 'picture' || n.name === 'img');

/** A table as a pipe table, under its name. A row that heads a group is a row of its own, in bold. */
function table(node: InstanceType<typeof Tag>): string {
  const rows: string[][] = [];
  let label = '';
  let width = 0;
  const walk = (n: RenderableTreeNode) => {
    if (!Tag.isTag(n)) return;
    if (n.name === 'caption') label = text(n);
    else if (n.name === 'tr') {
      const cells = n.children.filter((c) => Tag.isTag(c)).map((c) => inline((c as InstanceType<typeof Tag>).children).replace(/\s*\n\s*/g, ' ').replaceAll('|', '\\|').trim());
      rows.push(n.attributes.class === 'group' ? [`**${cells[0]}**`] : cells);
      width = Math.max(width, cells.length);
    } else n.children.forEach(walk);
  };
  walk(node);
  if (!rows.length) return '';
  const line = (cells: string[]) => `| ${Array.from({ length: width }, (_, i) => cells[i] ?? '').join(' | ')} |`;
  return `${label ? `**${label}**\n\n` : ''}${line(rows[0])}\n${line(Array(width).fill('---'))}\n${rows.slice(1).map(line).join('\n')}\n\n`;
}

/** The "how this was checked" note, in words; `date` formats the ISO date for the medium. */
export function checkedNote(c: NonNullable<Manifest['checked']>, date: (iso: string) => string = (d) => d): string {
  const n = (k: number, one: string, many: string) => `${k} ${k === 1 ? one : many}`;
  const who = c.verifiers.map((v) => `${v.name}, version ${v.version}${v.model ? `, running ${v.model}` : ''}`).join('; ');
  const found = c.supported === c.claims ? (c.claims === 1 ? 'it was supported' : `all ${c.claims} were supported`) : `${c.supported} were supported`;
  return [
    `${n(c.claims, 'claim cites', 'claims cite')} a source. The verifier (${who}) checked each against the passages it cites, last on ${date(c.lastVerified)}: ${found}.`,
    c.measured > 0 ? `${n(c.measured, 'statement', 'statements')} of this page’s own measurements ${c.measured === 1 ? 'is' : 'are'} regenerated and checked by its build.` : '',
    c.openFlags === 0 ? 'No flags are open.' : `${n(c.openFlags, 'flag is', 'flags are')} open.`,
  ].filter(Boolean).join(' ');
}

export function toMarkdown(tree: RenderableTreeNode, manifest: Manifest): string {
  const canonical = `https://christianjmills.com${manifest.urls.canonical}`;
  const updated = manifest.revisions.at(-1)!.date;
  const head = [
    `# ${manifest.title}`,
    '',
    `> ${manifest.summary}`,
    '',
    `Canonical: ${canonical}  `,
    `Published ${manifest.published}, updated ${updated} (version ${manifest.version}).  `,
    `Prose: ${manifest.licenses.prose.spdx}. Code: ${manifest.licenses.code.spdx}.`,
    '',
  ].join('\n');
  const body = block(tree).replaceAll('{canonical}', canonical).replace(/\n{3,}/g, '\n\n');
  const sources = manifest.sources.length
    ? '\n## Sources\n\n' + manifest.sources.map((s) => `- **${s.label}**: ${s.citation} (${'url' in s ? s.url : 'repo' in s ? s.repo : s.path})`).join('\n') + '\n' +
      (manifest.checked ? `\n**How this was checked.** ${checkedNote(manifest.checked)}\n` : '')
    : '';
  return `${head}\n${body.trim()}\n${sources}`;
}
