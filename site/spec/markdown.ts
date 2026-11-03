// The Markdown copy of a page, for agents (manifest `formats`). Rendered from the same validated
// Markdoc tree as the HTML, so references keep their resolved links and locators.
import Markdoc, { type RenderableTreeNode } from '@markdoc/markdoc';
import type { Manifest } from './manifest.ts';

const { Tag } = Markdoc;

function text(node: RenderableTreeNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(text).join('');
  if (!Tag.isTag(node)) return '';
  return node.children.map(text).join('');
}

function inline(node: RenderableTreeNode): string {
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
    case 'br':
      return '\n';
    default:
      return inner();
  }
}

function block(node: RenderableTreeNode, indent = ''): string {
  if (!Tag.isTag(node)) return Array.isArray(node) ? node.map((n) => block(n, indent)).join('') : inline(node);
  const kids = (sep = '') => node.children.map((c) => block(c, indent)).join(sep);
  const m = /^h([1-6])$/.exec(node.name);
  if (m) return `${'#'.repeat(Number(m[1]))} ${inline(node.children)}\n\n`;
  switch (node.name) {
    case 'article':
    case 'section':
    case 'div':
      return kids();
    case 'p':
      return `${indent}${inline(node.children)}\n\n`;
    case 'pre': {
      const lang = String(node.attributes['data-language'] ?? '');
      return `\`\`\`${lang}\n${text(node.children)}\n\`\`\`\n\n`;
    }
    case 'ul':
    case 'ol':
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
    ? '\n## Sources\n\n' + manifest.sources.map((s) => `- **${s.label}**: ${s.citation} (${'url' in s ? s.url : 'repo' in s ? s.repo : s.path})`).join('\n') + '\n'
    : '';
  return `${head}\n${body.trim()}\n${sources}`;
}
