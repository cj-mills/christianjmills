// The site's Markdown dialect, format 0.1 (draft): Markdoc with the tags and node rules below.
// Deliverables write content in it; the site validates it at build time and owns how it renders.
// A config is built per bundle, because references and assets are checked against that bundle.
import Markdoc, { type Config, type Node, type RenderableTreeNode, type ValidationError } from '@markdoc/markdoc';
import type { HighlighterGeneric } from 'shiki';
import type { Manifest, Source } from './manifest.ts';

// The package's ESM entry exports only the default object.
const { Tag } = Markdoc;
type Tag = InstanceType<typeof Tag>;

export interface BundleContext {
  manifest: Manifest;
  files: Map<string, Uint8Array>;
  /** Public URL prefix the bundle's files are served under, ending in "/". */
  assetBase: string;
  highlighter: HighlighterGeneric<any, any>;
}

export interface Reference {
  source: string;
  locator: string;
  href: string;
}

export interface Heading {
  depth: number;
  slug: string;
  text: string;
}

const RANGE = /^(\d+)(?:-(\d+))?$/;
const TIME = /^(\d{1,2}:)?\d{1,2}:\d{2}$/;

function error(id: string, message: string): ValidationError {
  return { id, level: 'error', message };
}

function parseRange(value: string): [number, number] | null {
  const m = RANGE.exec(value);
  if (!m) return null;
  const a = Number(m[1]);
  const b = m[2] ? Number(m[2]) : a;
  return a >= 1 && b >= a ? [a, b] : null;
}

function seconds(t: string): number {
  return t.split(':').reduce((acc, part) => acc * 60 + Number(part), 0);
}

function lineCount(bytes: Uint8Array): number {
  const text = new TextDecoder().decode(bytes);
  return text.endsWith('\n') ? text.split('\n').length - 1 : text.split('\n').length;
}

function pngSize(bytes: Uint8Array): { width: number; height: number } | null {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24 || !sig.every((b, i) => bytes[i] === b)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16), height: view.getUint32(20) };
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, '')
    .trim()
    .replace(/\s+/g, '-');
}

function textOf(node: Node): string {
  if (node.type === 'text') return String(node.attributes.content ?? '');
  return node.children.map(textOf).join('');
}

/** Which locator attributes each source kind accepts, and which it requires. */
const LOCATORS: Record<Source['kind'] | 'this', { allowed: string[]; required: string[] }> = {
  video: { allowed: ['t'], required: ['t'] },
  document: { allowed: ['page'], required: ['page'] },
  code: { allowed: ['path', 'lines'], required: ['path', 'lines'] },
  web: { allowed: ['anchor'], required: [] },
  image: { allowed: ['region'], required: ['region'] },
  this: { allowed: ['path', 'lines'], required: ['path', 'lines'] },
};

/** Builds the Markdoc config for one bundle. `refs` and `headings` collect what the page cites and its outline. */
export function dialect(ctx: BundleContext) {
  const { manifest, files, assetBase } = ctx;
  const sources = new Map(manifest.sources.map((s) => [s.id, s]));
  const interactives = new Map(manifest.interactives.map((i) => [i.id, i]));
  const refs: Reference[] = [];
  const headings: Heading[] = [];
  const usedIds = new Set<string>();

  function checkAsset(src: unknown): ValidationError[] {
    if (typeof src !== 'string' || /^[a-z]+:/i.test(src) || src.startsWith('/')) {
      return [error('asset-external', `images come from the bundle; "${src}" is not a bundle path`)];
    }
    const bytes = files.get(src);
    if (!bytes) return [error('asset-missing', `"${src}" is not in the bundle`)];
    if (!pngSize(bytes)) return [error('asset-format', `"${src}": format 0.1 accepts PNG images only`)];
    return [];
  }

  function image(src: string, alt: string, title?: string): Tag {
    const size = pngSize(files.get(src)!)!;
    return new Tag('img', {
      src: assetBase + src,
      alt,
      title,
      width: size.width,
      height: size.height,
      loading: 'lazy',
      decoding: 'async',
    });
  }

  function locate(attrs: Record<string, any>): { locator: string; href: string } {
    const id = attrs.source as string;
    if (id === 'this') {
      const [a, b] = parseRange(attrs.lines)!;
      return { locator: `${attrs.path}, lines ${a}–${b}`, href: assetBase + attrs.path };
    }
    const s = sources.get(id)!;
    switch (s.kind) {
      case 'video': {
        const [start, end] = String(attrs.t).split('-');
        const u = new URL(s.url);
        if (/(^|\.)youtube\.com$|^youtu\.be$/.test(u.hostname)) u.searchParams.set('t', `${seconds(start)}s`);
        else u.hash = `t=${seconds(start)}${end ? ',' + seconds(end) : ''}`;
        return { locator: end ? `${start}–${end}` : start, href: u.href };
      }
      case 'document': {
        const [a, b] = parseRange(String(attrs.page))!;
        return { locator: a === b ? `p. ${a}` : `pp. ${a}–${b}`, href: `${s.url}#page=${a}` };
      }
      case 'code': {
        const [a, b] = parseRange(attrs.lines)!;
        const repo = s.repo.replace(/\/$/, '');
        return {
          locator: `${attrs.path}, lines ${a}–${b}`,
          href: `${repo}/blob/${s.commit}/${attrs.path}#L${a}-L${b}`,
        };
      }
      case 'web':
        return { locator: attrs.anchor ? `§ ${attrs.anchor}` : '', href: attrs.anchor ? `${s.url}#${attrs.anchor}` : s.url };
      case 'image':
        return { locator: `region ${attrs.region}`, href: assetBase + s.path };
    }
  }

  const config: Config = {
    tags: {
      // Inline anchor to a manifest source, with a locator. A reference that doesn't resolve fails the build.
      ref: {
        inline: true,
        selfClosing: true,
        attributes: {
          source: { type: String, required: true },
          t: { type: String },
          page: { type: String },
          lines: { type: String },
          path: { type: String },
          region: { type: String },
          anchor: { type: String },
        },
        validate(node) {
          const attrs = node.attributes;
          const id = attrs.source;
          const kind = id === 'this' ? 'this' : sources.get(id)?.kind;
          if (!kind) return [error('ref-unknown-source', `{% ref %} cites "${id}", which is not in the manifest's sources`)];
          const rule = LOCATORS[kind];
          const errs: ValidationError[] = [];
          for (const k of ['t', 'page', 'lines', 'path', 'region', 'anchor']) {
            if (attrs[k] !== undefined && !rule.allowed.includes(k))
              errs.push(error('ref-locator', `"${k}" is not a locator for a ${kind} source ("${id}")`));
          }
          for (const k of rule.required)
            if (attrs[k] === undefined) errs.push(error('ref-locator', `a reference to ${kind} source "${id}" needs "${k}"`));
          if (errs.length) return errs;
          if (attrs.t !== undefined) {
            const parts = String(attrs.t).split('-');
            if (parts.length > 2 || !parts.every((p: string) => TIME.test(p)))
              errs.push(error('ref-locator', `t="${attrs.t}" is not mm:ss or mm:ss-mm:ss`));
            else if (parts.length === 2 && seconds(parts[1]) <= seconds(parts[0]))
              errs.push(error('ref-locator', `t="${attrs.t}" ends before it starts`));
          }
          if (attrs.page !== undefined) {
            const r = parseRange(String(attrs.page));
            const s = sources.get(id);
            if (!r) errs.push(error('ref-locator', `page="${attrs.page}" is not a page or page range`));
            else if (s?.kind === 'document' && r[1] > s.pages)
              errs.push(error('ref-locator', `page="${attrs.page}" is past the end of "${id}" (${s.pages} pages)`));
          }
          if (attrs.lines !== undefined && !parseRange(attrs.lines))
            errs.push(error('ref-locator', `lines="${attrs.lines}" is not a line or line range`));
          if (attrs.region !== undefined && !/^\d+,\d+,\d+,\d+$/.test(attrs.region))
            errs.push(error('ref-locator', `region="${attrs.region}" is not x,y,width,height`));
          if (kind === 'this' && !errs.length) {
            const bytes = files.get(attrs.path);
            if (!bytes) errs.push(error('ref-locator', `"${attrs.path}" is not in the bundle`));
            else if (parseRange(attrs.lines)![1] > lineCount(bytes))
              errs.push(error('ref-locator', `lines="${attrs.lines}" is past the end of ${attrs.path} (${lineCount(bytes)} lines)`));
          }
          return errs;
        },
        transform(node) {
          const attrs = node.attributes;
          const { locator, href } = locate(attrs);
          refs.push({ source: attrs.source, locator, href });
          const label = attrs.source === 'this' ? 'this page’s code' : sources.get(attrs.source)!.label;
          // One reference per claim in the tree, which the Markdown copy keeps; the HTML merges a
          // block's references into one marker at its end (mergeReferences, below).
          return new Tag('a', { class: 'ref', href, 'data-source': attrs.source, 'data-label': label, 'data-locator': locator }, [
            `[${label}${locator ? ', ' + locator : ''}]`,
          ]);
        },
      },

      // A declared mount for an interactive. Its children are the fallback, shown without JavaScript.
      interactive: {
        attributes: {
          id: { type: String, required: true },
          caption: { type: String, required: true },
        },
        validate(node) {
          const id = node.attributes.id;
          const errs: ValidationError[] = [];
          const decl = interactives.get(id);
          if (!decl) errs.push(error('interactive-unknown', `{% interactive id="${id}" %} is not declared in the manifest`));
          else if (!files.get(decl.module)) errs.push(error('interactive-module', `module "${decl.module}" is not in the bundle`));
          if (!node.children.length) errs.push(error('interactive-fallback', `{% interactive id="${id}" %} needs fallback content`));
          return errs;
        },
        transform(node, config) {
          const decl = interactives.get(node.attributes.id)!;
          return new Tag(
            'figure',
            {
              class: 'interactive',
              'data-interactive': decl.id,
              'data-module': assetBase + decl.module,
            },
            [
              new Tag('div', { class: 'interactive-fallback' }, node.transformChildren(config)),
              new Tag('figcaption', {}, [node.attributes.caption]),
            ],
          );
        },
      },

      figure: {
        selfClosing: true,
        attributes: {
          src: { type: String, required: true },
          alt: { type: String, required: true },
          caption: { type: String },
        },
        validate(node) {
          const errs = checkAsset(node.attributes.src);
          if (!String(node.attributes.alt).trim()) errs.push(error('figure-alt', 'a figure needs alt text'));
          return errs;
        },
        transform(node) {
          const { src, alt, caption } = node.attributes;
          return new Tag('figure', {}, [image(src, alt), ...(caption ? [new Tag('figcaption', {}, [caption])] : [])]);
        },
      },

      callout: {
        attributes: {
          type: { type: String, matches: ['note', 'tip', 'warning'], default: 'note' },
          title: { type: String },
        },
        transform(node, config) {
          const { type, title } = node.attributes;
          const label = title ?? type[0].toUpperCase() + type.slice(1);
          return new Tag('aside', { class: `callout callout-${type}`, 'aria-label': label }, [
            new Tag('p', { class: 'callout-title' }, [label]),
            ...node.transformChildren(config),
          ]);
        },
      },
    },

    nodes: {
      heading: {
        children: ['inline'],
        attributes: { level: { type: Number, required: true } },
        validate(node) {
          return node.attributes.level === 1
            ? [error('heading-level', 'the page title is the h1; content headings start at ##')]
            : [];
        },
        transform(node, config) {
          const text = textOf(node);
          let id = slugify(text);
          for (let n = 2; usedIds.has(id); n++) id = `${slugify(text)}-${n}`;
          usedIds.add(id);
          headings.push({ depth: node.attributes.level, slug: id, text });
          return new Tag(`h${node.attributes.level}`, { id }, node.transformChildren(config));
        },
      },

      image: {
        attributes: { src: { type: String, required: true }, alt: { type: String }, title: { type: String } },
        validate(node) {
          const errs = checkAsset(node.attributes.src);
          if (!String(node.attributes.alt ?? '').trim()) errs.push(error('image-alt', `image "${node.attributes.src}" needs alt text`));
          return errs;
        },
        transform(node) {
          return image(node.attributes.src, node.attributes.alt, node.attributes.title);
        },
      },

      fence: {
        attributes: { language: { type: String }, content: { type: String } },
        validate(node) {
          const lang = node.attributes.language;
          if (!lang) return [error('fence-language', 'a code block names its language')];
          if (!ctx.highlighter.getLoadedLanguages().includes(lang))
            return [error('fence-language', `"${lang}" is not a language the site highlights`)];
          return [];
        },
        transform(node) {
          const hast = ctx.highlighter.codeToHast(node.attributes.content.replace(/\n$/, ''), {
            lang: node.attributes.language,
            theme: 'site',
          });
          const lang = node.attributes.language as string;
          const pre = hastToTag(hast.children[0]) as Tag;
          pre.attributes['data-language'] = lang;
          // The block's head names the language (hub decision 2026-11-08-christianjmills-code-blocks);
          // the site's script adds the copy button to it.
          const name = LANGUAGE_NAMES[lang];
          return new Tag('div', { class: 'code-block' }, [
            new Tag('div', { class: 'code-head' }, name ? [new Tag('span', { class: 'code-language' }, [name])] : []),
            pre,
          ]);
        },
      },
    },
  };

  return { config, refs, headings };
}

function hastToTag(node: any): RenderableTreeNode {
  if (node.type === 'text') return node.value;
  if (node.type !== 'element') return null;
  const attrs: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node.properties ?? {})) {
    attrs[k === 'className' ? 'class' : k] = Array.isArray(v) ? v.join(' ') : v;
  }
  if (node.tagName === 'pre') attrs.tabindex = 0; // scrollable regions must be keyboard reachable
  return new Tag(node.tagName, attrs, node.children.map(hastToTag));
}

/** The blocks whose own inline content gathers its references into one marker. */
const MARKED_BLOCKS = new Set(['p', 'li', 'td', 'th']);

function isRef(node: RenderableTreeNode): node is Tag {
  return Tag.isTag(node) && node.name === 'a' && node.attributes.class === 'ref';
}

/** Removes the references from a block's inline content, collecting them in order. Nested blocks keep their own. */
function stripRefs(children: RenderableTreeNode[], found: Tag[]): RenderableTreeNode[] {
  const out: RenderableTreeNode[] = [];
  for (const child of children) {
    if (isRef(child)) {
      found.push(child);
      // "repeating {% ref %}." reads "repeating." once the reference has gone.
      const prev = out.at(-1);
      if (typeof prev === 'string') out[out.length - 1] = prev.replace(/\s+$/, '');
      continue;
    }
    if (Tag.isTag(child) && !MARKED_BLOCKS.has(child.name) && !['ul', 'ol'].includes(child.name)) {
      out.push(new Tag(child.name, child.attributes, stripRefs(child.children, found)));
    } else out.push(child);
  }
  // A space left before punctuation or at the end of the block, by a removed reference.
  for (let i = 0; i < out.length; i++) {
    const s = out[i];
    if (typeof s !== 'string') continue;
    const next = out[i + 1];
    if (next === undefined && found.length) out[i] = s.replace(/\s+$/, '');
    else if (typeof next === 'string' && /^[.,;:)]/.test(next)) out[i] = s.replace(/\s+$/, '');
  }
  return out.filter((n) => n !== '');
}

/** One marker for a block's references: each source's label once, linking to its entry in the
 *  page's Sources, then each of its locators, linking to the place in the source. Sources are
 *  separated by semicolons, and each label ends in a colon. */
function marker(refs: Tag[]): Tag {
  const groups = new Map<string, { label: string; items: { locator: string; href: string }[] }>();
  for (const r of refs) {
    const id = String(r.attributes['data-source']);
    const g = groups.get(id) ?? { label: String(r.attributes['data-label']), items: [] };
    const item = { locator: String(r.attributes['data-locator'] ?? ''), href: String(r.attributes.href) };
    if (!g.items.some((i) => i.href === item.href && i.locator === item.locator)) g.items.push(item);
    groups.set(id, g);
  }
  const parts: RenderableTreeNode[] = ['['];
  [...groups].forEach(([id, g], gi) => {
    if (gi > 0) parts.push('; ');
    const located = g.items.filter((i) => i.locator);
    if (id === 'this') parts.push(g.label);
    else if (!located.length) parts.push(new Tag('a', { class: 'ref-source', href: g.items[0].href }, [g.label]));
    else parts.push(new Tag('a', { class: 'ref-source', href: `#source-${id}` }, [g.label]));
    // "the talk's code: c03-cellular/src/main.rs, lines 17–34; lines 40–70": a locator that shares
    // the one before it's file or part drops it.
    let prev = '';
    located.forEach((i, ii) => {
      parts.push(ii === 0 ? ': ' : '; ');
      const head = /^(.*), /.exec(i.locator)?.[1];
      const text = head && prev.startsWith(head + ', ') ? i.locator.slice(head.length + 2) : i.locator;
      parts.push(new Tag('a', { class: 'ref', href: i.href }, [text]));
      prev = i.locator;
    });
  });
  parts.push(']');
  return new Tag('span', { class: 'refs' }, parts);
}

/** The reader's view of references (hub decision 2026-11-04-traced-deliverables-reader-view): a
 *  block's citations merge into one marker at its end. Returns a new tree; the input is unchanged. */
export function mergeReferences(node: RenderableTreeNode): RenderableTreeNode {
  if (Array.isArray(node)) return node.map(mergeReferences);
  if (!Tag.isTag(node)) return node;
  const children = node.children.map(mergeReferences);
  if (!MARKED_BLOCKS.has(node.name)) return new Tag(node.name, node.attributes, children);
  const found: Tag[] = [];
  const kept = stripRefs(children, found);
  if (!found.length) return new Tag(node.name, node.attributes, children);
  // In a list item, the marker goes after the item's own text, before any nested list.
  const nested = kept.findIndex((c) => Tag.isTag(c) && ['ul', 'ol'].includes(c.name));
  const at = nested === -1 ? kept.length : nested;
  return new Tag(node.name, node.attributes, [...kept.slice(0, at), ' ', marker(found), ...kept.slice(at)]);
}

export const LANGUAGES = ['javascript', 'rust', 'python', 'shell', 'json', 'text'];

/** Each language's name as its readers know it, for the code block's head. Plain text has none. */
const LANGUAGE_NAMES: Record<string, string> = {
  javascript: 'JavaScript',
  rust: 'Rust',
  python: 'Python',
  shell: 'Shell',
  json: 'JSON',
};

/** Parses, validates and renders one Markdoc document; throws with every error if any. */
export function renderDocument(source: string, file: string, ctx: BundleContext) {
  const ast = Markdoc.parse(source, { file });
  const { config, refs, headings } = dialect(ctx);
  const problems = Markdoc.validate(ast, config).filter((e) => !['debug', 'info'].includes(e.error.level));
  if (problems.length) {
    const lines = problems.map((p) => `  ${file}:${(p.lines[0] ?? 0) + 1}: ${p.error.message} [${p.error.id}]`);
    throw new Error(`Markdoc validation failed for ${file}:\n${lines.join('\n')}`);
  }
  const tree = Markdoc.transform(ast, config);
  return { html: Markdoc.renderers.html(mergeReferences(tree)), tree, refs, headings };
}
