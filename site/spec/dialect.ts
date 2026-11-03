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
          return new Tag('a', { class: 'ref', href, 'data-source': attrs.source }, [
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
            themes: { light: 'github-light-high-contrast', dark: 'github-dark-high-contrast' },
            defaultColor: false,
          });
          const pre = hastToTag(hast.children[0]) as Tag;
          pre.attributes['data-language'] = node.attributes.language;
          return pre;
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

export const LANGUAGES = ['javascript', 'rust', 'python', 'shell', 'json', 'text'];

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
  return { html: Markdoc.renderers.html(tree), tree, refs, headings };
}
