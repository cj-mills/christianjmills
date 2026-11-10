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

/** One numbered note: what a block cites, each source with its passages. Listed on the page's
 *  references page; the marker in the text is its number. */
export interface Note {
  n: number;
  sources: { id: string; label: string; items: { locator: string; href: string }[] }[];
}

export interface Heading {
  depth: number;
  slug: string;
  text: string;
  /** A card's title: the contents rail lists a section's cards under it. */
  card?: boolean;
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

/** "line 43" or "lines 40–54". */
const lines = (a: number, b: number) => (a === b ? `line ${a}` : `lines ${a}–${b}`);

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
  /** Each trait's scale, as its first mark gives it: one scale for a trait across the page. */
  const traitScales = new Map<string, string>();
  /** The ids of the cards being transformed, innermost last: a part's id starts with its card's. */
  const cards: string[] = [];

  /** An id no other heading on the page has: the base, or the base with a number. */
  function claim(base: string): string {
    let id = base;
    for (let n = 2; usedIds.has(id); n++) id = `${base}-${n}`;
    usedIds.add(id);
    return id;
  }

  function checkAsset(src: unknown): ValidationError[] {
    if (typeof src !== 'string' || /^[a-z]+:/i.test(src) || src.startsWith('/')) {
      return [error('asset-external', `images come from the bundle; "${src}" is not a bundle path`)];
    }
    const bytes = files.get(src);
    if (!bytes) return [error('asset-missing', `"${src}" is not in the bundle`)];
    if (!pngSize(bytes)) return [error('asset-format', `"${src}": format 0.1 accepts PNG images only`)];
    return [];
  }

  /** A figure's dark-theme image is a PNG in the bundle, the size of the light one. */
  function checkDark(src: unknown, dark: unknown): ValidationError[] {
    if (dark === undefined) return [];
    const errs = checkAsset(dark);
    if (errs.length || typeof src !== 'string') return errs;
    const [a, b] = [files.get(src), files.get(dark as string)].map((f) => (f ? pngSize(f) : null));
    if (a && b && (a.width !== b.width || a.height !== b.height))
      return [error('asset-dark-size', `"${dark}" is ${b.width} by ${b.height}; the image it stands in for, "${src}", is ${a.width} by ${a.height}`)];
    return [];
  }

  /** An image, or a light and dark pair: the dark one where the reader's system asks for dark. */
  function picture(src: string, dark: string | undefined, alt: string): Tag {
    const img = image(src, alt);
    if (!dark) return img;
    return new Tag('picture', {}, [new Tag('source', { media: '(prefers-color-scheme: dark)', srcset: assetBase + dark }), img]);
  }

  function image(src: string, alt: string, title?: string): Tag {
    const size = pngSize(files.get(src)!)!;
    return new Tag('img', {
      src: assetBase + src,
      alt,
      ...(title ? { title } : {}),
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
      return { locator: `${attrs.path}, ${lines(a, b)}`, href: assetBase + attrs.path };
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
          locator: `${attrs.path}, ${lines(a, b)}`,
          href: `${repo}/blob/${s.commit}/${attrs.path}#L${a}${a === b ? '' : `-L${b}`}`,
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
          // One reference per claim in the tree, which the Markdown copy keeps; the HTML gives a
          // block one numbered marker at its end for all of them (mergeReferences, below).
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
          // What this mount is given: any JSON object. One module may be mounted many times on a
          // page, each with its own parameters.
          params: { type: Object },
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
              ...(node.attributes.params ? { 'data-params': JSON.stringify(node.attributes.params) } : {}),
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
          // The same image drawn for the dark theme. Without it, one image serves both.
          dark: { type: String },
          alt: { type: String, required: true },
          caption: { type: String },
        },
        validate(node) {
          const errs = [...checkAsset(node.attributes.src), ...checkDark(node.attributes.src, node.attributes.dark)];
          if (!String(node.attributes.alt).trim()) errs.push(error('figure-alt', 'a figure needs alt text'));
          return errs;
        },
        transform(node) {
          const { src, dark, alt, caption } = node.attributes;
          return new Tag('figure', {}, [picture(src, dark, alt), ...(caption ? [new Tag('figcaption', {}, [caption])] : [])]);
        },
      },

      // A card: one entry of a catalog, under a title, in labeled parts. What comes before its
      // first part is its opening line. The contents rail lists a section's cards under it.
      card: {
        attributes: {
          title: { type: String, required: true },
          id: { type: String },
          kind: { type: String },
        },
        validate(node) {
          const id = node.attributes.id;
          return id !== undefined && !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)
            ? [error('card-id', `{% card id="${id}" %}: an id is lowercase words joined by hyphens`)]
            : [];
        },
        transform(node, config) {
          const { title, kind } = node.attributes;
          if (node.attributes.id && usedIds.has(node.attributes.id)) throw new Error(`{% card id="${node.attributes.id}" %}: the page already has that id`);
          const id = claim(node.attributes.id ?? slugify(title));
          headings.push({ depth: 3, slug: id, text: title, card: true });
          cards.push(id);
          const children = node.transformChildren(config);
          cards.pop();
          const first = children.findIndex((c) => Tag.isTag(c) && c.attributes.class === 'part');
          const opening = first === -1 ? children : children.slice(0, first);
          return new Tag('section', { class: 'card' }, [
            new Tag('header', { class: 'card-head' }, [
              ...(kind ? [new Tag('p', { class: 'card-kind' }, [kind])] : []),
              new Tag('h3', { id }, [title]),
              ...opening,
            ]),
            ...(first === -1 ? [] : children.slice(first)),
          ]);
        },
      },

      // One labeled part of a card.
      part: {
        attributes: { title: { type: String, required: true } },
        transform(node, config) {
          const title = node.attributes.title as string;
          const id = claim(`${cards.at(-1)}-${slugify(title)}`);
          headings.push({ depth: 4, slug: id, text: title });
          return new Tag('section', { class: 'part' }, [new Tag('h4', { id }, [title]), ...node.transformChildren(config)]);
        },
      },

      // A step list: each step a figure with its text. Inside {% interactive %} it is the fallback,
      // and the module reads each step's text from it by `key` (spec/README.md, Interactives).
      steps: {
        transform(node, config) {
          const items = node.transformChildren(config);
          // A step that doesn't say where it is in a longer run is numbered by its place in the list.
          items.forEach((li, i) => {
            if (!Tag.isTag(li) || li.attributes['data-at'] !== undefined) return;
            const at = li.children.findIndex((c) => Tag.isTag(c) && c.name !== 'picture' && c.name !== 'img');
            li.children.splice(at === -1 ? li.children.length : at, 0, new Tag('p', { class: 'step-label' }, [`Step ${i + 1}`]));
          });
          return new Tag('ol', { class: 'steps' }, items);
        },
      },
      step: {
        attributes: {
          src: { type: String },
          dark: { type: String },
          alt: { type: String },
          // The kind of step, for the module that reads the list.
          key: { type: String },
          // Where the step is in a longer run: step `at` of `of`.
          at: { type: Number },
          of: { type: Number },
        },
        validate(node) {
          const { src, dark, alt, at, of } = node.attributes;
          const errs: ValidationError[] = [];
          if (src !== undefined) {
            errs.push(...checkAsset(src), ...checkDark(src, dark));
            if (!String(alt ?? '').trim()) errs.push(error('step-alt', "a step's figure needs alt text"));
          } else if (dark !== undefined) errs.push(error('step-dark', 'a step with a dark image needs the light one, in src'));
          if ((at === undefined) !== (of === undefined) || (at !== undefined && !(Number.isInteger(at) && Number.isInteger(of) && at >= 1 && at <= of)))
            errs.push(error('step-at', 'a step says where it is with both at and of, whole numbers with at from 1 to of'));
          if (!node.children.length) errs.push(error('step-text', 'a step needs its text'));
          return errs;
        },
        transform(node, config) {
          const { src, dark, alt, key, at, of } = node.attributes;
          return new Tag('li', { class: 'step', ...given({ 'data-key': key, 'data-at': at, 'data-of': of }) }, [
            ...(src ? [picture(src, dark, alt)] : []),
            ...(at !== undefined ? [new Tag('p', { class: 'step-label' }, [`Step ${at} of ${of}`])] : []),
            ...node.transformChildren(config),
          ]);
        },
      },
      // A value in a step's text that the interactive replaces as the reader moves through the steps.
      value: {
        inline: true,
        attributes: { name: { type: String } },
        transform(node, config) {
          return new Tag('span', { class: 'step-value', 'data-value': node.attributes.name ?? '' }, node.transformChildren(config));
        },
      },

      // A table: named, in a region that scrolls sideways on a narrow screen with its first column
      // fixed. A body row of one cell heads a group of rows. `sort` lets the reader order the
      // rows by a column of numbers; `key` explains the marks in its cells.
      table: {
        attributes: {
          label: { type: String, required: true },
          rowheads: { type: Boolean, default: false },
          sort: { type: Boolean, default: false },
          key: { type: Boolean, default: false },
        },
        transform(node, config) {
          const { label, rowheads, sort, key } = node.attributes;
          const table = node.transformChildren(config).find((c): c is Tag => Tag.isTag(c) && c.name === 'table');
          if (!table) throw new Error(`{% table label="${label}" %} holds no table`);
          return dataTable(table, { label, rowheads, sort, key });
        },
      },

      // The mark for a measured distribution: a track for the scale (from, to), a bar from the
      // 10th to the 90th percentile, a dot at the median. The number itself is the cell's text.
      mark: {
        selfClosing: true,
        attributes: {
          p10: { type: Number, required: true },
          median: { type: Number, required: true },
          p90: { type: Number, required: true },
          from: { type: Number, default: 0 },
          to: { type: Number, required: true },
          min: { type: Number },
          max: { type: Number },
          n: { type: Number },
          unit: { type: String },
          // What is measured and which run of it, so an interactive's own value can be marked on it.
          trait: { type: String },
          series: { type: String },
        },
        validate(node) {
          const a = node.attributes;
          const order = [a.from ?? 0, a.min ?? a.p10, a.p10, a.median, a.p90, a.max ?? a.p90, a.to];
          return order.every((v, i) => typeof v === 'number' && (i === 0 || order[i - 1] <= v)) && a.to > (a.from ?? 0)
            ? []
            : [error('mark-order', '{% mark %} needs from ≤ min ≤ p10 ≤ median ≤ p90 ≤ max ≤ to, with from below to')];
        },
        transform(node) {
          const a = { from: 0, ...node.attributes };
          if (a.trait) {
            const scale = `${a.from} to ${a.to}`;
            const first = traitScales.get(a.trait) ?? scale;
            if (first !== scale) throw new Error(`{% mark trait="${a.trait}" %}: the page's marks for one trait are on different scales (${first}; ${scale})`);
            traitScales.set(a.trait, scale);
          }
          const at = (v: number) => ((v - a.from) / (a.to - a.from)).toFixed(4).replace(/\.?0+$/, '') || '0';
          const unit = a.unit ? ` ${a.unit}` : '';
          const words = [
            `Median ${a.median}.`,
            `8 of 10${unit} fall between ${a.p10} and ${a.p90}.`,
            a.min !== undefined && a.max !== undefined ? `Lowest ${a.min}, highest ${a.max}${a.n ? `, over ${a.n}${unit}` : ''}.` : '',
          ].filter(Boolean).join(' ');
          return new Tag(
            'span',
            {
              class: 'trait-mark',
              role: 'img',
              'aria-label': words,
              title: words,
              style: `--lo:${at(a.p10)};--mid:${at(a.median)};--hi:${at(a.p90)}`,
              ...given({ 'data-trait': a.trait, 'data-series': a.series }),
              'data-from': a.from,
              'data-to': a.to,
              'data-p10': a.p10,
              'data-p90': a.p90,
            },
            [new Tag('i', {}, []), new Tag('b', {}, [])],
          );
        },
      },

      // Where an interactive's own value for a trait is shown, beside the measured ones.
      now: {
        selfClosing: true,
        attributes: { trait: { type: String, required: true }, digits: { type: Number, default: 0 } },
        transform(node) {
          return new Tag('span', { class: 'trait-now', 'data-trait': node.attributes.trait, 'data-digits': node.attributes.digits ?? 0 }, []);
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
      // Markdoc hands a {% table %}'s attributes to the table inside it, which the tag builds.
      table: {
        render: 'table',
        attributes: { label: { type: String }, rowheads: { type: Boolean }, sort: { type: Boolean }, key: { type: Boolean } },
      },

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
          // Plain text is Shiki's own, never among the loaded languages.
          if (lang !== 'text' && !ctx.highlighter.getLoadedLanguages().includes(lang))
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

/** The attributes that were given: the renderer prints one that is undefined as the word. */
function given(attributes: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(attributes).filter(([, v]) => v !== undefined));
}

function find(node: RenderableTreeNode, test: (t: Tag) => boolean, out: Tag[] = []): Tag[] {
  if (Tag.isTag(node)) {
    if (test(node)) out.push(node);
    for (const c of node.children) find(c, test, out);
  }
  return out;
}
const hasClass = (name: string) => (t: Tag) => t.attributes.class === name;

/** What the marks' parts mean, for a table that asks for it. The marks say it themselves to a screen reader. */
function markKey(now: boolean): Tag {
  const item = (cls: string, text: string) => new Tag('span', {}, [new Tag('span', { class: `key ${cls}` }, []), text]);
  return new Tag('p', { class: 'trait-key', 'aria-hidden': 'true' }, [
    item('key-bar', 'the middle 8 of 10'),
    item('key-dot', 'the median'),
    ...(now ? [item('key-now', 'the map on screen')] : []),
    item('key-track', 'the column’s whole scale'),
  ]);
}

/** Builds a {% table %}: its name, its scrolling region, its row groups and row heads, and each
 *  column's scale at its head when the column's marks share one. */
function dataTable(table: Tag, opts: { label: string; rowheads: boolean; sort: boolean; key: boolean }): Tag {
  const rowsOf = (section: Tag) => section.children.filter((c): c is Tag => Tag.isTag(c) && c.name === 'tr');
  const cellsOf = (row: Tag) => row.children.filter((c): c is Tag => Tag.isTag(c) && (c.name === 'td' || c.name === 'th'));
  const sections = table.children.filter((c): c is Tag => Tag.isTag(c));
  const thead = sections.find((c) => c.name === 'thead');
  const head = thead ? cellsOf(rowsOf(thead)[0]) : [];
  for (const th of head) th.attributes.scope = 'col';

  const bodies: Tag[] = [];
  const scales = new Map<number, string>();
  for (const row of sections.filter((c) => c.name === 'tbody').flatMap(rowsOf)) {
    const cells = cellsOf(row);
    if (cells.length === 1 && head.length > 1) {
      // A row of one cell heads the rows after it.
      bodies.push(new Tag('tbody', {}, [new Tag('tr', { class: 'group' }, [new Tag('th', { scope: 'rowgroup', colspan: head.length }, [new Tag('span', {}, cells[0].children)])])]));
      continue;
    }
    cells.forEach((cell, i) => {
      if (i === 0 && opts.rowheads) {
        cell.name = 'th';
        cell.attributes.scope = 'row';
      }
      for (const m of find(cell, hasClass('trait-mark'))) {
        // A column whose marks share a scale prints it at its head; one that mixes scales (a
        // table with a trait to a row) prints none.
        const scale = `${m.attributes['data-from']} to ${m.attributes['data-to']}`;
        scales.set(i, scales.has(i) && scales.get(i) !== scale ? '' : scale);
      }
    });
    if (!bodies.length) bodies.push(new Tag('tbody', {}, []));
    bodies.at(-1)!.children.push(row);
  }
  for (const [i, scale] of scales) if (scale) head[i]?.children.push(' ', new Tag('span', { class: 'scale' }, [scale]));

  const built = new Tag('table', opts.sort ? { 'data-sort': '' } : {}, [
    new Tag('caption', { class: 'visually-hidden' }, [opts.label]),
    ...(thead ? [thead] : []),
    ...bodies,
  ]);
  return new Tag('div', { class: 'table-block' }, [
    ...(opts.key ? [markKey(find(built, hasClass('trait-now')).length > 0)] : []),
    new Tag('div', { class: 'table-scroll', tabindex: 0, role: 'region', 'aria-label': opts.label }, [built]),
  ]);
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

/** The page's notes, numbered in the order its blocks first cite them. Two blocks that cite the
 *  same passages share a note. */
class Notes {
  list: Note[] = [];
  #byKey = new Map<string, Note>();

  note(refs: Tag[]): Note {
    const sources: Note['sources'] = [];
    for (const r of refs) {
      const id = String(r.attributes['data-source']);
      let source = sources.find((x) => x.id === id);
      if (!source) sources.push((source = { id, label: String(r.attributes['data-label']), items: [] }));
      const item = { locator: String(r.attributes['data-locator'] ?? ''), href: String(r.attributes.href) };
      if (!source.items.some((i) => i.href === item.href && i.locator === item.locator)) source.items.push(item);
    }
    const key = JSON.stringify(sources.map((x) => [x.id, x.items.map((i) => [i.href, i.locator])]));
    let note = this.#byKey.get(key);
    if (!note) {
      note = { n: this.list.length + 1, sources };
      this.#byKey.set(key, note);
      this.list.push(note);
    }
    return note;
  }
}

/** A block's marker: its note's number, linking to the note on the page's references page. The
 *  space before it doesn't break, so a marker never starts a line, and is inside the marker, so it
 *  goes when a reader hides the markers. `span.refs` is what an interactive's module and the
 *  reader's switch know a marker by (spec/README.md, References). */
function marker(note: Note, references: string): Tag {
  return new Tag('span', { class: 'refs' }, ['\u00a0', new Tag('a', { href: `${references}#ref-${note.n}`, role: 'doc-noteref' }, [`[${note.n}]`])]);
}

/** The reader's view of references (hub decision 2026-11-10-christianjmills-numbered-reference-markers):
 *  a block's citations become one numbered marker at its end, and `notes` collects what each number
 *  stands for. `references` is the address of the page that lists them. Returns a new tree; the
 *  input is unchanged. */
export function mergeReferences(node: RenderableTreeNode, references: string, notes = new Notes()): { tree: RenderableTreeNode; notes: Note[] } {
  const walk = (node: RenderableTreeNode): RenderableTreeNode => {
    if (Array.isArray(node)) return node.map(walk);
    if (!Tag.isTag(node)) return node;
    const children = node.children.map(walk);
    if (!MARKED_BLOCKS.has(node.name)) return new Tag(node.name, node.attributes, children);
    const found: Tag[] = [];
    const kept = stripRefs(children, found);
    if (!found.length) return new Tag(node.name, node.attributes, children);
    // In a list item, the marker goes after the item's own text, before any nested list.
    const nested = kept.findIndex((c) => Tag.isTag(c) && ['ul', 'ol'].includes(c.name));
    const at = nested === -1 ? kept.length : nested;
    return new Tag(node.name, node.attributes, [...kept.slice(0, at), marker(notes.note(found), references), ...kept.slice(at)]);
  };
  return { tree: walk(node), notes: notes.list };
}

/** The tags that go directly inside another. */
const NESTS: Record<string, string> = { part: 'card', step: 'steps' };

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
  // A part belongs to a card, and a step to a step list.
  const within = (node: Node, inside: string[]) => {
    const parent = NESTS[node.tag ?? ''];
    if (node.type === 'tag' && parent && inside.at(-1) !== parent)
      problems.push({ type: 'tag', lines: node.lines, error: error('nesting', `{% ${node.tag} %} goes directly inside {% ${parent} %}`) } as (typeof problems)[number]);
    for (const child of node.children) within(child, node.type === 'tag' ? [...inside, node.tag!] : inside);
  };
  within(ast, []);
  if (problems.length) {
    const lines = problems.map((p) => `  ${file}:${(p.lines[0] ?? 0) + 1}: ${p.error.message} [${p.error.id}]`);
    throw new Error(`Markdoc validation failed for ${file}:\n${lines.join('\n')}`);
  }
  const tree = Markdoc.transform(ast, config);
  const merged = mergeReferences(tree, `${ctx.manifest.urls.canonical}references/`);
  return { html: Markdoc.renderers.html(merged.tree), tree, refs, headings, notes: merged.notes };
}
