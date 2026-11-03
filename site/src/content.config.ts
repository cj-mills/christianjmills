import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { manifest } from '../spec/manifest.ts';
import { deliverablesLoader } from './loaders/deliverables.ts';

const deliverables = defineCollection({
  loader: deliverablesLoader(),
  schema: manifest.extend({
    pinned: z.object({ sha256: z.string(), key: z.string() }).nullable(),
    refs: z.array(z.object({ source: z.string(), locator: z.string(), href: z.string() })),
    markdown: z.string(),
  }),
});

export const collections = { deliverables };
