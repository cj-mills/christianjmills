import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { manifest } from '../spec/manifest.ts';
import { deliverablesLoader } from './loaders/deliverables.ts';

const deliverables = defineCollection({
  loader: deliverablesLoader(),
  schema: manifest.extend({
    pinned: z.object({ sha256: z.string(), key: z.string() }).nullable(),
    refs: z.array(z.object({ source: z.string(), locator: z.string(), href: z.string() })),
    notes: z.array(
      z.object({
        n: z.number(),
        sources: z.array(z.object({ id: z.string(), label: z.string(), items: z.array(z.object({ locator: z.string(), href: z.string() })) })),
      }),
    ),
    markdown: z.string(),
  }),
});

export const collections = { deliverables };
