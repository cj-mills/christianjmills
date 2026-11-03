// Where deliverable bundles live. Each bucket has an R2 name (for publishing) and a base URL the
// build fetches bundle keys (sha256/<hash>.tar.gz) from.
//
// The public bucket is served from its custom domain. Set DELIVERABLES_PUBLIC_URL to override it,
// e.g. with a file: URL to a local folder laid out like the bucket, for offline builds.
export interface Bucket {
  /** The R2 bucket name. */
  name: string;
  /** Base URL the build reads from: https: for R2, file: for a local stand-in. */
  url: string | undefined;
}

export const buckets: Record<'public' | 'private', Bucket> = {
  public: {
    name: 'christianjmills-deliverables-public',
    url: process.env.DELIVERABLES_PUBLIC_URL ?? 'https://deliverables.christianjmills.com/',
  },
  // Paid bundles never enter a public build; a private bucket is read only by a gated build.
  private: { name: 'christianjmills-deliverables-private', url: process.env.DELIVERABLES_PRIVATE_URL },
};

export const lockfilePath = new URL('./deliverables.lock.json', import.meta.url);

/** Preview overrides: point a deliverable at a local working tree. Honoured only when DELIVERABLES_ALLOW_LOCAL=1. */
export const localOverridesPath = new URL('./deliverables.local.json', import.meta.url);

/** Where a deliverable's files are served from on the site. */
export const assetBase = (id: string, version: string) => `/_deliverables/${id}/${version}/`;
