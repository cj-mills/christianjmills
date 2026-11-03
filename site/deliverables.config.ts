// Where the build finds deliverable bundles. Each bucket maps to a base URL that bundle keys
// (sha256/<hash>.tar.gz) resolve against.
//
// PROTOTYPE: no R2 bucket exists yet. Until the user creates one, a local folder laid out exactly
// like the bucket stands in for it, so its contents can be uploaded to R2 unchanged. Set
// DELIVERABLES_PUBLIC_URL to the bucket's public URL (or another stand-in) to override.
export const buckets: Record<'public' | 'private', string | undefined> = {
  public: process.env.DELIVERABLES_PUBLIC_URL ?? 'file:///mnt/projects/scratch/christianjmills-r2-standin/public/',
  // Paid bundles never enter a public build; a private bucket is read only by a gated build.
  private: process.env.DELIVERABLES_PRIVATE_URL,
};

export const lockfilePath = new URL('./deliverables.lock.json', import.meta.url);

/** Preview overrides: point a deliverable at a local working tree. Honoured only when DELIVERABLES_ALLOW_LOCAL=1. */
export const localOverridesPath = new URL('./deliverables.local.json', import.meta.url);

/** Where a deliverable's files are served from on the site. */
export const assetBase = (id: string, version: string) => `/_deliverables/${id}/${version}/`;
