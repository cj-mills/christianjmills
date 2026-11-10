// The fixture's failing interactive: its mount rejects, so the site keeps its fallback.
export async function mount() {
  throw new Error('the fixture fails on purpose');
}
