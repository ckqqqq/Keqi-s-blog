import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';

// Check the actual Pages artifact, not just the Markdown image syntax.
const output = resolve(process.argv[2] || 'public');
const pages = { 'frontier-model-trends': 33, 'harness-rsi': 18 };
let total = 0;
let animations = 0;
for (const [slug, expected] of Object.entries(pages)) {
  const html = readFileSync(join(output, 'session', slug, 'index.html'), 'utf8');
  const images = [...html.matchAll(/<img\b[^>]*\bsrc=(?:"([^"]+)"|'([^']+)'|([^\s>]+))/g)];
  assert.equal(images.length, expected, `${slug}: image count changed`);
  assert.equal((html.match(/article-image-link/g) || []).length, expected, `${slug}: missing original-image links`);
  for (const match of images) {
    const source = match[1] || match[2] || match[3];
    const relative = source.slice(source.indexOf('media/session/'));
    assert.ok(relative.startsWith('media/session/'), `Unexpected image path: ${source}`);
    const bytes = readFileSync(join(output, relative));
    const digest = createHash('sha256').update(bytes).digest('hex').slice(0, 16);
    assert.ok(relative.includes(digest), `Image bytes changed: ${relative}`);
    if (relative.endsWith('.gif')) {
      assert.match(bytes.subarray(0, 6).toString(), /^GIF8[79]a$/);
      assert.ok(bytes.includes(Buffer.from('NETSCAPE2.0')), `GIF has no loop extension: ${relative}`);
      animations++;
    } else {
      // Some exported .png files contain JPEG bytes; browsers sniff both.
      assert.ok(bytes.subarray(1, 4).toString() === 'PNG' || bytes.subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff])), `Invalid raster: ${relative}`);
    }
    total++;
  }
}
console.log(`Verified ${total} images (${animations} looping GIFs), original bytes and full-size links.`);
