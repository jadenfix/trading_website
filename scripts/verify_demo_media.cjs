const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync, spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const manifestPath = path.join(root, 'docs/demo/capture.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const git = args => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');

assert.equal(manifest.schema_version, 1);
assert.equal(manifest.scope.mode, 'provider-free-fictional-fixture');
assert.equal(git(['rev-parse', `${manifest.captured_source_commit}^{tree}`]), manifest.captured_source_tree);
assert.equal(spawnSync('git', ['merge-base', '--is-ancestor', manifest.captured_source_commit, 'HEAD'], { cwd: root }).status, 0, 'captured source must be an ancestor of HEAD');

for (const input of manifest.source_inputs) {
  const committed = execFileSync('git', ['show', `${manifest.captured_source_commit}:${input.path}`], { cwd: root });
  assert.equal(sha256(committed), input.sha256, `${input.path}: captured SHA-256 drift`);
  assert.equal(git(['rev-parse', `${manifest.captured_source_commit}:${input.path}`]), input.git_blob, `${input.path}: captured blob drift`);
  assert.equal(sha256(fs.readFileSync(path.join(root, input.path))), input.sha256, `${input.path}: current runtime input differs from capture`);
}

for (const media of manifest.media) {
  const file = path.join(root, media.path);
  assert.equal(fs.statSync(file).size, media.bytes, `${media.path}: byte count mismatch`);
  assert.equal(sha256(fs.readFileSync(file)), media.sha256, `${media.path}: SHA-256 mismatch`);
  assert.equal(git(['hash-object', file]), media.git_blob, `${media.path}: Git blob mismatch`);
  execFileSync('ffmpeg', ['-v', 'error', '-i', file, '-f', 'null', '-'], { stdio: 'pipe' });
  const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,nb_frames:format=duration', '-of', 'json', file], { encoding: 'utf8' }));
  assert.deepEqual(probe, media.probe, `${media.path}: decoded stream metadata mismatch`);
  const stream = probe.streams[0];
  assert.ok(Number(stream.width) >= 800 && Number(stream.height) >= 450, `${media.path}: resolution too small`);
  if (media.path.endsWith('.gif') || media.path.endsWith('.mp4')) assert.ok(Number(stream.nb_frames) >= 40, `${media.path}: too few frames`);
}

console.log(`PASS: ${manifest.source_inputs.length} source inputs and ${manifest.media.length} fully decoded media files match the captured source.`);
