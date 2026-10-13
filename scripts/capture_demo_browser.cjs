const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || path.resolve(__dirname, '../frontend/node_modules/playwright-core'));
const { root, startDemoServers } = require('./browser_harness.cjs');

const outputIndex = process.argv.indexOf('--output');
const outputDir = path.resolve(root, outputIndex >= 0 ? process.argv[outputIndex + 1] : 'artifacts/demo');
if (!outputDir.startsWith(root + path.sep)) throw new Error('Demo output must stay inside the repository');
fs.mkdirSync(outputDir, { recursive: true });
if (fs.readdirSync(outputDir).length > 0) throw new Error(`Refusing to overwrite non-empty output directory: ${outputDir}`);

const sha256 = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const git = args => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const mediaName = name => path.join(outputDir, name);

function inspect(file) {
  return JSON.parse(execFileSync('ffprobe', [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=codec_name,width,height,nb_frames:format=duration',
    '-of', 'json', file
  ], { encoding: 'utf8' }));
}

(async () => {
  let servers;
  let browser;
  let context;
  const videoDir = fs.mkdtempSync(path.join(os.tmpdir(), 'trading-website-video-'));
  try {
    servers = await startDemoServers();
    browser = await chromium.launch({ executablePath: process.env.CHROME_BIN || undefined, headless: true, args: ['--no-sandbox'] });
    context = await browser.newContext({
      viewport: { width: 1280, height: 720 },
      recordVideo: { dir: videoDir, size: { width: 1280, height: 720 } }
    });
    const page = await context.newPage();
    const failures = [];
    page.on('pageerror', error => failures.push(error.message));
    await page.goto('http://127.0.0.1:5188/backtest', { waitUntil: 'networkidle' });
    await page.locator('.fixture-disclosure').waitFor();
    await page.waitForTimeout(1600);
    await page.locator('.backtest-form select').first().selectOption('btc_2024');
    await page.waitForTimeout(900);
    await page.getByRole('button', { name: 'Run Backtest', exact: false }).hover();
    await page.waitForTimeout(700);
    await page.getByRole('button', { name: 'Run Backtest', exact: false }).click();
    await page.locator('.results-container').waitFor();
    const receiptText = await page.locator('.results-container').innerText();
    assert.match(receiptText, /FICTIONAL-FIXTURE/i);
    assert.match(receiptText, /CHECKED-IN FICTIONAL OHLCV FIXTURE/i);
    await page.locator('.results-container').scrollIntoViewIfNeeded();
    await page.waitForTimeout(3200);
    await page.screenshot({ path: mediaName('trading-website-demo-poster.png') });
    await page.locator('.backtest-form select').first().selectOption('eth_2024');
    await page.waitForTimeout(900);
    await page.getByRole('button', { name: 'Run Backtest', exact: false }).click();
    await page.locator('.results-container').waitFor();
    await page.waitForTimeout(2400);
    assert.deepEqual(failures, []);

    const apiResponse = await fetch('http://127.0.0.1:8000/api/backtest/run?dataset=btc_2024&strategy=FIXTURE_BUY_HOLD&cash=100000');
    assert.equal(apiResponse.status, 200);
    const apiReceipt = await apiResponse.json();
    await page.close();
    await context.close();
    context = undefined;

    const rawVideo = fs.readdirSync(videoDir).filter(name => name.endsWith('.webm')).map(name => path.join(videoDir, name))[0];
    if (!rawVideo) throw new Error('Playwright did not produce a browser recording');
    execFileSync('ffmpeg', ['-y', '-v', 'error', '-i', rawVideo, '-vf', 'scale=1280:-2:flags=lanczos', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mediaName('trading-website-demo.mp4')]);
    execFileSync('ffmpeg', ['-y', '-v', 'error', '-i', rawVideo, '-filter_complex', '[0:v]fps=6,scale=800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=160:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle', '-loop', '0', mediaName('trading-website-demo.gif')]);

    const sourceCommit = git(['rev-parse', 'HEAD']);
    const sourceTree = git(['rev-parse', 'HEAD^{tree}']);
    const sourcePaths = git(['ls-files']).split('\n').filter(file =>
      file === 'backend/package.json' || file === 'backend/package-lock.json' ||
      file === 'frontend/package.json' || file === 'frontend/package-lock.json' ||
      file === 'frontend/vite.config.ts' || file.startsWith('frontend/src/') ||
      file === 'backend/src/app.ts' || file === 'backend/src/index.ts' ||
      file === 'backend/src/routes/backtestRoutes.ts' ||
      file === 'scripts/browser_harness.cjs' || file === 'scripts/capture_demo_browser.cjs'
    );
    const sourceInputs = sourcePaths.map(file => ({
      path: file,
      git_blob: git(['rev-parse', `${sourceCommit}:${file}`]),
      sha256: crypto.createHash('sha256').update(execFileSync('git', ['show', `${sourceCommit}:${file}`], { cwd: root })).digest('hex')
    }));
    const media = ['trading-website-demo.gif', 'trading-website-demo.mp4', 'trading-website-demo-poster.png'].map(name => ({
      path: `docs/demo/${name}`,
      bytes: fs.statSync(mediaName(name)).size,
      sha256: sha256(mediaName(name)),
      git_blob: git(['hash-object', mediaName(name)]),
      probe: inspect(mediaName(name))
    }));
    const capture = {
      schema_version: 1,
      captured_source_commit: sourceCommit,
      captured_source_tree: sourceTree,
      scope: {
        mode: 'provider-free-fictional-fixture',
        claim: 'Actual production-build browser replay of checked-in fictional OHLCV rows.',
        exclusions: ['live market data', 'investment advice', 'native strategy validation', 'trading performance']
      },
      browser: { engine: 'Chromium', viewport: '1280x720' },
      receipt: apiReceipt[0],
      source_inputs: sourceInputs,
      media
    };
    fs.writeFileSync(mediaName('capture.json'), JSON.stringify(capture, null, 2) + '\n');
    console.log(JSON.stringify({ output: outputDir, sourceCommit, sourceTree, media }, null, 2));
  } finally {
    if (context) await context.close();
    if (browser) await browser.close();
    if (servers) await servers.stop();
  }
})().catch(error => { console.error(error); process.exit(1); });
