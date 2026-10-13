const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || path.resolve(__dirname, '../frontend/node_modules/playwright-core'));
(async () => {
  const server = spawn(process.execPath, [path.resolve(__dirname, '../frontend/node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '5188'], { cwd: path.resolve(__dirname, '../frontend'), stdio: ['ignore', 'pipe', 'pipe'] });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { server.kill(); reject(new Error('Vite startup timed out')); }, 15000);
    server.stdout.on('data', chunk => { if (chunk.toString().includes('Local:')) { clearTimeout(timer); resolve(); } });
    server.on('exit', code => { clearTimeout(timer); reject(new Error('Vite exited: ' + code)); });
  });
  const browser = await chromium.launch({ executablePath: process.env.CHROME_BIN, headless: true, args: ['--no-sandbox'] });
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      const failures = [];
      page.on('pageerror', error => failures.push(error.message));
      for (const route of ['/', '/research', '/backtest', '/csv-backtest', '/collaborate', '/codegen']) {
        await page.goto('http://127.0.0.1:5188' + route);
        await page.locator('h1').waitFor();
        assert.ok((await page.locator('h1').innerText()).trim());
        const size = await page.evaluate(() => ({ width: document.documentElement.clientWidth, content: document.documentElement.scrollWidth }));
        assert.ok(size.content <= size.width + 1, route + ': ' + JSON.stringify(size));
      }
      await page.getByRole('link', { name: 'Backtesting', exact: true }).click();
      assert.ok(page.url().endsWith('/backtest'));
      await page.locator('.backtest-form select').first().selectOption('btc_2024');
      assert.equal(await page.locator('.backtest-form select').nth(1).inputValue(), 'MAC_5_20');
      await page.getByRole('button', { name: 'Run Backtest', exact: false }).click();
      await page.locator('.error-message').first().waitFor();
      assert.ok((await page.locator('.error-message').first().innerText()).trim());
      assert.deepEqual(failures, []);
      const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, content: document.documentElement.scrollWidth }));
      if (dimensions.content > dimensions.width + 1) console.log(await page.evaluate(() => Array.from(document.querySelectorAll('*')).map(e => ({ tag:e.tagName, cls:e.className, right:e.getBoundingClientRect().right, width:e.getBoundingClientRect().width })).filter(e => e.right > innerWidth + 1 && e.width > 200).slice(0, 15)));
      assert.ok(dimensions.content <= dimensions.width + 1, JSON.stringify(dimensions));
      await page.close();
      console.log('PASS: actual browser navigation and unavailable-backend error, viewport ' + width);
    }
  } finally { await browser.close(); server.kill(); }
})().catch(error => { console.error(error); process.exit(1); });
