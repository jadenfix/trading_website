const assert = require('node:assert/strict');
const path = require('node:path');
const { startDemoServers } = require('./browser_harness.cjs');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || path.resolve(__dirname, '../frontend/node_modules/playwright-core'));

(async () => {
  let servers;
  let browser;
  try {
    servers = await startDemoServers();
    browser = await chromium.launch({ executablePath: process.env.CHROME_BIN || undefined, headless: true, args: ['--no-sandbox'] });
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
      assert.equal(await page.locator('.backtest-form select').nth(1).inputValue(), 'FIXTURE_BUY_HOLD');
      await page.getByRole('button', { name: 'Run Backtest', exact: false }).click();
      await page.locator('.results-container').waitFor();
      const receipt = await page.locator('.results-container').innerText();
      assert.match(receipt, /FICTIONAL-FIXTURE/i);
      assert.match(receipt, /CHECKED-IN FICTIONAL OHLCV FIXTURE/i);
      assert.deepEqual(failures, []);
      const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, content: document.documentElement.scrollWidth }));
      assert.ok(dimensions.content <= dimensions.width + 1, JSON.stringify(dimensions));
      await page.close();
      console.log('PASS: production browser navigation and provider-free fixture replay, viewport ' + width);
    }
  } finally {
    if (browser) await browser.close();
    if (servers) await servers.stop();
  }
})().catch(error => { console.error(error); process.exit(1); });
