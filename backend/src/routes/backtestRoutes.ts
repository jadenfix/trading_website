// backend/src/routes/backtestRoutes.ts
import { Router, RequestHandler } from 'express';
import { execFile }               from 'child_process';
import { promisify }              from 'util';
import path                       from 'path';
import fs                         from 'fs';

const router        = Router();
const execFileAsync = promisify(execFile);

const repoRoot = path.resolve(__dirname, '../../..');
const BIN_PATH = path.join(repoRoot, 'cpp_engine','build','trading_system');

// make sure we fail fast if the binary isn’t there
if (!fs.existsSync(BIN_PATH)) {
  throw new Error(`C++ binary not found at ${BIN_PATH}`);
}

const runBacktest: RequestHandler = async (req, res) => {
  const dataset  = String(req.query.dataset  || '');
  const strategy = String(req.query.strategy || '');
  const cash     = String(req.query.cash     || '');

  const args: string[] = [];
  if (dataset)  args.push(dataset);
  if (strategy) args.push(strategy);
  if (cash)     args.push(cash);

  try {
    const { stdout } = await execFileAsync(
      BIN_PATH,
      args,
      {
        cwd: path.dirname(BIN_PATH),       // so "../data" -> <repo>/cpp_engine/data
        maxBuffer: 10*1024*1024
      }
    );
    res.json(JSON.parse(stdout));          // just call .json, don’t return it
  } catch (err: any) {
    console.error('backtest failed', err);
    res.status(500).json({
      error: err.message,
      rawOutput: err.stdout?.slice?.(0,200) ?? ''
    });
  }
};

router.get('/run',     runBacktest);
router.get('/summary', (_req,res) => {
  res.json({ message: 'Not yet implemented' });
});

export default router;