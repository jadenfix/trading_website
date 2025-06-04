import { Router } from 'express'
import { StrategyResult } from '../models/portfolio'

const router = Router()

router.get('/run', async (req, res, next) => {
  try {
    const { dataset, strategy, cash } = req.query as {
      dataset?: string
      strategy?: string
      cash?: string
    }

    if (!dataset || !strategy || !cash) {
      res.status(400).json({ error: 'Missing dataset, strategy or cash' })
      return
    }

    const initialCash = Number(cash)

    // In a real implementation this would run the backtest engine.
    // For now we return a mocked result using the provided parameters.
    const result: StrategyResult = {
      total_return_pct: 12.34,
      max_drawdown_pct: 3.21,
      realized_pnl: initialCash * 0.1234,
      total_commission: initialCash * 0.001,
      num_fills: 42,
      final_equity: initialCash + initialCash * 0.1234 - initialCash * 0.001,
    }

    res.json([
      {
        strategy_on_dataset: `${strategy}@${dataset}`,
        return_pct: result.total_return_pct,
        max_dd_pct: result.max_drawdown_pct,
        realized_pnl: result.realized_pnl,
        commission: result.total_commission,
        fills: result.num_fills,
        final_equity: result.final_equity,
      },
    ])
  } catch (err) {
    next(err)
  }
})

export default router
