// backend/src/models/portfolio.ts

export interface StrategyResult {
    total_return_pct: number
    max_drawdown_pct: number
    realized_pnl: number
    total_commission: number
    num_fills: number
    final_equity: number
  }