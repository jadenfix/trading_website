// src/api/backtest.ts
export interface StrategyResult {
    strategy_on_dataset: string;
    return_pct: number;
    max_dd_pct: number;
    realized_pnl: number;
    commission: number;
    fills: number;
    final_equity: number;
  }
  
  export async function runBacktest(
    dataset: string,
    strategy: string,
    cash: number
  ): Promise<StrategyResult[]> {
    const params = new URLSearchParams({ dataset, strategy, cash: String(cash) });
    const resp = await fetch(`/api/backtest/run?${params}`);
    if (!resp.ok) throw new Error(`Backtest failed: ${resp.statusText}`);
    return resp.json();
  }