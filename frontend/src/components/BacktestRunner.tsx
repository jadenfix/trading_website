// src/components/BacktestRunner.tsx
import React, { useState } from 'react';
import { runBacktest, StrategyResult } from '../api/backtest';

export function BacktestRunner() {
  const [results, setResults] = useState<StrategyResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string|undefined>();

  const handleRun = async () => {
    setLoading(true);
    setError(undefined);
    try {
      const data = await runBacktest('stocks_april', 'MAC_5_20', 100000);
      setResults(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={handleRun} disabled={loading}>
        {loading ? 'Running…' : 'Run Backtest'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {results.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Strategy@Dataset</th>
              <th>Return&nbsp;(%)</th>
              <th>Max DD&nbsp;(%)</th>
              <th>PnL</th>
              <th>Comm</th>
              <th>Fills</th>
              <th>Equity</th>
            </tr>
          </thead>
          <tbody>
            {results.map(r => (
              <tr key={r.strategy_on_dataset}>
                <td>{r.strategy_on_dataset}</td>
                <td>{r.return_pct.toFixed(2)}</td>
                <td>{r.max_dd_pct.toFixed(2)}</td>
                <td>{r.realized_pnl.toFixed(2)}</td>
                <td>{r.commission.toFixed(2)}</td>
                <td>{r.fills}</td>
                <td>{r.final_equity.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}