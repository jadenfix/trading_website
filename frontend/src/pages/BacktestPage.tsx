// frontend/src/pages/BacktestPage.tsx
import { useState, FormEvent, useMemo } from 'react'
import { client } from '../api/client'

const STRATEGIES_BY_DATASET: Record<string,string[]> = {
  stocks_april: [
    'MAC_5_20','VWAP_2.0','ORB_30','Momentum_5_10_2_3',
    'Pairs_MSFT_NVDA','Pairs_NVDA_GOOG','Pairs_MSFT_GOOG',
    'LeadLag_MSFT->NVDA','LeadLag_NVDA->MSFT'
  ],
  '2024_only': [
    'MAC_5_20','VWAP_2.0','ORB_30','Momentum_5_10_2_3',
    'Pairs_BTC_ETH','Pairs_ETH_SOL','Pairs_BTC_SOL','Pairs_ETH_ADA','Pairs_SOL_ADA',
    'LeadLag_BTC->ETH','LeadLag_ETH->BTC','LeadLag_ETH->SOL','LeadLag_SOL->ETH'
  ],
  '2024_2025': [
    'MAC_5_20','VWAP_2.0','ORB_30','Momentum_5_10_2_3',
    'Pairs_BTC_ETH','Pairs_ETH_SOL','Pairs_BTC_SOL','Pairs_ETH_ADA','Pairs_SOL_ADA',
    'LeadLag_BTC->ETH','LeadLag_ETH->BTC','LeadLag_ETH->SOL','LeadLag_SOL->ETH'
  ],
}

export default function BacktestPage() {
  const [dataset, setDataset]   = useState('stocks_april')
  const [strategy, setStrategy] = useState(STRATEGIES_BY_DATASET['stocks_april'][0])
  const [cash, setCash]         = useState(100000)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string|null>(null)
  const [results, setResults]   = useState<any[]>([])

  // whenever dataset changes, reset strategy to first in that list
  useMemo(() => {
    setStrategy(STRATEGIES_BY_DATASET[dataset][0])
  }, [dataset])

  const handleRun = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await client.get<any[]>('/backtest/run', {
        params: { dataset, strategy, cash }
      })
      if (res.data.length === 0) {
        setError('No results returned. Check your parameters or data directory.')
      } else {
        setResults(res.data)
      }
    } catch (err: any) {
      setError(err.response?.data?.error || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>Run a Backtest</h2>
      <form onSubmit={handleRun}>
        <label>
          Dataset:{' '}
          <select value={dataset} onChange={e => setDataset(e.target.value)}>
            {Object.keys(STRATEGIES_BY_DATASET).map(ds => (
              <option key={ds} value={ds}>{ds}</option>
            ))}
          </select>
        </label>

        <label style={{ marginLeft: 10 }}>
          Strategy:{' '}
          <select value={strategy} onChange={e => setStrategy(e.target.value)}>
            {STRATEGIES_BY_DATASET[dataset].map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>

        <label style={{ marginLeft: 10 }}>
          Cash:{' '}
          <input
            type="number"
            value={cash}
            onChange={e => setCash(Number(e.target.value))}
          />
        </label>

        <button type="submit" disabled={loading} style={{ marginLeft: 10 }}>
          {loading ? 'Running…' : 'Run Backtest'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{`Error: ${error}`}</p>}

      {results.length > 0 && (
        <table border={1} cellPadding={5} style={{ marginTop: 20 }}>
          <thead>
            <tr>
              {Object.keys(results[0]).map(k => <th key={k}>{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {results.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((v, j) => <td key={j}>{String(v)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}