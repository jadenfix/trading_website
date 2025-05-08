// frontend/src/pages/BacktestPage.tsx
import { useState, FormEvent } from 'react'
import client from '../api/client'

const DATASETS = ['stocks_april','2024_only','2024_2025']
const STRATEGIES = [
  'MAC_5_20','VWAP_2.0','ORB_30','Momentum_5_10_2_3',
  'Pairs_MSFT_NVDA','Pairs_NVDA_GOOG','Pairs_MSFT_GOOG',
  'Pairs_BTC_ETH','Pairs_ETH_SOL','Pairs_BTC_SOL','Pairs_ETH_ADA','Pairs_SOL_ADA',
  'LeadLag_MSFT->NVDA','LeadLag_NVDA->MSFT',
  'LeadLag_BTC->ETH','LeadLag_ETH->BTC','LeadLag_ETH->SOL','LeadLag_SOL->ETH'
]

export default function BacktestPage() {
  const [dataset, setDataset]     = useState(DATASETS[0])
  const [strategy, setStrategy]   = useState(STRATEGIES[0])
  const [cash, setCash]           = useState(100000)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string|null>(null)
  const [rawErrorOutput, setRawErrorOutput] = useState<string|null>(null)
  const [results, setResults]     = useState<any[]>([])

  const handleRun = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setRawErrorOutput(null)
    setResults([])

    try {
      const res = await client.get<any[]>('/backtest/run', {
        params: { dataset, strategy, cash },
      })
      setResults(res.data)
      if (res.data.length === 0) {
        setError('No results returned. Check your parameters or data directory.')
      }
    } catch (err: any) {
      // Axios error shape
      const msg = err.response?.data?.error || err.message
      setError(msg)
      setRawErrorOutput(err.response?.data?.rawOutput || null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>Run a Backtest</h2>
      <form onSubmit={handleRun} style={{ marginBottom: 20 }}>
        <label>
          Dataset:{' '}
          <select
            value={dataset}
            onChange={e => setDataset(e.target.value)}
          >
            {DATASETS.map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          Strategy:{' '}
          <select
            value={strategy}
            onChange={e => setStrategy(e.target.value)}
          >
            {STRATEGIES.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>{' '}
        <label>
          Cash:{' '}
          <input
            type="number"
            value={cash}
            onChange={e => setCash(Number(e.target.value))}
            min={0}
          />
        </label>{' '}
        <button type="submit" disabled={loading}>
          {loading ? 'Running…' : 'Run Backtest'}
        </button>
      </form>

      {error && (
        <div style={{ color: 'red', marginBottom: 20 }}>
          <p><strong>Error:</strong> {error}</p>
          {rawErrorOutput && (
            <pre style={{ background: '#f5f5f5', padding: 10 }}>
              {rawErrorOutput}
            </pre>
          )}
        </div>
      )}

      {!loading && results.length > 0 && (
        <div>
          <h3>Results</h3>
          <table border={1} cellPadding={5} style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {Object.keys(results[0]).map(key => (
                  <th key={key}>{key.replace(/_/g, ' ')}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((row, i) => (
                <tr key={i}>
                  {Object.values(row).map((val, j) => (
                    <td key={j}>{typeof val === 'number' ? val.toFixed(4) : String(val)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}