// frontend/src/pages/BacktestPage.tsx
import React, { useState, useMemo, type FormEvent } from 'react'
import { client } from '../api/client'
import './BacktestPage.css'

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

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  'MAC_5_20': 'Moving Average Crossover (5-day vs 20-day)',
  'VWAP_2.0': 'Volume Weighted Average Price strategy',
  'ORB_30': 'Opening Range Breakout (30-minute)',
  'Momentum_5_10_2_3': 'Multi-timeframe momentum strategy',
  'Pairs_MSFT_NVDA': 'Pairs trading between Microsoft and Nvidia',
  'Pairs_NVDA_GOOG': 'Pairs trading between Nvidia and Google',
  'Pairs_MSFT_GOOG': 'Pairs trading between Microsoft and Google',
  'Pairs_BTC_ETH': 'Cryptocurrency pairs trading BTC/ETH',
  'Pairs_ETH_SOL': 'Cryptocurrency pairs trading ETH/SOL',
  'Pairs_BTC_SOL': 'Cryptocurrency pairs trading BTC/SOL',
  'Pairs_ETH_ADA': 'Cryptocurrency pairs trading ETH/ADA',
  'Pairs_SOL_ADA': 'Cryptocurrency pairs trading SOL/ADA',
  'LeadLag_MSFT->NVDA': 'Lead-lag strategy with MSFT leading NVDA',
  'LeadLag_NVDA->MSFT': 'Lead-lag strategy with NVDA leading MSFT',
  'LeadLag_BTC->ETH': 'Lead-lag strategy with BTC leading ETH',
  'LeadLag_ETH->BTC': 'Lead-lag strategy with ETH leading BTC',
  'LeadLag_ETH->SOL': 'Lead-lag strategy with ETH leading SOL',
  'LeadLag_SOL->ETH': 'Lead-lag strategy with SOL leading ETH',
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

  const formatCurrency = (value: any) => {
    const num = Number(value)
    if (isNaN(num)) return value
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(num)
  }

  const formatPercentage = (value: any) => {
    const num = Number(value)
    if (isNaN(num)) return value
    return `${(num * 100).toFixed(2)}%`
  }

  const isNumericColumn = (key: string, value: any) => {
    return !isNaN(Number(value)) && key !== 'timestamp' && key !== 'date'
  }

  const getNumericValue = (value: any) => {
    return Number(value)
  }

  return (
    <div className="backtest-page">
      <div className="backtest-header">
        <h1>📈 Strategy Backtesting</h1>
        <p>Test your trading strategies against historical data using our C++ engine</p>
      </div>

      <div className="backtest-form-container">
        <form onSubmit={handleRun} className="backtest-form">
          <div className="form-group">
            <label className="form-label">
              <span className="label-text">Dataset</span>
              <select 
                value={dataset} 
                onChange={e => setDataset(e.target.value)}
                className="form-select"
              >
                {Object.keys(STRATEGIES_BY_DATASET).map(ds => (
                  <option key={ds} value={ds}>{ds}</option>
                ))}
              </select>
            </label>
          </div>

          <div className="form-group">
            <label className="form-label">
              <span className="label-text">Strategy</span>
              <select 
                value={strategy} 
                onChange={e => setStrategy(e.target.value)}
                className="form-select"
              >
                {STRATEGIES_BY_DATASET[dataset].map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            {STRATEGY_DESCRIPTIONS[strategy] && (
              <div className="strategy-description">
                {STRATEGY_DESCRIPTIONS[strategy]}
              </div>
            )}
          </div>

          <div className="form-group">
            <label className="form-label">
              <span className="label-text">Initial Cash</span>
              <input
                type="number"
                value={cash}
                onChange={e => setCash(Number(e.target.value))}
                className="form-input"
                min="1000"
                step="1000"
              />
            </label>
          </div>

          <button 
            type="submit" 
            disabled={loading} 
            className={`run-button ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Running Backtest...
              </>
            ) : (
              <>
                <span>🚀</span>
                Run Backtest
              </>
            )}
          </button>
        </form>
      </div>

      {error && (
        <div className="error-container">
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="results-container">
          <h2>Backtest Results</h2>
          <div className="results-table-wrapper">
            <table className="results-table">
              <thead>
                <tr>
                  {Object.keys(results[0]).map(key => (
                    <th key={key} className="table-header">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((row, i) => (
                  <tr key={i} className="table-row">
                    {Object.entries(row).map(([key, value], j) => (
                      <td key={j} className="table-cell">
                        {key.toLowerCase().includes('return') || key.toLowerCase().includes('pnl') 
                          ? formatCurrency(value)
                          : key.toLowerCase().includes('rate') || key.toLowerCase().includes('percent')
                          ? formatPercentage(value)
                          : isNumericColumn(key, value) && Math.abs(getNumericValue(value)) > 1000
                          ? formatCurrency(value)
                          : String(value)
                        }
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}