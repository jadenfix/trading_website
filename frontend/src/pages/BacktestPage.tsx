// frontend/src/pages/BacktestPage.tsx
import React, { useState, useMemo, type FormEvent, useEffect } from 'react'
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
  'btc_2024': [
    'MAC_5_20','VWAP_2.0','ORB_30','Momentum_5_10_2_3',
    'Bayesian_ML','CNN_LSTM','RNN_Momentum'
  ],
  'eth_2024': [
    'MAC_5_20','VWAP_2.0','ORB_30','Momentum_5_10_2_3',
    'Bayesian_ML','CNN_LSTM','RNN_Momentum'
  ]
}

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  'MAC_5_20': 'Moving Average Crossover (5-day vs 20-day)',
  'VWAP_2.0': 'Volume Weighted Average Price strategy',
  'ORB_30': 'Opening Range Breakout (30-minute)',
  'Momentum_5_10_2_3': 'Multi-timeframe momentum strategy',
  'Bayesian_ML': 'Bayesian Machine Learning with uncertainty estimation',
  'CNN_LSTM': 'Convolutional LSTM for pattern recognition',
  'RNN_Momentum': 'Recurrent Neural Network momentum strategy',
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

interface DatasetInfo {
  dataset: string;
  available_dates: string[];
  symbols: string[];
  total_records: number;
}

interface SimulationResult {
  simulation_id: string;
  parameters: any;
  results: {
    initial_cash: number;
    final_equity: number;
    total_return_pct: number;
    total_trades: number;
    winning_trades: number;
    win_rate_pct: number;
    trades: any[];
    equity_curve: any[];
  };
  metadata: any;
}

export default function BacktestPage() {
  const [dataset, setDataset]   = useState('stocks_april')
  const [strategy, setStrategy] = useState(STRATEGIES_BY_DATASET['stocks_april'][0])
  const [cash, setCash]         = useState(100000)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string|null>(null)
  const [results, setResults]   = useState<any[]>([])
  
  // ML Prediction state
  const [mlLoading, setMlLoading] = useState(false)
  const [mlError, setMlError] = useState<string|null>(null)
  const [mlPrediction, setMlPrediction] = useState<any>(null)
  const [marketData, setMarketData] = useState({
    open: 50000,
    high: 51000,
    low: 49500,
    close: 50500,
    volume: 1000000,
    asset: 'btc'
  })

  // Date-based prediction state
  const [useHistoricalData, setUseHistoricalData] = useState(true)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [selectedDate, setSelectedDate] = useState('')
  const [selectedSymbol, setSelectedSymbol] = useState('')
  const [historicalDataLoading, setHistoricalDataLoading] = useState(false)

  // Simulation state
  const [simulationLoading, setSimulationLoading] = useState(false)
  const [simulationError, setSimulationError] = useState<string|null>(null)
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null)
  const [simulationParams, setSimulationParams] = useState({
    dataset: 'btc_2024',
    startDate: '2024-01-01',
    endDate: '2024-01-05',
    symbol: 'BTC',
    initialCash: 100000,
    confidenceThreshold: 0.7
  })

  // Advanced simulation state
  const [advancedSimLoading, setAdvancedSimLoading] = useState(false)
  const [advancedSimError, setAdvancedSimError] = useState<string|null>(null)
  const [advancedSimResult, setAdvancedSimResult] = useState<any>(null)
  const [advancedSimParams, setAdvancedSimParams] = useState({
    dataset: 'btc_2024',
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    symbol: 'BTC',
    initialCash: 100000,
    strategy: 'ML_BAYESIAN',
    confidenceThreshold: 0.7,
    riskLevel: 'medium',
    positionSizing: 'kelly'
  })

  // whenever dataset changes, reset strategy to first in that list
  useMemo(() => {
    setStrategy(STRATEGIES_BY_DATASET[dataset][0])
  }, [dataset])

  // Load dataset info when prediction dataset changes
  useEffect(() => {
    if (useHistoricalData) {
      loadDatasetInfo(simulationParams.dataset)
    }
  }, [simulationParams.dataset, useHistoricalData])

  const loadDatasetInfo = async (datasetName: string) => {
    setHistoricalDataLoading(true)
    try {
      const res = await client.get<DatasetInfo>(`/backtest/dates/${datasetName}`)
      setDatasetInfo(res.data)
      if (res.data.available_dates.length > 0) {
        setSelectedDate(res.data.available_dates[0])
      }
      if (res.data.symbols.length > 0) {
        setSelectedSymbol(res.data.symbols[0])
      }
    } catch (err: any) {
      console.error('Failed to load dataset info:', err)
    } finally {
      setHistoricalDataLoading(false)
    }
  }

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

  const handleMLPredict = async () => {
    setMlLoading(true)
    setMlError(null)
    try {
      let requestData;
      
      if (useHistoricalData && selectedDate && selectedSymbol) {
        // Use date-based lookup
        requestData = {
          dataset: simulationParams.dataset,
          date: selectedDate,
          symbol: selectedSymbol
        }
      } else {
        // Use manual OHLCV input
        requestData = marketData
      }
      
      const res = await client.post('/backtest/predict', requestData)
      setMlPrediction(res.data)
    } catch (err: any) {
      setMlError(err.response?.data?.error || err.message)
    } finally {
      setMlLoading(false)
    }
  }

  const handleSimulation = async () => {
    setSimulationLoading(true)
    setSimulationError(null)
    try {
      const res = await client.post<SimulationResult>('/backtest/simulate', simulationParams)
      setSimulationResult(res.data)
    } catch (err: any) {
      setSimulationError(err.response?.data?.error || err.message)
    } finally {
      setSimulationLoading(false)
    }
  }

  const handleAdvancedSimulation = async () => {
    setAdvancedSimLoading(true)
    setAdvancedSimError(null)
    try {
      const res = await client.post('/backtest/simulate-advanced', advancedSimParams)
      setAdvancedSimResult(res.data)
    } catch (err: any) {
      setAdvancedSimError(err.response?.data?.error || err.message)
    } finally {
      setAdvancedSimLoading(false)
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
        <h1>📈 Advanced Trading Engine</h1>
        <p>Test strategies, get ML predictions, and run simulations with our Bayesian engine</p>
      </div>

      <div className="trading-grid">
        {/* Traditional Backtesting */}
        <div className="trading-section">
          <h2>🎯 Strategy Backtesting</h2>
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

            {error && (
              <div className="error-message">
                <span>❌</span>
                {error}
              </div>
            )}

            {results.length > 0 && (
              <div className="results-container">
                <h3>📊 Backtest Results</h3>
                <div className="results-table-container">
                  <table className="results-table">
                    <thead>
                      <tr>
                        {Object.keys(results[0]).map(key => (
                          <th key={key}>{key.replace(/_/g, ' ').toUpperCase()}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((result, idx) => (
                        <tr key={idx}>
                          {Object.entries(result).map(([key, value], colIdx) => (
                            <td key={colIdx} className={isNumericColumn(key, value) ? 'numeric' : ''}>
                              {key.includes('pct') || key.includes('return') ? 
                                formatPercentage(getNumericValue(value) / 100) :
                                key.includes('pnl') || key.includes('equity') || key.includes('commission') ?
                                formatCurrency(value) :
                                String(value)
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
        </div>

        {/* ML Prediction Engine */}
        <div className="trading-section">
          <h2>🧠 Bayesian ML Prediction Engine</h2>
          <div className="ml-prediction-container">
            <div className="prediction-mode-toggle">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={useHistoricalData}
                  onChange={e => setUseHistoricalData(e.target.checked)}
                />
                <span className="toggle-text">
                  {useHistoricalData ? '📅 Use Historical Data' : '✍️ Manual Input'}
                </span>
              </label>
            </div>

            {useHistoricalData ? (
              <div className="historical-data-form">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Dataset</span>
                    <select 
                      value={simulationParams.dataset} 
                      onChange={e => setSimulationParams(prev => ({...prev, dataset: e.target.value}))}
                      className="form-select"
                    >
                      <option value="stocks_april">Stocks April 2024</option>
                      <option value="btc_2024">Bitcoin 2024</option>
                      <option value="eth_2024">Ethereum 2024</option>
                    </select>
                  </label>
                </div>

                {datasetInfo && (
                  <>
                    <div className="form-group">
                      <label className="form-label">
                        <span className="label-text">Symbol</span>
                        <select 
                          value={selectedSymbol} 
                          onChange={e => setSelectedSymbol(e.target.value)}
                          className="form-select"
                        >
                          {datasetInfo.symbols.map(symbol => (
                            <option key={symbol} value={symbol}>{symbol}</option>
                          ))}
                        </select>
                      </label>
                    </div>

                    <div className="form-group">
                      <label className="form-label">
                        <span className="label-text">Date</span>
                        <select 
                          value={selectedDate} 
                          onChange={e => setSelectedDate(e.target.value)}
                          className="form-select"
                        >
                          {datasetInfo.available_dates.map(date => (
                            <option key={date} value={date}>{date}</option>
                          ))}
                        </select>
                      </label>
                    </div>

                    <div className="dataset-info">
                      <span className="info-badge">
                        📊 {datasetInfo.total_records} records
                      </span>
                      <span className="info-badge">
                        📅 {datasetInfo.available_dates.length} dates
                      </span>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="manual-input-form">
                <div className="market-data-grid">
                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-text">Asset</span>
                      <select 
                        value={marketData.asset} 
                        onChange={e => setMarketData(prev => ({...prev, asset: e.target.value}))}
                        className="form-select"
                      >
                        <option value="btc">Bitcoin (BTC)</option>
                        <option value="eth">Ethereum (ETH)</option>
                        <option value="sol">Solana (SOL)</option>
                        <option value="ada">Cardano (ADA)</option>
                      </select>
                    </label>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-text">Open</span>
                      <input
                        type="number"
                        value={marketData.open}
                        onChange={e => setMarketData(prev => ({...prev, open: Number(e.target.value)}))}
                        className="form-input"
                        step="0.01"
                      />
                    </label>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-text">High</span>
                      <input
                        type="number"
                        value={marketData.high}
                        onChange={e => setMarketData(prev => ({...prev, high: Number(e.target.value)}))}
                        className="form-input"
                        step="0.01"
                      />
                    </label>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-text">Low</span>
                      <input
                        type="number"
                        value={marketData.low}
                        onChange={e => setMarketData(prev => ({...prev, low: Number(e.target.value)}))}
                        className="form-input"
                        step="0.01"
                      />
                    </label>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-text">Close</span>
                      <input
                        type="number"
                        value={marketData.close}
                        onChange={e => setMarketData(prev => ({...prev, close: Number(e.target.value)}))}
                        className="form-input"
                        step="0.01"
                      />
                    </label>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <span className="label-text">Volume</span>
                      <input
                        type="number"
                        value={marketData.volume}
                        onChange={e => setMarketData(prev => ({...prev, volume: Number(e.target.value)}))}
                        className="form-input"
                        step="1000"
                      />
                    </label>
                  </div>
                </div>
              </div>
            )}

            <button 
              onClick={handleMLPredict}
              disabled={mlLoading || (useHistoricalData && (!selectedDate || !selectedSymbol))}
              className={`predict-button ${mlLoading ? 'loading' : ''}`}
            >
              {mlLoading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                <>
                  <span>🔮</span>
                  Get ML Prediction
                </>
              )}
            </button>

            {mlError && (
              <div className="error-message">
                <span>❌</span>
                {mlError}
              </div>
            )}

            {mlPrediction && (
              <div className="prediction-results">
                <h3>🎯 Prediction Results</h3>
                <div className="prediction-cards">
                  <div className={`prediction-card signal-${mlPrediction.ml_prediction.signal.toLowerCase()}`}>
                    <div className="card-header">
                      <span className="card-icon">📊</span>
                      <span className="card-title">Signal</span>
                    </div>
                    <div className="card-value">
                      {mlPrediction.ml_prediction.signal}
                    </div>
                  </div>

                  <div className="prediction-card">
                    <div className="card-header">
                      <span className="card-icon">📈</span>
                      <span className="card-title">Prediction</span>
                    </div>
                    <div className="card-value">
                      {(mlPrediction.ml_prediction.prediction * 100).toFixed(3)}%
                    </div>
                  </div>

                  <div className="prediction-card">
                    <div className="card-header">
                      <span className="card-icon">🎯</span>
                      <span className="card-title">Confidence</span>
                    </div>
                    <div className="card-value">
                      {(mlPrediction.ml_prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="prediction-card">
                    <div className="card-header">
                      <span className="card-icon">⚠️</span>
                      <span className="card-title">Uncertainty</span>
                    </div>
                    <div className="card-value">
                      {(mlPrediction.ml_prediction.uncertainty * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>

                {mlPrediction.market_data && (
                  <div className="market-data-display">
                    <h4>📊 Market Data ({mlPrediction.market_data.date || 'Current'})</h4>
                    <div className="market-data-row">
                      <span>Open: {formatCurrency(mlPrediction.market_data.open)}</span>
                      <span>High: {formatCurrency(mlPrediction.market_data.high)}</span>
                      <span>Low: {formatCurrency(mlPrediction.market_data.low)}</span>
                      <span>Close: {formatCurrency(mlPrediction.market_data.close)}</span>
                      <span>Volume: {mlPrediction.market_data.volume.toLocaleString()}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ML Simulation Engine */}
        <div className="trading-section">
          <h2>🎯 ML Trading Simulation</h2>
          <div className="simulation-container">
            <div className="simulation-form">
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Dataset</span>
                    <select 
                      value={simulationParams.dataset} 
                      onChange={e => setSimulationParams(prev => ({...prev, dataset: e.target.value}))}
                      className="form-select"
                    >
                      <option value="btc_2024">Bitcoin 2024</option>
                      <option value="eth_2024">Ethereum 2024</option>
                      <option value="stocks_april">Stocks April 2024</option>
                    </select>
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Symbol</span>
                    <select 
                      value={simulationParams.symbol} 
                      onChange={e => setSimulationParams(prev => ({...prev, symbol: e.target.value}))}
                      className="form-select"
                    >
                      <option value="BTC">Bitcoin</option>
                      <option value="ETH">Ethereum</option>
                      <option value="MSFT">Microsoft</option>
                    </select>
                  </label>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Start Date</span>
                    <input
                      type="date"
                      value={simulationParams.startDate}
                      onChange={e => setSimulationParams(prev => ({...prev, startDate: e.target.value}))}
                      className="form-input"
                    />
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">End Date</span>
                    <input
                      type="date"
                      value={simulationParams.endDate}
                      onChange={e => setSimulationParams(prev => ({...prev, endDate: e.target.value}))}
                      className="form-input"
                    />
                  </label>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Initial Capital</span>
                    <input
                      type="number"
                      value={simulationParams.initialCash}
                      onChange={e => setSimulationParams(prev => ({...prev, initialCash: Number(e.target.value)}))}
                      className="form-input"
                      min="1000"
                      step="1000"
                    />
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Confidence Threshold</span>
                    <input
                      type="range"
                      min="0.5"
                      max="0.95"
                      step="0.05"
                      value={simulationParams.confidenceThreshold}
                      onChange={e => setSimulationParams(prev => ({...prev, confidenceThreshold: Number(e.target.value)}))}
                      className="form-range"
                    />
                    <span className="range-value">{(simulationParams.confidenceThreshold * 100).toFixed(0)}%</span>
                  </label>
                </div>
              </div>

              <button 
                onClick={handleSimulation}
                disabled={simulationLoading}
                className={`simulate-button ${simulationLoading ? 'loading' : ''}`}
              >
                {simulationLoading ? (
                  <>
                    <span className="spinner"></span>
                    Running Simulation...
                  </>
                ) : (
                  <>
                    <span>🎯</span>
                    Run ML Simulation
                  </>
                )}
              </button>
            </div>

            {simulationError && (
              <div className="error-message">
                <span>❌</span>
                {simulationError}
              </div>
            )}

            {simulationResult && (
              <div className="simulation-results">
                <h3>📊 Simulation Results</h3>
                
                <div className="simulation-summary">
                  <div className="summary-cards">
                    <div className={`summary-card ${simulationResult.results.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                      <div className="card-header">
                        <span className="card-icon">💰</span>
                        <span className="card-title">Total Return</span>
                      </div>
                      <div className="card-value">
                        {simulationResult.results.total_return_pct.toFixed(2)}%
                      </div>
                    </div>

                    <div className="summary-card">
                      <div className="card-header">
                        <span className="card-icon">🏦</span>
                        <span className="card-title">Final Equity</span>
                      </div>
                      <div className="card-value">
                        {formatCurrency(simulationResult.results.final_equity)}
                      </div>
                    </div>

                    <div className="summary-card">
                      <div className="card-header">
                        <span className="card-icon">📈</span>
                        <span className="card-title">Total Trades</span>
                      </div>
                      <div className="card-value">
                        {simulationResult.results.total_trades}
                      </div>
                    </div>

                    <div className="summary-card">
                      <div className="card-header">
                        <span className="card-icon">🎯</span>
                        <span className="card-title">Win Rate</span>
                      </div>
                      <div className="card-value">
                        {simulationResult.results.win_rate_pct.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {simulationResult.results.trades.length > 0 && (
                  <div className="trades-table-container">
                    <h4>📋 Trading History</h4>
                    <div className="trades-scroll">
                      <table className="trades-table">
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Action</th>
                            <th>Price</th>
                            <th>Quantity</th>
                            <th>Confidence</th>
                            <th>Prediction</th>
                          </tr>
                        </thead>
                        <tbody>
                          {simulationResult.results.trades.map((trade, idx) => (
                            <tr key={idx} className={`trade-${trade.action.toLowerCase()}`}>
                              <td>{trade.date}</td>
                              <td className={`action-${trade.action.toLowerCase()}`}>
                                {trade.action === 'BUY' ? '🟢' : '🔴'} {trade.action}
                              </td>
                              <td>{formatCurrency(trade.price)}</td>
                              <td>{trade.quantity}</td>
                              <td>{(trade.confidence * 100).toFixed(1)}%</td>
                              <td>{(trade.prediction * 100).toFixed(3)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Advanced Bayesian ML Simulation with financial_cpp_advanced */}
        <div className="trading-section advanced-section">
          <h2>🧠 Advanced Bayesian ML Engine</h2>
          <p className="section-description">
            Full backtesting with the financial_cpp_advanced system using Bayesian ML models, 
            uncertainty quantification, and sophisticated risk management.
          </p>
          
          <div className="simulation-container">
            <div className="simulation-form">
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Dataset</span>
                    <select 
                      value={advancedSimParams.dataset} 
                      onChange={e => setAdvancedSimParams(prev => ({...prev, dataset: e.target.value}))}
                      className="form-select"
                    >
                      <option value="btc_2024">Bitcoin 2024</option>
                      <option value="eth_2024">Ethereum 2024</option>
                      <option value="stocks_april">Stocks April 2024</option>
                    </select>
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Symbol</span>
                    <select 
                      value={advancedSimParams.symbol} 
                      onChange={e => setAdvancedSimParams(prev => ({...prev, symbol: e.target.value}))}
                      className="form-select"
                    >
                      <option value="BTC">Bitcoin</option>
                      <option value="ETH">Ethereum</option>
                      <option value="MSFT">Microsoft</option>
                    </select>
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Strategy</span>
                    <select 
                      value={advancedSimParams.strategy} 
                      onChange={e => setAdvancedSimParams(prev => ({...prev, strategy: e.target.value}))}
                      className="form-select"
                    >
                      <option value="ML_BAYESIAN">Bayesian ML Ensemble</option>
                      <option value="ML_NEURAL">Neural Network Strategy</option>
                      <option value="ML_HYBRID">Hybrid ML Strategy</option>
                    </select>
                  </label>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Start Date</span>
                    <input
                      type="date"
                      value={advancedSimParams.startDate}
                      onChange={e => setAdvancedSimParams(prev => ({...prev, startDate: e.target.value}))}
                      className="form-input"
                    />
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">End Date</span>
                    <input
                      type="date"
                      value={advancedSimParams.endDate}
                      onChange={e => setAdvancedSimParams(prev => ({...prev, endDate: e.target.value}))}
                      className="form-input"
                    />
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Initial Capital</span>
                    <input
                      type="number"
                      value={advancedSimParams.initialCash}
                      onChange={e => setAdvancedSimParams(prev => ({...prev, initialCash: Number(e.target.value)}))}
                      className="form-input"
                      min="10000"
                      step="10000"
                    />
                  </label>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Confidence Threshold</span>
                    <input
                      type="range"
                      min="0.5"
                      max="0.95"
                      step="0.05"
                      value={advancedSimParams.confidenceThreshold}
                      onChange={e => setAdvancedSimParams(prev => ({...prev, confidenceThreshold: Number(e.target.value)}))}
                      className="form-range"
                    />
                    <span className="range-value">{(advancedSimParams.confidenceThreshold * 100).toFixed(0)}%</span>
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Risk Level</span>
                    <select 
                      value={advancedSimParams.riskLevel} 
                      onChange={e => setAdvancedSimParams(prev => ({...prev, riskLevel: e.target.value}))}
                      className="form-select"
                    >
                      <option value="low">Low Risk (10% max position)</option>
                      <option value="medium">Medium Risk (20% max position)</option>
                      <option value="high">High Risk (30% max position)</option>
                    </select>
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span className="label-text">Position Sizing</span>
                    <select 
                      value={advancedSimParams.positionSizing} 
                      onChange={e => setAdvancedSimParams(prev => ({...prev, positionSizing: e.target.value}))}
                      className="form-select"
                    >
                      <option value="kelly">Kelly Criterion</option>
                      <option value="fixed">Fixed Percentage</option>
                      <option value="volatility">Volatility Adjusted</option>
                    </select>
                  </label>
                </div>
              </div>

              <button 
                onClick={handleAdvancedSimulation}
                disabled={advancedSimLoading}
                className={`simulate-button advanced ${advancedSimLoading ? 'loading' : ''}`}
              >
                {advancedSimLoading ? (
                  <>
                    <span className="spinner"></span>
                    Running Advanced Simulation...
                  </>
                ) : (
                  <>
                    <span>🧠</span>
                    Run Advanced ML Backtest
                  </>
                )}
              </button>
            </div>

            {advancedSimError && (
              <div className="error-message">
                <span>❌</span>
                {advancedSimError}
              </div>
            )}

            {advancedSimResult && (
              <div className="simulation-results advanced-results">
                <h3>🎯 Advanced ML Backtest Results</h3>
                <div className="strategy-info">
                  <h4>{advancedSimResult.strategy_name}</h4>
                  <p>Strategy: {advancedSimResult.parameters.risk_level} risk, {advancedSimResult.parameters.position_sizing} sizing</p>
                </div>
                
                <div className="performance-grid">
                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">💰</span>
                      <span className="card-title">Total Return</span>
                    </div>
                    <div className="card-value large">
                      {advancedSimResult.performance.total_return}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">📊</span>
                      <span className="card-title">Sharpe Ratio</span>
                    </div>
                    <div className="card-value">
                      {advancedSimResult.performance.sharpe_ratio}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">📉</span>
                      <span className="card-title">Max Drawdown</span>
                    </div>
                    <div className="card-value negative">
                      {advancedSimResult.performance.max_drawdown}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">🎯</span>
                      <span className="card-title">Win Rate</span>
                    </div>
                    <div className="card-value">
                      {advancedSimResult.performance.win_rate}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">📈</span>
                      <span className="card-title">Total Trades</span>
                    </div>
                    <div className="card-value">
                      {advancedSimResult.performance.total_trades}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">💵</span>
                      <span className="card-title">Final Value</span>
                    </div>
                    <div className="card-value">
                      {advancedSimResult.performance.final_value}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">📊</span>
                      <span className="card-title">Volatility</span>
                    </div>
                    <div className="card-value">
                      {advancedSimResult.performance.volatility}
                    </div>
                  </div>

                  <div className="perf-card">
                    <div className="card-header">
                      <span className="card-icon">💸</span>
                      <span className="card-title">Total Fees</span>
                    </div>
                    <div className="card-value">
                      {advancedSimResult.performance.total_fees}
                    </div>
                  </div>
                </div>

                {advancedSimResult.trades && advancedSimResult.trades.length > 0 && (
                  <div className="trades-table-container">
                    <h4>📋 ML Trading History</h4>
                    <div className="trades-scroll">
                      <table className="trades-table">
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Action</th>
                            <th>Price</th>
                            <th>Quantity</th>
                            <th>Value</th>
                            <th>Confidence</th>
                            <th>Signal Strength</th>
                            <th>Portfolio Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {advancedSimResult.trades.map((trade, idx) => (
                            <tr key={idx} className={`trade-${trade.action.toLowerCase()}`}>
                              <td>{trade.date}</td>
                              <td className={`action-${trade.action.toLowerCase()}`}>
                                {trade.action === 'BUY' ? '🟢' : '🔴'} {trade.action}
                              </td>
                              <td>{formatCurrency(trade.price)}</td>
                              <td>{trade.quantity.toLocaleString()}</td>
                              <td>{formatCurrency(trade.value)}</td>
                              <td>{(trade.confidence * 100).toFixed(1)}%</td>
                              <td>{(trade.signal_strength * 100).toFixed(2)}%</td>
                              <td>{formatCurrency(trade.portfolio_value)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}