import React, { useState, useRef, useEffect } from 'react'
import { client } from '../api/client'
import './CSVBacktestPage.css'

interface Strategy {
  name: string
  config: {
    params: string[]
    defaults: Record<string, any>
    description: string
  }
}

interface UploadedDataset {
  data_id: string
  symbol: string
  stats: {
    total_rows: number
    date_range: { start: string; end: string }
    symbols: string[]
  }
  uploaded_at: string
  size_mb: string
}

interface BacktestResult {
  strategy_name: string
  symbol: string
  data_period: { start: string; end: string }
  total_return_pct: number
  max_drawdown_pct: number
  sharpe_ratio: number
  realized_pnl: number
  total_commission: number
  num_fills: number
  final_equity: number
  win_rate: number
  total_trades: number
  data_points: number
  strategy_params: Record<string, any>
  mock?: boolean
  error?: string
}

export default function CSVBacktestPage() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [strategyParams, setStrategyParams] = useState<Record<string, any>>({})
  const [uploadedDatasets, setUploadedDatasets] = useState<UploadedDataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  
  // Upload state
  const [file, setFile] = useState<File | null>(null)
  const [symbol, setSymbol] = useState<string>('')
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Backtest state
  const [backtesting, setBacktesting] = useState(false)
  const [backtestError, setBacktestError] = useState<string | null>(null)
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([])
  const [initialCash, setInitialCash] = useState(100000)
  const [commission, setCommission] = useState(0.001)

  // Load strategies and datasets on mount
  useEffect(() => {
    loadStrategies()
    loadUploadedDatasets()
  }, [])

  // Update strategy params when strategy changes
  useEffect(() => {
    if (selectedStrategy) {
      const strategy = strategies.find(s => s.name === selectedStrategy)
      if (strategy) {
        setStrategyParams({ ...strategy.config.defaults })
      }
    }
  }, [selectedStrategy, strategies])

  const loadStrategies = async () => {
    try {
      const res = await client.get<{ strategies: Strategy[] }>('/backtest/strategies')
      setStrategies(res.data.strategies)
      if (res.data.strategies.length > 0) {
        setSelectedStrategy(res.data.strategies[0].name)
      }
    } catch (err: any) {
      console.error('Failed to load strategies:', err)
    }
  }

  const loadUploadedDatasets = async () => {
    try {
      const res = await client.get<{ datasets: UploadedDataset[] }>('/backtest/uploaded-datasets')
      setUploadedDatasets(res.data.datasets)
    } catch (err: any) {
      console.error('Failed to load uploaded datasets:', err)
    }
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.type !== 'text/csv' && !selectedFile.name.endsWith('.csv')) {
        setUploadError('Please select a CSV file')
        return
      }
      setFile(selectedFile)
      setUploadError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setUploadError('Please select a CSV file')
      return
    }

    setUploading(true)
    setUploadError(null)

    try {
      const formData = new FormData()
      formData.append('csv', file)
      if (symbol) {
        formData.append('symbol', symbol)
      }

      const res = await client.post('/backtest/upload-csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      console.log('✅ CSV uploaded successfully:', res.data)
      
      // Refresh the datasets list
      await loadUploadedDatasets()
      
      // Clear form
      setFile(null)
      setSymbol('')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      // Auto-select the newly uploaded dataset
      setSelectedDataset(res.data.data_id)

    } catch (err: any) {
      setUploadError(err.response?.data?.error || err.message || 'Upload failed')
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
    }
  }

  const handleBacktest = async () => {
    if (!selectedDataset || !selectedStrategy) {
      setBacktestError('Please select a dataset and strategy')
      return
    }

    setBacktesting(true)
    setBacktestError(null)

    try {
      const res = await client.post<BacktestResult>('/backtest/backtest-csv', {
        data_id: selectedDataset,
        strategy: selectedStrategy,
        strategy_params: strategyParams,
        initial_cash: initialCash,
        commission: commission
      })

      setBacktestResults(prev => [res.data, ...prev])
      console.log('✅ Backtest completed:', res.data)

    } catch (err: any) {
      setBacktestError(err.response?.data?.error || err.message || 'Backtest failed')
      console.error('Backtest error:', err)
    } finally {
      setBacktesting(false)
    }
  }

  const handleParamChange = (paramName: string, value: any) => {
    setStrategyParams(prev => ({
      ...prev,
      [paramName]: value
    }))
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`
  }

  const getSelectedDatasetInfo = () => {
    return uploadedDatasets.find(d => d.data_id === selectedDataset)
  }

  const getSelectedStrategyInfo = () => {
    return strategies.find(s => s.name === selectedStrategy)
  }

  return (
    <div className="csv-backtest-page">
      <div className="page-header">
        <h1>📊 Custom CSV Backtesting</h1>
        <p>Upload your own OHLCV data and test with advanced C++ strategies</p>
      </div>

      <div className="backtest-grid">
        {/* CSV Upload Section */}
        <div className="section upload-section">
          <h2>📁 Upload CSV Data</h2>
          <p className="section-description">
            Upload CSV files with columns: timestamp, open, high, low, close, volume
          </p>
          
          <div className="upload-form">
            <div className="form-group">
              <label>CSV File</label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="file-input"
              />
              {file && (
                <div className="file-info">
                  📄 {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </div>
              )}
            </div>

            <div className="form-group">
              <label>Symbol (optional)</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="e.g., AAPL, BTC, SPY"
                className="text-input"
              />
              <small>Leave empty to auto-detect from CSV</small>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className={`upload-button ${uploading ? 'loading' : ''}`}
            >
              {uploading ? (
                <>
                  <span className="spinner"></span>
                  Uploading...
                </>
              ) : (
                <>
                  <span>📤</span>
                  Upload CSV
                </>
              )}
            </button>

            {uploadError && (
              <div className="error-message">
                ❌ {uploadError}
              </div>
            )}
          </div>
        </div>

        {/* Strategy Selection */}
        <div className="section strategy-section">
          <h2>⚡ Strategy Configuration</h2>
          
          <div className="form-group">
            <label>Strategy</label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="strategy-select"
            >
              {strategies.map(strategy => (
                <option key={strategy.name} value={strategy.name}>
                  {strategy.name}
                </option>
              ))}
            </select>
            {getSelectedStrategyInfo() && (
              <div className="strategy-description">
                {getSelectedStrategyInfo()!.config.description}
              </div>
            )}
          </div>

          {/* Strategy Parameters */}
          {getSelectedStrategyInfo() && getSelectedStrategyInfo()!.config.params.length > 0 && (
            <div className="strategy-params">
              <h3>Parameters</h3>
              {getSelectedStrategyInfo()!.config.params.map(param => (
                <div key={param} className="param-group">
                  <label>{param.replace(/_/g, ' ')}</label>
                  <input
                    type="number"
                    value={strategyParams[param] || ''}
                    onChange={(e) => handleParamChange(param, parseFloat(e.target.value) || 0)}
                    step={param.includes('rate') || param.includes('threshold') ? 0.01 : 1}
                    className="param-input"
                  />
                </div>
              ))}
            </div>
          )}

          <div className="backtest-params">
            <h3>Backtest Settings</h3>
            <div className="param-group">
              <label>Initial Cash</label>
              <input
                type="number"
                value={initialCash}
                onChange={(e) => setInitialCash(parseInt(e.target.value) || 0)}
                className="param-input"
              />
            </div>
            <div className="param-group">
              <label>Commission (%)</label>
              <input
                type="number"
                value={commission}
                onChange={(e) => setCommission(parseFloat(e.target.value) || 0)}
                step={0.001}
                className="param-input"
              />
            </div>
          </div>
        </div>

        {/* Dataset Selection & Backtest */}
        <div className="section dataset-section">
          <h2>🎯 Run Backtest</h2>
          
          <div className="form-group">
            <label>Select Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="dataset-select"
            >
              <option value="">Select uploaded dataset...</option>
              {uploadedDatasets.map(dataset => (
                <option key={dataset.data_id} value={dataset.data_id}>
                  {dataset.symbol} ({dataset.stats.total_rows} rows) - {new Date(dataset.uploaded_at).toLocaleDateString()}
                </option>
              ))}
            </select>
          </div>

          {getSelectedDatasetInfo() && (
            <div className="dataset-info">
              <h3>Dataset Info</h3>
              <div className="info-grid">
                <div>Symbol: <strong>{getSelectedDatasetInfo()!.symbol}</strong></div>
                <div>Rows: <strong>{getSelectedDatasetInfo()!.stats.total_rows}</strong></div>
                <div>Period: <strong>{getSelectedDatasetInfo()!.stats.date_range.start}</strong> to <strong>{getSelectedDatasetInfo()!.stats.date_range.end}</strong></div>
                <div>Size: <strong>{getSelectedDatasetInfo()!.size_mb} MB</strong></div>
              </div>
            </div>
          )}

          <button
            onClick={handleBacktest}
            disabled={!selectedDataset || !selectedStrategy || backtesting}
            className={`backtest-button ${backtesting ? 'loading' : ''}`}
          >
            {backtesting ? (
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

          {backtestError && (
            <div className="error-message">
              ❌ {backtestError}
            </div>
          )}
        </div>
      </div>

      {/* Results Section */}
      {backtestResults.length > 0 && (
        <div className="results-section">
          <h2>📈 Backtest Results</h2>
          <div className="results-container">
            {backtestResults.map((result, index) => (
              <div key={index} className="result-card">
                <div className="result-header">
                  <h3>{result.strategy_name} on {result.symbol}</h3>
                  <div className="result-period">
                    {result.data_period.start} → {result.data_period.end}
                  </div>
                  {result.mock && <span className="mock-badge">Mock Result</span>}
                  {result.error && <span className="error-badge">Error</span>}
                </div>
                
                <div className="result-metrics">
                  <div className="metric">
                    <span className="metric-label">Total Return</span>
                    <span className={`metric-value ${result.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                      {formatPercentage(result.total_return_pct)}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Sharpe Ratio</span>
                    <span className="metric-value">{result.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Max Drawdown</span>
                    <span className="metric-value negative">{formatPercentage(result.max_drawdown_pct)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Win Rate</span>
                    <span className="metric-value">{formatPercentage(result.win_rate)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Total Trades</span>
                    <span className="metric-value">{result.total_trades}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Final Equity</span>
                    <span className="metric-value">{formatCurrency(result.final_equity)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">P&L</span>
                    <span className={`metric-value ${result.realized_pnl >= 0 ? 'positive' : 'negative'}`}>
                      {formatCurrency(result.realized_pnl)}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Data Points</span>
                    <span className="metric-value">{result.data_points}</span>
                  </div>
                </div>

                {Object.keys(result.strategy_params).length > 0 && (
                  <div className="strategy-params-display">
                    <h4>Strategy Parameters</h4>
                    <div className="params-grid">
                      {Object.entries(result.strategy_params).map(([key, value]) => (
                        <div key={key} className="param-display">
                          <span className="param-name">{key.replace(/_/g, ' ')}</span>
                          <span className="param-value">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
} 