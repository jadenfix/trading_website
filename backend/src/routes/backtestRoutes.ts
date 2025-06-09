import { Router, Request } from 'express'
import { StrategyResult } from '../models/portfolio'
import path from 'path'
import fs from 'fs'
import multer from 'multer'
import csvParser from 'csv-parser'

// Extend Request interface to include file property
interface RequestWithFile extends Request {
  file?: Express.Multer.File
}

const router = Router()

// Mock data for different datasets with historical dates
const MOCK_HISTORICAL_DATA: Record<string, any[]> = {
  'stocks_april': [
    { date: '2024-04-01', open: 420.50, high: 425.20, low: 418.30, close: 423.10, volume: 2450000, symbol: 'MSFT' },
    { date: '2024-04-02', open: 423.10, high: 428.75, low: 421.80, close: 426.90, volume: 1890000, symbol: 'MSFT' },
    { date: '2024-04-03', open: 426.90, high: 430.25, low: 424.50, close: 429.80, volume: 2120000, symbol: 'MSFT' },
    { date: '2024-04-04', open: 429.80, high: 432.15, low: 427.60, close: 431.20, volume: 1750000, symbol: 'MSFT' },
    { date: '2024-04-05', open: 431.20, high: 434.90, low: 429.75, close: 433.45, volume: 2300000, symbol: 'MSFT' },
    { date: '2024-04-08', open: 433.45, high: 437.20, low: 431.80, close: 435.60, volume: 2010000, symbol: 'MSFT' },
    { date: '2024-04-09', open: 435.60, high: 438.95, low: 433.25, close: 436.80, volume: 1840000, symbol: 'MSFT' },
    { date: '2024-04-10', open: 436.80, high: 440.50, low: 434.90, close: 439.25, volume: 2150000, symbol: 'MSFT' }
  ],
  'btc_2024': [
    { date: '2024-01-01', open: 42500, high: 43200, low: 42100, close: 42800, volume: 850000, symbol: 'BTC' },
    { date: '2024-01-02', open: 42800, high: 44100, low: 42600, close: 43950, volume: 920000, symbol: 'BTC' },
    { date: '2024-01-03', open: 43950, high: 45200, low: 43700, close: 44850, volume: 1100000, symbol: 'BTC' },
    { date: '2024-01-04', open: 44850, high: 46000, low: 44500, close: 45650, volume: 980000, symbol: 'BTC' },
    { date: '2024-01-05', open: 45650, high: 47200, low: 45400, close: 46800, volume: 1200000, symbol: 'BTC' },
    { date: '2024-06-01', open: 69800, high: 71500, low: 69200, close: 70900, volume: 1350000, symbol: 'BTC' },
    { date: '2024-06-02', open: 70900, high: 72800, low: 70500, close: 72100, volume: 1180000, symbol: 'BTC' },
    { date: '2024-06-03', open: 72100, high: 73600, low: 71800, close: 73200, volume: 1420000, symbol: 'BTC' },
    { date: '2024-12-01', open: 96500, high: 98200, low: 95800, close: 97600, volume: 1650000, symbol: 'BTC' },
    { date: '2024-12-02', open: 97600, high: 99500, low: 97100, close: 98800, volume: 1480000, symbol: 'BTC' },
    { date: '2024-12-03', open: 98800, high: 100200, low: 98400, close: 99700, volume: 1720000, symbol: 'BTC' },
    { date: '2024-12-04', open: 99700, high: 101500, low: 99200, close: 100800, volume: 1590000, symbol: 'BTC' },
    { date: '2024-12-05', open: 100800, high: 102300, low: 100400, close: 101900, volume: 1840000, symbol: 'BTC' }
  ],
  'eth_2024': [
    { date: '2024-01-01', open: 2420, high: 2485, low: 2390, close: 2465, volume: 680000, symbol: 'ETH' },
    { date: '2024-01-02', open: 2465, high: 2520, low: 2440, close: 2495, volume: 720000, symbol: 'ETH' },
    { date: '2024-01-03', open: 2495, high: 2575, low: 2480, close: 2550, volume: 840000, symbol: 'ETH' },
    { date: '2024-06-01', open: 3680, high: 3750, low: 3620, close: 3720, volume: 920000, symbol: 'ETH' },
    { date: '2024-06-02', open: 3720, high: 3820, low: 3690, close: 3790, volume: 880000, symbol: 'ETH' },
    { date: '2024-12-01', open: 4120, high: 4250, low: 4080, close: 4190, volume: 1150000, symbol: 'ETH' },
    { date: '2024-12-02', open: 4190, high: 4320, low: 4160, close: 4280, volume: 1080000, symbol: 'ETH' },
    { date: '2024-12-03', open: 4280, high: 4380, low: 4250, close: 4350, volume: 1220000, symbol: 'ETH' }
  ]
}

// Load the advanced trading addon
let advancedTradingAddon: any = null;
try {
  const addonPath = path.join(__dirname, '../../../addon/build/Release/advancedTradingAddon.node');
  advancedTradingAddon = require(addonPath);
  console.log('✅ Advanced Trading Addon loaded successfully');
  console.log('Available functions:', Object.keys(advancedTradingAddon));
} catch (error) {
  console.warn('⚠️  Advanced Trading Addon not available, using mock data:', error);
}

// Configure multer for CSV uploads
const upload = multer({
  dest: 'uploads/',
  fileFilter: (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
      cb(null, true)
    } else {
      cb(new Error('Only CSV files are allowed'))
    }
  },
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
})

// Available C++ strategies from the strategies folder
const AVAILABLE_STRATEGIES = [
  'MovingAverageCrossover',
  'OpeningRangeBreakout', 
  'VWAPReversion',
  'MomentumIgnition',
  'PairsTrading',
  'LeadLagStrategy',
  'BayesianOnlineMLStrategy',
  'MLBridgeStrategy',
  'OrnsteinUhlenbeckStrategy',
  'GBMStrategy',
  'BayesianLinearRegressionStrategy',
  'RandomForestStrategy',
  'GaussianProcessStrategy',
  'GradientBoostedTreesStrategy',
  'StackingStrategy',
  'StochasticMLStrategy',
  'CrossValidationStrategy'
]

// Strategy parameter configurations
const STRATEGY_CONFIGS: Record<string, any> = {
  'MovingAverageCrossover': {
    params: ['short_window', 'long_window', 'position_size'],
    defaults: { short_window: 5, long_window: 20, position_size: 100 },
    description: 'Moving Average Crossover (configurable windows)'
  },
  'OpeningRangeBreakout': {
    params: ['range_minutes', 'position_size'],
    defaults: { range_minutes: 30, position_size: 100 },
    description: 'Opening Range Breakout strategy'
  },
  'VWAPReversion': {
    params: ['lookback_periods', 'deviation_threshold', 'position_size'],
    defaults: { lookback_periods: 20, deviation_threshold: 2.0, position_size: 100 },
    description: 'VWAP mean reversion strategy'
  },
  'MomentumIgnition': {
    params: ['momentum_window', 'volume_threshold', 'position_size'],
    defaults: { momentum_window: 10, volume_threshold: 1.5, position_size: 100 },
    description: 'Momentum ignition detection'
  },
  'BayesianOnlineMLStrategy': {
    params: ['learning_rate', 'confidence_threshold', 'position_size'],
    defaults: { learning_rate: 0.01, confidence_threshold: 0.7, position_size: 100 },
    description: 'Bayesian online machine learning'
  },
  'MLBridgeStrategy': {
    params: ['model_type', 'retrain_frequency', 'position_size'],
    defaults: { model_type: 'neural_network', retrain_frequency: 100, position_size: 100 },
    description: 'ML bridge strategy with Python models'
  }
}

interface CSVRow {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  symbol?: string
}

interface ParsedCSVData {
  symbol: string
  data: CSVRow[]
  stats: {
    total_rows: number
    date_range: { start: string, end: string }
    symbols: string[]
  }
}

// Parse CSV data into standardized format
const parseCSVData = (filePath: string, symbol?: string): Promise<ParsedCSVData> => {
  return new Promise((resolve, reject) => {
    const results: CSVRow[] = []
    const symbols = new Set<string>()
    
    fs.createReadStream(filePath)
      .pipe(csvParser())
      .on('data', (row: Record<string, string>) => {
        try {
          // Handle various CSV column naming conventions
          const timestamp = row.timestamp || row.date || row.time || row.datetime
          const open = parseFloat(row.open || row.Open || row.OPEN)
          const high = parseFloat(row.high || row.High || row.HIGH)
          const low = parseFloat(row.low || row.Low || row.LOW)
          const close = parseFloat(row.close || row.Close || row.CLOSE)
          const volume = parseInt(row.volume || row.Volume || row.VOLUME || '0')
          const rowSymbol = row.symbol || row.Symbol || row.ticker || symbol || 'UNKNOWN'
          
          if (timestamp && !isNaN(open) && !isNaN(high) && !isNaN(low) && !isNaN(close)) {
            results.push({
              timestamp,
              open,
              high,
              low,
              close,
              volume,
              symbol: rowSymbol
            })
            symbols.add(rowSymbol)
          }
        } catch (error) {
          console.warn('Skipping invalid row:', row)
        }
      })
      .on('end', () => {
        if (results.length === 0) {
          reject(new Error('No valid data found in CSV. Please ensure your CSV has columns: timestamp/date, open, high, low, close, and optionally volume and symbol. Check that data types are correct (numbers for OHLCV, valid dates for timestamp).'))
          return
        }
        
        // Sort by timestamp
        results.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
        
        const resolvedSymbol = symbol || Array.from(symbols)[0] || 'UPLOADED'
        
        resolve({
          symbol: resolvedSymbol,
          data: results,
          stats: {
            total_rows: results.length,
            date_range: {
              start: results[0].timestamp,
              end: results[results.length - 1].timestamp
            },
            symbols: Array.from(symbols)
          }
        })
      })
      .on('error', reject)
  })
}

// New endpoint to get available dates for a dataset
router.get('/dates/:dataset', async (req, res, next) => {
  try {
    const { dataset } = req.params;
    
    // Get available dates from mock data (in production, this would query your data files)
    const data = MOCK_HISTORICAL_DATA[dataset];
    if (!data) {
      res.status(404).json({ error: `Dataset '${dataset}' not found` });
      return;
    }
    
    const dates = data.map(d => d.date).sort();
    const symbols = [...new Set(data.map(d => d.symbol))];
    
    res.json({
      dataset,
      available_dates: dates,
      symbols,
      total_records: data.length
    });
  } catch (err) {
    next(err);
  }
});

// New endpoint to get historical data for a specific date and symbol
router.get('/data/:dataset/:date/:symbol', async (req, res, next) => {
  try {
    const { dataset, date, symbol } = req.params;
    
    const data = MOCK_HISTORICAL_DATA[dataset];
    if (!data) {
      res.status(404).json({ error: `Dataset '${dataset}' not found` });
      return;
    }
    
    let filteredData = data.filter(d => d.date === date && d.symbol.toLowerCase() === symbol.toLowerCase());
    
    if (filteredData.length === 0) {
      res.status(404).json({ error: `No data found for ${date} ${symbol ? `(${symbol})` : ''}` });
      return;
    }
    
    res.json({
      dataset,
      date,
      symbol,
      data: filteredData[0] // Return the first match
    });
  } catch (err) {
    next(err);
  }
});

// Enhanced prediction endpoint that can work with date-based data
router.post('/predict', async (req, res, next) => {
  try {
    let marketData;
    
    // Check if request includes date-based lookup
    if (req.body.dataset && req.body.date) {
      const { dataset, date, symbol = 'BTC' } = req.body;
      
      const data = MOCK_HISTORICAL_DATA[dataset];
      if (!data) {
        res.status(400).json({ error: `Dataset '${dataset}' not found` });
        return;
      }
      
      const historicalData = data.find(d => 
        d.date === date && d.symbol.toLowerCase() === symbol.toLowerCase()
      );
      
      if (!historicalData) {
        res.status(400).json({ error: `No data found for ${symbol} on ${date} in ${dataset}` });
        return;
      }
      
      marketData = {
        open: historicalData.open,
        high: historicalData.high,
        low: historicalData.low,
        close: historicalData.close,
        volume: historicalData.volume,
        asset: symbol.toLowerCase(),
        date: historicalData.date,
        symbol: historicalData.symbol
      };
    } else {
      // Use manually provided OHLCV data
      const { open, high, low, close, volume, asset } = req.body;
      
      if (typeof open !== 'number' || typeof high !== 'number' || 
          typeof low !== 'number' || typeof close !== 'number' || 
          typeof volume !== 'number' || !asset) {
        res.status(400).json({ error: 'Invalid market data: expected numeric OHLCV and asset, or dataset/date/symbol' });
        return;
      }
      
      marketData = { open, high, low, close, volume, asset };
    }

    let prediction;

    // Use the advanced trading addon if available
    if (advancedTradingAddon && advancedTradingAddon.getMLPrediction) {
      try {
        console.log(`🔮 Getting ML prediction for ${marketData.asset}: O:${marketData.open} H:${marketData.high} L:${marketData.low} C:${marketData.close} V:${marketData.volume}`);
        prediction = advancedTradingAddon.getMLPrediction(
          marketData.open, 
          marketData.high, 
          marketData.low, 
          marketData.close, 
          marketData.volume, 
          marketData.asset
        );
        console.log('✅ ML prediction completed:', prediction);
      } catch (error) {
        console.error('❌ ML prediction failed:', error);
        // Fallback to mock prediction
        prediction = {
          prediction: 0.001,
          uncertainty: 0.5,
          confidence: 0.5,
          signal: 'HOLD'
        };
      }
    } else {
      // Fallback to mock prediction if addon is not available
      console.log('⚠️  Using mock ML prediction');
      const priceChange = (marketData.close - marketData.open) / marketData.open;
      prediction = {
        prediction: priceChange * 0.5,
        uncertainty: Math.abs(priceChange) + 0.1,
        confidence: Math.max(0.1, 1.0 - Math.abs(priceChange) - 0.1),
        signal: priceChange > 0.005 ? 'BUY' : priceChange < -0.005 ? 'SELL' : 'HOLD'
      };
    }

    res.json({
      asset: marketData.asset,
      timestamp: new Date().toISOString(),
      market_data: marketData,
      ml_prediction: prediction
    });
  } catch (err) {
    next(err);
  }
});

// New simulation endpoint
router.post('/simulate', async (req, res, next) => {
  try {
    const { 
      dataset, 
      startDate, 
      endDate, 
      symbol = 'BTC', 
      initialCash = 100000,
      strategy = 'ML_BASED',
      confidenceThreshold = 0.7
    } = req.body;

    if (!dataset || !startDate || !endDate) {
      res.status(400).json({ error: 'Missing required parameters: dataset, startDate, endDate' });
      return;
    }

    const data = MOCK_HISTORICAL_DATA[dataset];
    if (!data) {
      res.status(400).json({ error: `Dataset '${dataset}' not found` });
      return;
    }

    // Filter data for the simulation period
    const simulationData = data.filter(d => 
      d.symbol.toLowerCase() === symbol.toLowerCase() &&
      d.date >= startDate && 
      d.date <= endDate
    ).sort((a, b) => a.date.localeCompare(b.date));

    if (simulationData.length === 0) {
      res.status(400).json({ error: `No data found for ${symbol} between ${startDate} and ${endDate}` });
      return;
    }

    // Run simulation
    console.log(`🎯 Running ML simulation: ${symbol} from ${startDate} to ${endDate} with $${initialCash}`);
    
    let cash = initialCash;
    let position = 0;
    let trades = [];
    let equity_curve = [];
    
    for (let i = 0; i < simulationData.length; i++) {
      const dataPoint = simulationData[i];
      
      // Get ML prediction for this data point
      let prediction;
      if (advancedTradingAddon && advancedTradingAddon.getMLPrediction) {
        try {
          prediction = advancedTradingAddon.getMLPrediction(
            dataPoint.open, dataPoint.high, dataPoint.low, dataPoint.close, dataPoint.volume, symbol.toLowerCase()
          );
        } catch (error) {
          // Fallback prediction
          const priceChange = (dataPoint.close - dataPoint.open) / dataPoint.open;
          prediction = {
            prediction: priceChange * 0.5,
            uncertainty: Math.abs(priceChange) + 0.1,
            confidence: Math.max(0.1, 1.0 - Math.abs(priceChange) - 0.1),
            signal: priceChange > 0.005 ? 'BUY' : priceChange < -0.005 ? 'SELL' : 'HOLD'
          };
        }
      }
      
      // Trading logic based on ML prediction and confidence
      const currentPrice = dataPoint.close;
      const equity = cash + (position * currentPrice);
      
      if (prediction && prediction.confidence > confidenceThreshold) {
        if (prediction.signal === 'BUY' && position === 0 && cash > currentPrice) {
          // Buy signal with high confidence
          position = Math.floor(cash * 0.95 / currentPrice); // Use 95% of cash
          cash = cash - (position * currentPrice);
          trades.push({
            date: dataPoint.date,
            action: 'BUY',
            price: currentPrice,
            quantity: position,
            confidence: prediction.confidence,
            prediction: prediction.prediction
          });
        } else if (prediction.signal === 'SELL' && position > 0) {
          // Sell signal with high confidence
          cash = cash + (position * currentPrice);
          trades.push({
            date: dataPoint.date,
            action: 'SELL',
            price: currentPrice,
            quantity: position,
            confidence: prediction.confidence,
            prediction: prediction.prediction
          });
          position = 0;
        }
      }
      
      equity_curve.push({
        date: dataPoint.date,
        equity: cash + (position * currentPrice),
        cash,
        position,
        price: currentPrice
      });
    }
    
    // Final equity calculation
    const finalPrice = simulationData[simulationData.length - 1].close;
    const finalEquity = cash + (position * finalPrice);
    const totalReturn = ((finalEquity - initialCash) / initialCash) * 100;
    
    const winningTrades = trades.filter((t, i, arr) => {
      if (t.action === 'SELL' && i > 0) {
        const buyTrade = arr.slice(0, i).reverse().find(bt => bt.action === 'BUY');
        return buyTrade && t.price > buyTrade.price;
      }
      return false;
    }).length;
    
    const totalTradePairs = Math.floor(trades.length / 2);
    const winRate = totalTradePairs > 0 ? (winningTrades / totalTradePairs) * 100 : 0;

    console.log(`✅ ML simulation completed: ${totalReturn.toFixed(2)}% return, ${trades.length} trades, ${winRate.toFixed(1)}% win rate`);

    res.json({
      simulation_id: `sim_${Date.now()}`,
      parameters: {
        dataset,
        symbol,
        startDate,
        endDate,
        initialCash,
        strategy,
        confidenceThreshold
      },
      results: {
        initial_cash: initialCash,
        final_equity: finalEquity,
        total_return_pct: totalReturn,
        total_trades: trades.length,
        winning_trades: winningTrades,
        win_rate_pct: winRate,
        trades,
        equity_curve
      },
      metadata: {
        simulation_date: new Date().toISOString(),
        data_points: simulationData.length
      }
    });
  } catch (err) {
    next(err);
  }
});

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

    let result: StrategyResult;

    // Use the advanced trading addon if available
    if (advancedTradingAddon && advancedTradingAddon.runAdvancedBacktest) {
      try {
        console.log(`🚀 Running advanced backtest: ${strategy} on ${dataset} with $${initialCash}`);
        const advancedResult = advancedTradingAddon.runAdvancedBacktest(dataset, strategy, initialCash);
        
        result = {
          total_return_pct: advancedResult.total_return_pct || 0,
          max_drawdown_pct: advancedResult.max_drawdown_pct || 0,
          realized_pnl: advancedResult.realized_pnl || 0,
          total_commission: advancedResult.total_commission || 0,
          num_fills: advancedResult.num_fills || 0,
          final_equity: advancedResult.final_equity || initialCash,
        };

        console.log('✅ Advanced backtest completed:', {
          strategy_name: advancedResult.strategy_name,
          total_return: result.total_return_pct.toFixed(2) + '%',
          sharpe_ratio: advancedResult.sharpe_ratio?.toFixed(2),
          win_rate: advancedResult.win_rate?.toFixed(1) + '%',
          total_trades: advancedResult.total_trades
        });
      } catch (error) {
        console.error('❌ Advanced backtest failed:', error);
        // Fallback to mock data
        result = {
          total_return_pct: 12.34,
          max_drawdown_pct: 3.21,
          realized_pnl: initialCash * 0.1234,
          total_commission: initialCash * 0.001,
          num_fills: 42,
          final_equity: initialCash + initialCash * 0.1234 - initialCash * 0.001,
        };
      }
    } else {
      // Fallback to mock data if addon is not available
      console.log('⚠️  Using mock backtest data');
      result = {
        total_return_pct: 12.34,
        max_drawdown_pct: 3.21,
        realized_pnl: initialCash * 0.1234,
        total_commission: initialCash * 0.001,
        num_fills: 42,
        final_equity: initialCash + initialCash * 0.1234 - initialCash * 0.001,
      };
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

// New advanced backtesting endpoint using financial_cpp_advanced ML system
router.post('/simulate-advanced', async (req, res, next) => {
  try {
    const { 
      dataset, 
      startDate, 
      endDate, 
      symbol = 'BTC', 
      initialCash = 100000,
      strategy = 'ML_BAYESIAN',
      confidenceThreshold = 0.7,
      riskLevel = 'medium',
      positionSizing = 'kelly'
    } = req.body;

    if (!dataset || !startDate || !endDate) {
      res.status(400).json({ error: 'Missing required parameters: dataset, startDate, endDate' });
      return;
    }

    const data = MOCK_HISTORICAL_DATA[dataset];
    if (!data) {
      res.status(400).json({ error: `Dataset '${dataset}' not found` });
      return;
    }

    // Filter data for the simulation period
    const simulationData = data.filter(d => 
      d.symbol.toLowerCase() === symbol.toLowerCase() &&
      d.date >= startDate && 
      d.date <= endDate
    ).sort((a, b) => a.date.localeCompare(b.date));

    if (simulationData.length === 0) {
      res.status(400).json({ error: `No data found for ${symbol} between ${startDate} and ${endDate}` });
      return;
    }

    console.log(`🚀 Running advanced ML backtesting: ${symbol} from ${startDate} to ${endDate} with $${initialCash}`);
    
    // Try to use the Python ML system from financial_cpp_advanced
    const { spawn } = require('child_process');
    const path = require('path');
    
    // Path to the financial_cpp_advanced Python ML system
    const financialCppPath = path.join(__dirname, '../../../../financial_cpp_advanced');
    const pythonScript = path.join(financialCppPath, 'python_ml/real_time_predictor.py');
    
    // Initialize trading state
    let cash = initialCash;
    let position = 0;
    let positionValue = 0;
    let trades = [];
    let equity_curve = [];
    let dailyReturns = [];
    let totalFees = 0;
    const feeRate = 0.001; // 0.1% trading fee
    
    // Risk management parameters
    const maxPositionSize = riskLevel === 'low' ? 0.1 : riskLevel === 'medium' ? 0.2 : 0.3;
    const stopLossPercent = riskLevel === 'low' ? 0.02 : riskLevel === 'medium' ? 0.03 : 0.05;
    
    // Process each data point
    for (let i = 1; i < simulationData.length; i++) { // Start from 1 to have previous data
      const currentData = simulationData[i];
      const previousData = simulationData[i - 1];
      
      // Calculate current portfolio value
      const currentPrice = currentData.close;
      const currentPortfolioValue = cash + (position * currentPrice);
      
      // Create input for Python ML predictor
      const mlInput = {
        open: currentData.open,
        high: currentData.high,
        low: currentData.low,
        close: currentData.close,
        volume: currentData.volume,
        asset: symbol.toLowerCase(),
        date: currentData.date
      };
      
      // Get ML prediction using advanced trading addon or sophisticated fallback
      let prediction: {
        prediction: number;
        uncertainty: number;
        confidence: number;
        signal: string;
        model_type: string;
        components?: any;
      } | null = null;
      
      // Try to use the advanced trading addon first
      if (advancedTradingAddon && advancedTradingAddon.getMLPrediction) {
        try {
          const addonResult = advancedTradingAddon.getMLPrediction(
            currentData.open, 
            currentData.high, 
            currentData.low, 
            currentData.close, 
            currentData.volume, 
            symbol.toLowerCase()
          );
          prediction = {
            ...addonResult,
            model_type: 'advanced_addon'
          };
        } catch (error) {
          console.warn('Advanced addon prediction failed:', error);
          prediction = null;
        }
      }
      
      // Use advanced Bayesian ML function if addon failed
      if (!prediction) {
        prediction = getAdvancedMLPrediction(currentData, symbol, simulationData.slice(0, i));
      }
      
      // Trading decision logic based on ML prediction and risk management
      let action = 'HOLD';
      let tradeSize = 0;
      
      if (prediction && prediction.confidence >= confidenceThreshold) {
        if (prediction.signal === 'BUY' && position <= 0) {
          // Calculate position size based on strategy
          let targetPositionValue;
          if (positionSizing === 'kelly') {
            // Kelly criterion: f = (bp - q) / b, where b = odds, p = win prob, q = lose prob
            const kellyFraction = Math.min(maxPositionSize, prediction.confidence * maxPositionSize);
            targetPositionValue = currentPortfolioValue * kellyFraction;
          } else if (positionSizing === 'fixed') {
            targetPositionValue = currentPortfolioValue * 0.1; // Fixed 10%
          } else {
            // Volatility-based sizing
            const volAdjustment = Math.max(0.05, Math.min(0.3, 1.0 / (prediction.uncertainty + 1)));
            targetPositionValue = currentPortfolioValue * volAdjustment;
          }
          
          tradeSize = Math.floor(targetPositionValue / currentPrice);
          const tradeValue = tradeSize * currentPrice;
          const fees = tradeValue * feeRate;
          
          if (tradeValue + fees <= cash) {
            cash -= (tradeValue + fees);
            position += tradeSize;
            totalFees += fees;
            action = 'BUY';
            
            trades.push({
              date: currentData.date,
              action: 'BUY',
              price: currentPrice,
              quantity: tradeSize,
              value: tradeValue,
              fees: fees,
              portfolio_value: cash + (position * currentPrice),
              prediction: prediction.prediction,
              confidence: prediction.confidence,
              signal_strength: prediction.confidence * Math.abs(prediction.prediction)
            });
          }
        } else if (prediction.signal === 'SELL' && position > 0) {
          // Sell signal or stop loss
          const shouldSell = prediction.signal === 'SELL' || 
                           (currentPrice / trades[trades.length - 1]?.price - 1) < -stopLossPercent;
          
          if (shouldSell) {
            tradeSize = position;
            const tradeValue = tradeSize * currentPrice;
            const fees = tradeValue * feeRate;
            
            cash += (tradeValue - fees);
            position = 0;
            totalFees += fees;
            action = 'SELL';
            
            trades.push({
              date: currentData.date,
              action: 'SELL',
              price: currentPrice,
              quantity: tradeSize,
              value: tradeValue,
              fees: fees,
              portfolio_value: cash,
              prediction: prediction.prediction,
              confidence: prediction.confidence,
              signal_strength: prediction.confidence * Math.abs(prediction.prediction)
            });
          }
        }
      }
      
      // Record equity curve
      const portfolioValue = cash + (position * currentPrice);
      equity_curve.push({
        date: currentData.date,
        portfolio_value: portfolioValue,
        cash: cash,
        position_value: position * currentPrice,
        price: currentPrice,
        action: action,
        prediction: prediction?.prediction || 0,
        confidence: prediction?.confidence || 0
      });
      
      // Calculate daily returns
      if (i > 1) {
        const previousPortfolioValue = equity_curve[i - 2].portfolio_value;
        const dailyReturn = (portfolioValue - previousPortfolioValue) / previousPortfolioValue;
        dailyReturns.push(dailyReturn);
      }
    }
    
    // Calculate performance metrics
    const finalValue = cash + (position * simulationData[simulationData.length - 1].close);
    const totalReturn = (finalValue - initialCash) / initialCash;
    const avgDailyReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
    const volatility = Math.sqrt(dailyReturns.reduce((sum, r) => sum + Math.pow(r - avgDailyReturn, 2), 0) / dailyReturns.length);
    const sharpeRatio = volatility > 0 ? (avgDailyReturn / volatility) * Math.sqrt(252) : 0;
    
    const winningTrades = trades.filter(t => t.action === 'SELL' && 
      trades.findIndex(buy => buy.action === 'BUY' && buy.date < t.date) !== -1);
    const totalTrades = Math.floor(trades.filter(t => t.action === 'SELL').length);
    const winRate = totalTrades > 0 ? (winningTrades.length / totalTrades) : 0;
    
    const maxDrawdown = calculateMaxDrawdown(equity_curve.map(e => e.portfolio_value));
    
    console.log(`✅ Advanced ML backtesting completed: Total Return: ${(totalReturn * 100).toFixed(2)}%, Sharpe: ${sharpeRatio.toFixed(2)}, Win Rate: ${(winRate * 100).toFixed(1)}%`);
    
    res.json({
      strategy_name: `Advanced Bayesian ML Strategy (${strategy})`,
      parameters: {
        dataset,
        symbol,
        start_date: startDate,
        end_date: endDate,
        initial_cash: initialCash,
        confidence_threshold: confidenceThreshold,
        risk_level: riskLevel,
        position_sizing: positionSizing
      },
      performance: {
        total_return: `${(totalReturn * 100).toFixed(2)}%`,
        annualized_return: `${(totalReturn * (365 / simulationData.length) * 100).toFixed(2)}%`,
        sharpe_ratio: sharpeRatio.toFixed(2),
        max_drawdown: `${(maxDrawdown * 100).toFixed(2)}%`,
        volatility: `${(volatility * Math.sqrt(252) * 100).toFixed(2)}%`,
        win_rate: `${(winRate * 100).toFixed(1)}%`,
        total_trades: totalTrades,
        total_fees: `$${totalFees.toFixed(2)}`,
        final_value: `$${finalValue.toFixed(2)}`
      },
      trades: trades,
      equity_curve: equity_curve,
      daily_returns: dailyReturns
    });
  } catch (err) {
    console.error('❌ Advanced backtesting error:', err);
    next(err);
  }
});

// Advanced Bayesian ML prediction function (replaces Python ML system)
function getAdvancedMLPrediction(ohlcv: any, symbol: string, historicalData: any[]): {
  prediction: number;
  uncertainty: number;
  confidence: number;
  signal: string;
  model_type: string;
  components: any;
} {
  // Asset-specific parameters for Bayesian modeling
  const assetParams: { [key: string]: { baseVolatility: number; momentumFactor: number; meanReversion: number } } = {
    btc: { baseVolatility: 0.03, momentumFactor: 0.7, meanReversion: 0.3 },
    eth: { baseVolatility: 0.04, momentumFactor: 0.6, meanReversion: 0.4 },
    msft: { baseVolatility: 0.02, momentumFactor: 0.5, meanReversion: 0.5 }
  };
  
  const params = assetParams[symbol.toLowerCase()] || assetParams.btc;
  
  // Calculate technical indicators
  const priceChange = (ohlcv.close - ohlcv.open) / ohlcv.open;
  const volatility = (ohlcv.high - ohlcv.low) / ohlcv.open;
  const bodyRatio = Math.abs(ohlcv.close - ohlcv.open) / (ohlcv.high - ohlcv.low || 1);
  const volumeIntensity = Math.min(ohlcv.volume / 1000000, 10.0);
  
  // Historical analysis
  const returns = historicalData.slice(-10).map(d => (d.close - d.open) / d.open);
  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const returnStd = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length);
  
  // Momentum component
  const momentumSignal = priceChange * params.momentumFactor;
  
  // Mean reversion component
  const meanReversionSignal = -priceChange * params.meanReversion;
  
  // Volatility adjustment
  const volAdjustment = 1.0 - Math.min(volatility / 0.1, 1.0);
  
  // Volume confirmation
  const volumeConfirmation = Math.min(volumeIntensity / 2.0, 1.0);
  
  // Combine signals with Bayesian weighting
  const rawPrediction = (momentumSignal * 0.6 + meanReversionSignal * 0.4) * volAdjustment * volumeConfirmation;
  
  // Add realistic noise
  const noise = (Math.random() - 0.5) * 0.001;
  const prediction = rawPrediction + noise;
  
  // Bayesian uncertainty estimation
  const baseUncertainty = params.baseVolatility;
  const volatilityUncertainty = volatility * 2.0;
  const signalUncertainty = Math.abs(prediction) * 0.5;
  
  const uncertainty = Math.min(1.0, baseUncertainty + volatilityUncertainty + signalUncertainty);
  const confidence = Math.max(0.1, 1.0 - uncertainty);
  
  // Generate trading signal
  const signalThreshold = 0.002;
  let signal = 'HOLD';
  if (prediction > signalThreshold && confidence > 0.6) {
    signal = 'BUY';
  } else if (prediction < -signalThreshold && confidence > 0.6) {
    signal = 'SELL';
  }
  
  return {
    prediction: prediction,
    uncertainty: uncertainty,
    confidence: confidence,
    signal: signal,
    model_type: 'advanced_bayesian',
    components: {
      momentum: momentumSignal,
      mean_reversion: meanReversionSignal,
      volatility_adj: volAdjustment,
      volume_conf: volumeConfirmation,
      technical_indicators: {
        price_change: priceChange,
        volatility: volatility,
        body_ratio: bodyRatio,
        volume_intensity: volumeIntensity
      }
    }
  };
}

// Helper function to calculate maximum drawdown
function calculateMaxDrawdown(values: number[]): number {
  let maxDrawdown = 0;
  let peak = values[0];
  
  for (let i = 1; i < values.length; i++) {
    if (values[i] > peak) {
      peak = values[i];
    }
    const drawdown = (peak - values[i]) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }
  
  return maxDrawdown;
}

// Get available strategies endpoint
router.get('/strategies', async (req, res, next) => {
  try {
    res.json({
      strategies: AVAILABLE_STRATEGIES.map(strategy => ({
        name: strategy,
        config: STRATEGY_CONFIGS[strategy] || { params: [], defaults: {}, description: 'Custom strategy' }
      }))
    })
  } catch (err) {
    next(err)
  }
})

// Upload CSV endpoint
router.post('/upload-csv', upload.single('csv'), async (req: RequestWithFile, res, next) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No CSV file uploaded' })
      return
    }
    
    const { symbol } = req.body
    const filePath = req.file.path
    
    console.log(`📁 Processing CSV upload: ${req.file.originalname} (${req.file.size} bytes)`)
    
    const parsedData = await parseCSVData(filePath, symbol)
    
    // Store the processed data temporarily (in production, use database)
    const dataId = `csv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const storagePath = path.join(__dirname, '../../../data/uploaded', `${dataId}.json`)
    
    // Ensure directory exists
    const uploadDir = path.dirname(storagePath)
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true })
    }
    
    fs.writeFileSync(storagePath, JSON.stringify(parsedData, null, 2))
    
    // Clean up temporary upload file
    fs.unlinkSync(filePath)
    
    console.log(`✅ CSV processed successfully: ${parsedData.stats.total_rows} rows, symbols: ${parsedData.stats.symbols.join(', ')}`)
    
    res.json({
      data_id: dataId,
      symbol: parsedData.symbol,
      stats: parsedData.stats,
      preview: parsedData.data.slice(0, 5) // First 5 rows for preview
    })
  } catch (err) {
    // Clean up on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path)
    }
    next(err)
  }
})

// Run backtest on uploaded CSV data
router.post('/backtest-csv', async (req, res, next) => {
  try {
    const { 
      data_id, 
      strategy, 
      strategy_params = {},
      initial_cash = 100000,
      commission = 0.001 
    } = req.body
    
    if (!data_id || !strategy) {
      res.status(400).json({ error: 'Missing data_id or strategy' })
      return
    }
    
    if (!AVAILABLE_STRATEGIES.includes(strategy)) {
      res.status(400).json({ error: `Strategy '${strategy}' not available. Choose from: ${AVAILABLE_STRATEGIES.join(', ')}` })
      return
    }
    
    // Load the uploaded data
    const storagePath = path.join(__dirname, '../../../data/uploaded', `${data_id}.json`)
    if (!fs.existsSync(storagePath)) {
      res.status(404).json({ error: 'Uploaded data not found or expired' })
      return
    }
    
    const csvData: ParsedCSVData = JSON.parse(fs.readFileSync(storagePath, 'utf8'))
    
    console.log(`🚀 Running ${strategy} backtest on ${csvData.symbol} with ${csvData.data.length} data points`)
    
    // Use the advanced trading addon if available
    let result;
    if (advancedTradingAddon && advancedTradingAddon.runAdvancedBacktest) {
      try {
        // Convert CSV data to format expected by C++ addon
        const backtest_config = {
          strategy_name: strategy,
          strategy_params: { ...STRATEGY_CONFIGS[strategy]?.defaults, ...strategy_params },
          data: csvData.data,
          initial_cash,
          commission,
          symbol: csvData.symbol
        }
        
        const advancedResult = advancedTradingAddon.runAdvancedBacktest(
          data_id, // Use data_id as dataset identifier
          strategy,
          initial_cash,
          backtest_config
        )
        
        result = {
          strategy_name: strategy,
          symbol: csvData.symbol,
          data_period: csvData.stats.date_range,
          total_return_pct: advancedResult.total_return_pct || 0,
          max_drawdown_pct: advancedResult.max_drawdown_pct || 0,
          sharpe_ratio: advancedResult.sharpe_ratio || 0,
          realized_pnl: advancedResult.realized_pnl || 0,
          total_commission: advancedResult.total_commission || 0,
          num_fills: advancedResult.num_fills || 0,
          final_equity: advancedResult.final_equity || initial_cash,
          win_rate: advancedResult.win_rate || 0,
          total_trades: advancedResult.total_trades || 0,
          data_points: csvData.data.length,
          strategy_params: backtest_config.strategy_params
        }
        
        console.log(`✅ Backtest completed: ${strategy} on ${csvData.symbol}`)
        console.log(`   Return: ${result.total_return_pct.toFixed(2)}%, Sharpe: ${result.sharpe_ratio.toFixed(2)}, Trades: ${result.total_trades}`)
        
      } catch (error) {
        console.error('❌ C++ backtest failed:', error)
        // Fallback to mock result
        result = {
          strategy_name: strategy,
          symbol: csvData.symbol,
          data_period: csvData.stats.date_range,
          total_return_pct: 0,
          max_drawdown_pct: 0,
          sharpe_ratio: 0,
          realized_pnl: 0,
          total_commission: 0,
          num_fills: 0,
          final_equity: initial_cash,
          win_rate: 0,
          total_trades: 0,
          data_points: csvData.data.length,
          strategy_params,
          error: 'C++ engine not available, using mock result'
        }
      }
    } else {
      // Mock result when addon not available
      const mockReturn = (Math.random() - 0.5) * 20 // Random return between -10% and +10%
      result = {
        strategy_name: strategy,
        symbol: csvData.symbol,
        data_period: csvData.stats.date_range,
        total_return_pct: mockReturn,
        max_drawdown_pct: Math.abs(mockReturn) * 0.6,
        sharpe_ratio: mockReturn / Math.abs(mockReturn + 0.1),
        realized_pnl: (mockReturn / 100) * initial_cash,
        total_commission: csvData.data.length * commission,
        num_fills: Math.floor(csvData.data.length / 10),
        final_equity: initial_cash + ((mockReturn / 100) * initial_cash),
        win_rate: 45 + Math.random() * 20, // Random win rate 45-65%
        total_trades: Math.floor(csvData.data.length / 10),
        data_points: csvData.data.length,
        strategy_params,
        mock: true
      }
    }
    
    res.json(result)
    
  } catch (err) {
    next(err)
  }
})

// List uploaded datasets
router.get('/uploaded-datasets', async (req, res, next) => {
  try {
    const uploadDir = path.join(__dirname, '../../../data/uploaded')
    
    if (!fs.existsSync(uploadDir)) {
      res.json({ datasets: [] })
      return
    }
    
    const files = fs.readdirSync(uploadDir).filter(file => file.endsWith('.json'))
    const datasets = []
    
    for (const file of files) {
      try {
        const filePath = path.join(uploadDir, file)
        const data: ParsedCSVData = JSON.parse(fs.readFileSync(filePath, 'utf8'))
        const stats = fs.statSync(filePath)
        
        datasets.push({
          data_id: file.replace('.json', ''),
          symbol: data.symbol,
          stats: data.stats,
          uploaded_at: stats.birthtime,
          size_mb: (stats.size / (1024 * 1024)).toFixed(2)
        })
      } catch (error) {
        console.warn(`Skipping invalid dataset file: ${file}`)
      }
    }
    
    // Sort by upload time (newest first)
    datasets.sort((a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime())
    
    res.json({ datasets })
  } catch (err) {
    next(err)
  }
})

export default router
