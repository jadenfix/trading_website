import { Router } from 'express'
import { StrategyResult } from '../models/portfolio'
import path from 'path'

const router = Router()

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

// New endpoint for ML predictions
router.post('/predict', async (req, res, next) => {
  try {
    const { open, high, low, close, volume, asset } = req.body;

    if (typeof open !== 'number' || typeof high !== 'number' || 
        typeof low !== 'number' || typeof close !== 'number' || 
        typeof volume !== 'number' || !asset) {
      res.status(400).json({ error: 'Invalid market data: expected numeric OHLCV and asset' });
      return;
    }

    let prediction;

    // Use the advanced trading addon if available
    if (advancedTradingAddon && advancedTradingAddon.getMLPrediction) {
      try {
        console.log(`🔮 Getting ML prediction for ${asset}: O:${open} H:${high} L:${low} C:${close} V:${volume}`);
        prediction = advancedTradingAddon.getMLPrediction(open, high, low, close, volume, asset);
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
      const priceChange = (close - open) / open;
      prediction = {
        prediction: priceChange * 0.5,
        uncertainty: Math.abs(priceChange) + 0.1,
        confidence: Math.max(0.1, 1.0 - Math.abs(priceChange) - 0.1),
        signal: priceChange > 0.005 ? 'BUY' : priceChange < -0.005 ? 'SELL' : 'HOLD'
      };
    }

    res.json({
      asset,
      timestamp: new Date().toISOString(),
      market_data: { open, high, low, close, volume },
      ml_prediction: prediction
    });
  } catch (err) {
    next(err);
  }
});

export default router
