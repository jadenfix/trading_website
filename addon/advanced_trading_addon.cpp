#include <napi.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>

// Import the ML Bridge from the advanced financial C++ repo
// We'll need to copy the relevant files or reference them
class MLBridge {
public:
    MLBridge(
        const std::string& pythonPath = "python3", 
        const std::string& scriptPath = "",
        const std::string& modelDir = "",
        const std::string& asset = "btc",
        bool useMock = true
    ) : pythonPath_(pythonPath), 
        scriptPath_(scriptPath),
        modelDir_(modelDir),
        asset_(asset),
        useMock_(useMock) {}
    
    std::pair<double, double> getPredictionOHLCV(
        double open, double high, double low, double close, double volume) {
        
        // Create market data map
        std::map<std::string, double> marketData;
        marketData[asset_ + "_open"] = open;
        marketData[asset_ + "_high"] = high;
        marketData[asset_ + "_low"] = low;
        marketData[asset_ + "_close"] = close;
        marketData[asset_ + "_volume"] = volume;
        
        // For now, return mock values - will implement full ML bridge later
        if (useMock_) {
            // Simulate some trading logic
            double price_change = (close - open) / open;
            double volatility = (high - low) / close;
            
            // Mock prediction based on simple momentum
            double prediction = price_change * 0.5;
            double uncertainty = volatility * 2.0;
            
            return {prediction, uncertainty};
        }
        
        // TODO: Implement real ML prediction
        return {0.0, 1.0};
    }

private:
    std::string pythonPath_;
    std::string scriptPath_;
    std::string modelDir_;
    std::string asset_;
    bool useMock_;
};

// Trading Strategy Results
struct BacktestResult {
    double total_return_pct;
    double max_drawdown_pct;
    double realized_pnl;
    double total_commission;
    int num_fills;
    double final_equity;
    double sharpe_ratio;
    double win_rate;
    int total_trades;
    std::string strategy_name;
};

// Simple backtesting engine
class TradingBacktester {
public:
    TradingBacktester(const std::string& asset, double initialCash) 
        : asset_(asset), initialCash_(initialCash), cash_(initialCash) {
        
        // Initialize ML Bridge with mock model for now
        mlBridge_ = std::make_unique<MLBridge>(
            "python3", "", "", asset, true
        );
    }
    
    BacktestResult runBacktest(const std::vector<std::map<std::string, double>>& marketData) {
        BacktestResult result = {};
        result.strategy_name = "Bayesian ML Strategy";
        
        double maxEquity = initialCash_;
        double minEquity = initialCash_;
        double totalPnl = 0.0;
        int trades = 0;
        int winningTrades = 0;
        
        bool inPosition = false;
        double entryPrice = 0.0;
        bool isLong = false;
        
        for (const auto& bar : marketData) {
            if (bar.find("close") == bar.end()) continue;
            
            double open = bar.at("open");
            double high = bar.at("high");
            double low = bar.at("low");
            double close = bar.at("close");
            double volume = bar.count("volume") ? bar.at("volume") : 1000.0;
            
            // Get ML prediction
            auto [prediction, uncertainty] = mlBridge_->getPredictionOHLCV(
                open, high, low, close, volume);
            
            // Trading logic
            if (!inPosition) {
                // Enter position based on prediction
                if (prediction > 0.002 && uncertainty < 0.5) {
                    // Go long
                    inPosition = true;
                    isLong = true;
                    entryPrice = close;
                } else if (prediction < -0.002 && uncertainty < 0.5) {
                    // Go short
                    inPosition = true;
                    isLong = false;
                    entryPrice = close;
                }
            } else {
                // Exit logic
                bool shouldExit = false;
                
                if (isLong && (prediction < -0.001 || close < entryPrice * 0.98)) {
                    shouldExit = true;
                } else if (!isLong && (prediction > 0.001 || close > entryPrice * 1.02)) {
                    shouldExit = true;
                }
                
                if (shouldExit) {
                    double pnl;
                    if (isLong) {
                        pnl = (close - entryPrice) / entryPrice * initialCash_ * 0.1;
                    } else {
                        pnl = (entryPrice - close) / entryPrice * initialCash_ * 0.1;
                    }
                    
                    totalPnl += pnl;
                    trades++;
                    if (pnl > 0) winningTrades++;
                    
                    cash_ += pnl;
                    maxEquity = std::max(maxEquity, cash_);
                    minEquity = std::min(minEquity, cash_);
                    
                    inPosition = false;
                }
            }
        }
        
        // Calculate metrics
        result.total_return_pct = ((cash_ - initialCash_) / initialCash_) * 100.0;
        result.max_drawdown_pct = ((maxEquity - minEquity) / maxEquity) * 100.0;
        result.realized_pnl = totalPnl;
        result.total_commission = trades * 2.0; // $2 per trade
        result.num_fills = trades * 2; // Entry + exit
        result.final_equity = cash_;
        result.sharpe_ratio = result.total_return_pct / std::max(1.0, result.max_drawdown_pct);
        result.win_rate = trades > 0 ? (double(winningTrades) / trades) * 100.0 : 0.0;
        result.total_trades = trades;
        
        return result;
    }

private:
    std::string asset_;
    double initialCash_;
    double cash_;
    std::unique_ptr<MLBridge> mlBridge_;
};

// Node.js wrapper for advanced backtesting
Napi::Object RunAdvancedBacktest(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    try {
        // Parse parameters
        std::string dataset = info[0].As<Napi::String>().Utf8Value();
        std::string strategy = info[1].As<Napi::String>().Utf8Value();
        double cash = info[2].As<Napi::Number>().DoubleValue();
        
        // Generate mock market data for now
        std::vector<std::map<std::string, double>> marketData;
        
        // Generate 100 bars of mock data
        double basePrice = 50000.0; // Starting price
        for (int i = 0; i < 100; i++) {
            std::map<std::string, double> bar;
            
            // Add some realistic market movement
            double change = (rand() % 200 - 100) / 10000.0; // -1% to +1%
            basePrice *= (1.0 + change);
            
            bar["open"] = basePrice * 0.999;
            bar["high"] = basePrice * 1.005;
            bar["low"] = basePrice * 0.995;
            bar["close"] = basePrice;
            bar["volume"] = 1000000 + (rand() % 500000);
            
            marketData.push_back(bar);
        }
        
        // Determine asset from dataset
        std::string asset = "btc";
        if (dataset.find("eth") != std::string::npos) asset = "eth";
        if (dataset.find("sol") != std::string::npos) asset = "sol";
        
        // Run backtest
        TradingBacktester backtester(asset, cash);
        BacktestResult result = backtester.runBacktest(marketData);
        
        // Create result object
        Napi::Object output = Napi::Object::New(env);
        output.Set("strategy_on_dataset", Napi::String::New(env, strategy + "@" + dataset));
        output.Set("total_return_pct", Napi::Number::New(env, result.total_return_pct));
        output.Set("max_drawdown_pct", Napi::Number::New(env, result.max_drawdown_pct));
        output.Set("realized_pnl", Napi::Number::New(env, result.realized_pnl));
        output.Set("total_commission", Napi::Number::New(env, result.total_commission));
        output.Set("num_fills", Napi::Number::New(env, result.num_fills));
        output.Set("final_equity", Napi::Number::New(env, result.final_equity));
        output.Set("sharpe_ratio", Napi::Number::New(env, result.sharpe_ratio));
        output.Set("win_rate", Napi::Number::New(env, result.win_rate));
        output.Set("total_trades", Napi::Number::New(env, result.total_trades));
        output.Set("strategy_name", Napi::String::New(env, result.strategy_name));
        
        return output;
        
    } catch (const std::exception& e) {
        Napi::TypeError::New(env, "Backtest failed: " + std::string(e.what()))
            .ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }
}

// Get ML prediction for current market data
Napi::Object GetMLPrediction(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    try {
        // Parse market data
        double open = info[0].As<Napi::Number>().DoubleValue();
        double high = info[1].As<Napi::Number>().DoubleValue();
        double low = info[2].As<Napi::Number>().DoubleValue();
        double close = info[3].As<Napi::Number>().DoubleValue();
        double volume = info[4].As<Napi::Number>().DoubleValue();
        std::string asset = info[5].As<Napi::String>().Utf8Value();
        
        // Create ML Bridge
        MLBridge bridge("python3", "", "", asset, true);
        
        // Get prediction
        auto [prediction, uncertainty] = bridge.getPredictionOHLCV(
            open, high, low, close, volume);
        
        // Return prediction object
        Napi::Object result = Napi::Object::New(env);
        result.Set("prediction", Napi::Number::New(env, prediction));
        result.Set("uncertainty", Napi::Number::New(env, uncertainty));
        result.Set("confidence", Napi::Number::New(env, 1.0 - uncertainty));
        result.Set("signal", Napi::String::New(env, 
            prediction > 0.002 ? "BUY" : 
            prediction < -0.002 ? "SELL" : "HOLD"));
        
        return result;
        
    } catch (const std::exception& e) {
        Napi::TypeError::New(env, "Prediction failed: " + std::string(e.what()))
            .ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }
}

// Legacy compatibility
Napi::Number RunBacktest(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    // std::string symbol = info[0].As<Napi::String>().Utf8Value();
    // int lookbackDays = info[1].As<Napi::Number>().Int32Value();
    
    // Simple mock return for legacy compatibility
    return Napi::Number::New(env, 12.34);
}

Napi::Object GetPortfolioResults(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    double initialCash = info[0].As<Napi::Number>().DoubleValue();
    
    // Mock portfolio results
    Napi::Object out = Napi::Object::New(env);
    out.Set("total_return_pct", Napi::Number::New(env, 15.25));
    out.Set("max_drawdown_pct", Napi::Number::New(env, 4.32));
    out.Set("realized_pnl", Napi::Number::New(env, initialCash * 0.1525));
    out.Set("total_commission", Napi::Number::New(env, initialCash * 0.002));
    out.Set("num_fills", Napi::Number::New(env, 47));
    out.Set("final_equity", Napi::Number::New(env, initialCash * 1.1525));
    
    return out;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("runAdvancedBacktest", Napi::Function::New(env, RunAdvancedBacktest));
    exports.Set("getMLPrediction", Napi::Function::New(env, GetMLPrediction));
    exports.Set("runBacktest", Napi::Function::New(env, RunBacktest));
    exports.Set("getPortfolioResults", Napi::Function::New(env, GetPortfolioResults));
    return exports;
}

NODE_API_MODULE(advancedTradingAddon, Init) 