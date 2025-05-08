#include <napi.h>
#include "../cpp_engine/src/Portfolio.h"

// Wrap Portfolio::get_results_summary() → JS object
Napi::Object GetPortfolioResults(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // 1) Read our parameters from JS (e.g. initial cash)
  double initialCash = info[0].As<Napi::Number>().DoubleValue();

  // 2) Build a Portfolio and—**in a real use case**—feed it market events here.
  Portfolio p(initialCash);
  // TODO: in production, you’d call p.handle_fill_event(...) etc. from data.

  // 3) Get the results struct
  StrategyResult res = p.get_results_summary();

  // 4) Construct a JS object to return
  Napi::Object out = Napi::Object::New(env);
  out.Set("total_return_pct",   Napi::Number::New(env, res.total_return_pct));
  out.Set("max_drawdown_pct",   Napi::Number::New(env, res.max_drawdown_pct));
  out.Set("realized_pnl",       Napi::Number::New(env, res.realized_pnl));
  out.Set("total_commission",   Napi::Number::New(env, res.total_commission));
  out.Set("num_fills",          Napi::Number::New(env, res.num_fills));
  out.Set("final_equity",       Napi::Number::New(env, res.final_equity));
  return out;
}

// Example of a simpler helper that just returns the ending equity for symbol/date window
Napi::Number RunBacktest(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string symbol = info[0].As<Napi::String>().Utf8Value();
  int lookbackDays = info[1].As<Napi::Number>().Int32Value();

  // Construct and run your backtest in C++
  double result = Portfolio::runBacktest(symbol, lookbackDays);
  return Napi::Number::New(env, result);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("runBacktest",
              Napi::Function::New(env, RunBacktest));
  exports.Set("getPortfolioResults",
              Napi::Function::New(env, GetPortfolioResults));
  return exports;
}

NODE_API_MODULE(financialAddon, Init)