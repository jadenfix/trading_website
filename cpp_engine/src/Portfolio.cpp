#include "../include/Portfolio.h"

Portfolio::Portfolio(double initialCash) : cash(initialCash) {}

StrategyResult Portfolio::get_results_summary() const {
    StrategyResult res{};
    res.total_return_pct = 0.0;
    res.max_drawdown_pct = 0.0;
    res.realized_pnl = 0.0;
    res.total_commission = 0.0;
    res.num_fills = 0;
    res.final_equity = cash;
    return res;
}

double Portfolio::runBacktest(const std::string& /*symbol*/, int lookbackDays) {
    return static_cast<double>(lookbackDays);
}
