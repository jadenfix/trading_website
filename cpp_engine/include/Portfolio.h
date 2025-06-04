#ifndef PORTFOLIO_H
#define PORTFOLIO_H
#include <string>

struct StrategyResult {
    double total_return_pct;
    double max_drawdown_pct;
    double realized_pnl;
    double total_commission;
    int    num_fills;
    double final_equity;
};

class Portfolio {
public:
    explicit Portfolio(double initialCash = 0.0);
    StrategyResult get_results_summary() const;
    static double runBacktest(const std::string& symbol, int lookbackDays);
private:
    double cash;
};

#endif // PORTFOLIO_H
