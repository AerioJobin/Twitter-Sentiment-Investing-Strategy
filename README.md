# Twitter Sentiment Investing Strategy

A quantitative trading strategy that exploits relative Twitter/X social sentiment to generate alpha. The core hypothesis: stocks with rising, high-engagement sentiment momentum relative to their peers tend to outperform over the next month.

## 🎯 The Core Idea

This strategy bets on **relative sentiment**, not absolute levels. We don't care if the whole market is bullish or bearish — we only care which stocks are getting more positive buzz than their peers right now.

The key insight: Twitter sentiment can act as a leading indicator for price movements, especially when combined with engagement metrics and cross-sectional normalization.

## 📊 Strategy Overview

### Step 1: Raw Data Collection
For each stock, daily metrics include:
- Tweet count and volume
- Positive vs negative tweet ratio  
- Average sentiment score (-1 to +1)
- Engagement metrics (likes/retweets)
- Influencer-weighted sentiment scores

*Note: In this notebook, data is simulated. In production, use APIs from StockTwits, Quiverquant, or similar providers.*

### Step 2: Feature Engineering

Raw daily sentiment is too noisy to trade directly. We construct robust derived features:

- **Rolling Means** (3-day, 10-day, 21-day): Smooth out noise and identify trends
- **Sentiment Momentum**: Difference between short-term and medium-term averages — captures acceleration, not just level
- **Sentiment Z-Score**: Measures how surprising today's sentiment is relative to the stock's own history
- **Volume-Weighted Sentiment**: Gives more weight to high-activity days
- **Engagement-Weighted Sentiment**: Amplifies viral, high-engagement content

### Step 3: Composite Score Construction

All features are combined into a single score using fixed weights:
- Sentiment Momentum: 20%
- Volume-Weighted Sentiment: 25%  
- Engagement-Weighted Sentiment: 20%
- Z-Score: 15%
- Other features: 20%

**Critical step**: Before combining, each feature is **cross-sectionally normalized** — converted to standard deviations above/below the monthly average across all stocks. This removes market-wide sentiment bias and makes stocks directly comparable.

### Step 4: Cross-Sectional Ranking

Each month, all ~40 stocks are ranked 1 to 40 by composite score:
- Rank 1 = Highest positive sentiment momentum
- Rank 40 = Lowest/most negative momentum  

The ranking drives portfolio construction — absolute scores don't matter, only relative position.

### Step 5: Signal Validation (IC Analysis)

Before building portfolios, we validate the signal using **Information Coefficient (IC)**:

For each month, compute the Spearman rank correlation between:
- Each stock's composite score rank
- Its actual return the following month

**Key metrics**:
- **Mean IC > 0.05**: Considered good in professional quant finance
- **ICIR (IC ÷ Std Dev)**: Measures signal consistency — want stable, not random

If IC hovers around zero, the signal has no predictive power.

### Step 6: Quintile Analysis  

Secondary validation: Split stocks into 5 buckets (quintiles) by score each month.

Measure whether average returns monotonically increase from Q1 (lowest) to Q5 (highest).

If Q3 outperforms Q5, something is broken.

### Step 7: Portfolio Construction

**Long-Only Portfolio**:  
- Each month-end: Select top 10 ranked stocks
- Hold equally weighted for next month
- Rebalance monthly

**Long-Short Portfolio** (optional):  
- Long top 10 stocks
- Short bottom 10 stocks  
- Market-neutral exposure

**Transaction Costs**: Deduct 10 basis points (0.1%) per position changed — realistic slippage + commissions. This penalizes high turnover strategies.

### Step 8: Machine Learning Enhancement

Final layer: **Ridge Regression** trained on rolling 24-month window.

**Features**:  
- All sentiment features from Step 2
- Price-based features: 1M, 3M, 6M returns, volatility
- **Sentiment-Price Divergence**: Novel feature capturing when sentiment rises but price hasn't moved yet — exploits sentiment as a leading indicator

**Validation**: Compare ML model IC vs raw composite score IC to measure improvement.

## 🔬 Critical Validation Steps

The **most important** outputs to check:

1. **IC Analysis**: Does the signal actually predict returns?
2. **Quintile Chart**: Do returns increase monotonically from Q1 to Q5?  
3. **IC Decay**: How long does the signal persist? (Monthly vs weekly rebalancing)

If these don't show clear patterns, no amount of clever backtesting will save the strategy.

## 📈 Performance Metrics

The notebook computes:
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Alpha vs Benchmark**: Excess return over market (QQQ/SPY)
- **Turnover**: Monthly portfolio churn rate
- **Win Rate**: Percentage of profitable months

## 🛠️ Pipeline Summary

```
Raw Sentiment Data
    ↓
Feature Engineering (momentum, z-scores, engagement)
    ↓
Cross-Sectional Normalization  
    ↓
Composite Score Ranking
    ↓
IC Validation (Does it work?)
    ↓
Portfolio Construction (Top 10 long, Bottom 10 short)
    ↓
Backtest with Transaction Costs
    ↓
ML Enhancement (Ridge + Divergence Feature)
    ↓  
Final Performance Report
```

## 📦 Requirements

```python
pip install pandas numpy matplotlib yfinance scikit-learn vaderSentiment textblob pypfopt
```

## 🚀 Usage

1. Open the notebook in Google Colab or Jupyter
2. Run all cells sequentially  
3. Review IC analysis and quintile charts first
4. Examine backtest performance vs benchmark
5. Compare ML model vs baseline composite score

## ⚠️ Important Notes

- **Data is simulated**: Replace with real Twitter API data for production
- **Transaction costs matter**: The 10bps assumption is conservative but realistic
- **Overfitting risk**: ML model uses rolling window to avoid look-ahead bias
- **Market regime**: Strategy may underperform in low-volatility, momentum-driven markets

## 📚 Key Concepts

- **Cross-Sectional Normalization**: Critical for removing market-wide bias
- **Information Coefficient (IC)**: Industry-standard signal validation metric  
- **Sentiment-Price Divergence**: Novel alpha source — sentiment leads price
- **Spearman Rank Correlation**: Robust to outliers, measures monotonic relationship

## 🔮 Future Enhancements

- [ ] Incorporate options market data (implied volatility)
- [ ] Add sector neutrality constraints  
- [ ] Dynamic position sizing based on signal confidence
- [ ] Short-term (weekly) rebalancing analysis
- [ ] Ensemble ML models (XGBoost, LightGBM)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss proposed changes.

---

**Disclaimer**: This is a research project for educational purposes. Past performance does not guarantee future results. Always conduct your own due diligence before trading.
