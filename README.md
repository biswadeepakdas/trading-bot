# ML-Powered Indian Market Prediction Bot

An ensemble ML trading bot that predicts next-day direction and magnitude for Indian stock market sectors using LSTM neural networks and XGBoost, combined with global market cues.

## Features

- **LSTM + XGBoost Ensemble** — Self-attention LSTM with VMD denoising + regularized XGBoost, adaptively weighted
- **33 Trained Models** — Covers NIFTY 50, BANKNIFTY, and 31 NSE stocks across 8 sectors
- **Global Market Integration** — Correlates 25+ global indices, commodities, forex, and volatility indicators
- **Real-Time Dashboard** — Professional fintech-grade HTML dashboard with live price updates via TradingView
- **~70% Average Accuracy** — Validated on historical data with 15-year training window

## Architecture

```
run_prediction.py        # Main entry point — orchestrates everything
config.py                # Sectors, global markets, hyperparameters
features.py              # Data fetching (TradingView) + 80+ technical indicators
lstm_model.py            # AttentionLSTM + XGBoost + adaptive ensemble
backtester.py            # Strategy backtesting engine
live_server.py           # HTTP server for real-time dashboard updates
models/                  # Pre-trained model weights (.keras, .pkl)
reports/                 # Generated HTML dashboards
```

## Sectors Covered

| Sector | Stocks |
|--------|--------|
| Banking & Financials | HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, BAJFINANCE |
| IT & Technology | TCS, INFY, WIPRO, HCLTECH, TECHM |
| Oil Upstream | ONGC, OIL INDIA, RELIANCE |
| Oil Downstream (OMCs) | BPCL, IOC, HINDPETRO |
| Metals & Mining | TATASTEEL, HINDALCO, JSWSTEEL, COALINDIA |
| Pharma & Healthcare | SUNPHARMA, DRREDDY, CIPLA, DIVISLAB |
| Auto & EV | M&M, MARUTI, BAJAJ-AUTO |
| FMCG & Consumer | HINDUNILVR, ITC, NESTLEIND, BRITANNIA |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run daily prediction (uses pre-trained models)
python run_prediction.py

# Retrain all models (takes ~30 min)
python run_prediction.py --train

# Start live dashboard with real-time price updates
python live_server.py
# Open http://localhost:8080
```

## Model Details

- **Data**: 15 years of daily OHLCV from TradingView via tvdatafeed
- **Features**: 80+ technical indicators (RSI, MACD, Bollinger, ADX, OBV, etc.) + VMD denoised price
- **LSTM**: 3-layer (128→64→32) with self-attention, RobustScaler, dropout 0.3
- **XGBoost**: 500 estimators, max_depth 8, L1/L2 regularization
- **Ensemble**: Adaptive weights based on validation performance, class-weight balancing

## Dashboard

The dashboard shows:
- Market mood gauge (bullish/bearish/neutral)
- India VIX and US VIX fear indicators
- NIFTY 50 and BANKNIFTY live prices
- Global markets: US, Europe, Asia, Commodities, Forex
- China & Japan major stocks
- Sector prediction heatmap with expected magnitude
- Deep-dive cards with global drivers and per-stock ML predictions

When `live_server.py` is running, prices auto-update every 30 seconds.

## Disclaimer

This is an educational project. Models are trained on historical data and cannot predict black-swan events. Always do your own research and consult a SEBI-registered advisor before making investment decisions.
