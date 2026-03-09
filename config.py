"""
Configuration for the ML-powered Market Prediction Bot.
"""

# ============================================================
# CAPITAL & COSTS
# ============================================================
INITIAL_CAPITAL = 100_000   # ₹1,00,000
COMMISSION_PCT = 0.03       # 0.03% brokerage (Zerodha-like)
SLIPPAGE_PCT = 0.01         # 0.01% slippage

# ============================================================
# MODEL HYPERPARAMETERS (v4 — Enhanced Accuracy)
# ============================================================
LOOKBACK = 60               # Increased from 30 — research shows 60-day window captures better patterns
EPOCHS = 150                # More epochs for larger dataset — early stopping will handle overfitting
BATCH_SIZE = 32
LSTM_UNITS_1 = 128          # Increased from 64 — more capacity
LSTM_UNITS_2 = 64           # Increased from 32
LSTM_UNITS_3 = 32           # NEW — third LSTM layer
DROPOUT = 0.3               # Increased from 0.2 — better regularization
LEARNING_RATE = 0.0005      # Decreased from 0.001 — more stable training
TRAIN_SPLIT = 0.8
HISTORY_PERIOD = '15y'          # 15 years — optimal balance of data volume and market relevance

# VMD (Variational Mode Decomposition) — noise reduction
VMD_MODES = 4               # Number of decomposition modes (3-5 recommended)
VMD_ALPHA = 2000            # Quadratic penalty factor
VMD_TAU = 0                 # Noise tolerance (0 = no noise)

# Attention mechanism
USE_ATTENTION = True        # Self-attention layer in LSTM

# Feature selection
FEATURE_IMPORTANCE_THRESHOLD = 0.005  # Drop features below this importance

# News sentiment
USE_SENTIMENT = True        # FinBERT news sentiment integration
SENTIMENT_LOOKBACK = 5      # Days of news to aggregate

# XGBoost tuning
XGB_N_ESTIMATORS = 500      # Increased from 200
XGB_MAX_DEPTH = 8           # Increased from 6
XGB_LEARNING_RATE = 0.02    # Decreased from 0.05 — more trees, smaller steps
XGB_MIN_CHILD_WEIGHT = 3    # Regularization
XGB_GAMMA = 0.1             # Pruning threshold
XGB_REG_ALPHA = 0.1         # L1 regularization
XGB_REG_LAMBDA = 1.0        # L2 regularization

# Ensemble weights (learned from validation)
LSTM_WEIGHT = 0.4           # Was 0.5 — LSTM slightly less reliable
XGB_WEIGHT = 0.35           # Was 0.5
SENTIMENT_WEIGHT = 0.25     # NEW — sentiment component

# ============================================================
# GLOBAL MARKETS
# ============================================================
US_MARKETS = {
    'S&P 500': '^GSPC', 'NASDAQ': '^IXIC',
    'Dow Jones': '^DJI', 'Russell 2000': '^RUT',
}
EUROPE_MARKETS = {
    'FTSE 100': '^FTSE', 'DAX (Germany)': '^GDAXI',
    'CAC 40 (France)': '^FCHI',
}
ASIA_MARKETS = {
    'Nikkei 225': '^N225', 'TOPIX (Japan)': '^TOPX',
    'Hang Seng': '^HSI', 'Hang Seng Tech': '^HSTECH',
    'Shanghai Composite': '000001.SS', 'Shenzhen Component': '399001.SZ',
    'CSI 300 (China)': '000300.SS',
    'KOSPI': '^KS11',
}
CHINA_JAPAN_DETAIL = {
    # Japan major stocks
    'Toyota': '7203.T', 'Sony': '6758.T', 'SoftBank': '9984.T',
    'Keyence': '6861.T', 'Nintendo': '7974.T',
    # China major stocks (HK-listed)
    'Alibaba': '9988.HK', 'Tencent': '0700.HK', 'BYD': '1211.HK',
    'Meituan': '3690.HK', 'JD.com': '9618.HK', 'PetroChina': '0857.HK',
}
COMMODITIES_FX = {
    'Crude Oil (WTI)': 'CL=F', 'Brent Crude': 'BZ=F',
    'Gold': 'GC=F', 'Silver': 'SI=F', 'Copper': 'HG=F',
    'Natural Gas': 'NG=F', 'USD/INR': 'INR=X',
    'US 10Y Yield': '^TNX', 'Dollar Index': 'DX-Y.NYB',
}
VOLATILITY = {'India VIX': '^INDIAVIX', 'US VIX': '^VIX'}

# ============================================================
# INDIAN SECTORS
# ============================================================
SECTORS = {
    'Banking & Financials': {
        'color': '#3b82f6',
        'stocks': {'HDFCBANK': 'HDFCBANK.NS', 'ICICIBANK': 'ICICIBANK.NS',
                   'KOTAKBANK': 'KOTAKBANK.NS', 'SBIN': 'SBIN.NS', 'BAJFINANCE': 'BAJFINANCE.NS'},
        'index': {'BANKNIFTY': '^NSEBANK'},
        'global_drivers': {'US 10Y Yield': +0.7, 'S&P 500': +0.6, 'Dollar Index': -0.5,
                           'USD/INR': -0.6, 'Gold': -0.2},
        'description': 'Driven by FII flows, US yields, and rupee strength',
    },
    'IT & Technology': {
        'color': '#8b5cf6',
        'stocks': {'TCS': 'TCS.NS', 'INFY': 'INFY.NS', 'WIPRO': 'WIPRO.NS',
                   'HCLTECH': 'HCLTECH.NS', 'TECHM': 'TECHM.NS'},
        'index': {'NIFTY IT': '^CNXIT'},
        'global_drivers': {'NASDAQ': +0.85, 'S&P 500': +0.5, 'USD/INR': +0.6,
                           'Dollar Index': +0.5, 'US 10Y Yield': -0.3},
        'description': 'Directly tracks NASDAQ; benefits from weak rupee (dollar earners)',
    },
    'Oil Upstream (Producers)': {
        'color': '#f59e0b',
        'stocks': {'ONGC': 'ONGC.NS', 'OIL INDIA': 'OIL.NS', 'RELIANCE': 'RELIANCE.NS'},
        'index': {},
        'global_drivers': {'Crude Oil (WTI)': +0.75, 'Brent Crude': +0.75,
                           'Natural Gas': +0.4, 'S&P 500': +0.2, 'USD/INR': +0.3},
        'description': 'DIRECT BENEFICIARIES of crude spike — ONGC, Oil India rally when crude rises.',
    },
    'Oil Downstream (OMCs)': {
        'color': '#dc2626',
        'stocks': {'BPCL': 'BPCL.NS', 'IOC': 'IOC.NS', 'HINDPETRO': 'HINDPETRO.NS'},
        'index': {},
        'global_drivers': {'Crude Oil (WTI)': -0.8, 'Brent Crude': -0.8,
                           'Natural Gas': -0.3, 'S&P 500': +0.2, 'USD/INR': -0.6},
        'description': 'WORST HIT by crude spike — buy crude at high prices, can\'t pass on costs.',
    },
    'Metals & Mining': {
        'color': '#64748b',
        'stocks': {'TATASTEEL': 'TATASTEEL.NS', 'HINDALCO': 'HINDALCO.NS',
                   'JSWSTEEL': 'JSWSTEEL.NS', 'COALINDIA': 'COALINDIA.NS'},
        'index': {'NIFTY METAL': '^CNXMETAL'},
        'global_drivers': {'Copper': +0.8, 'Gold': +0.3, 'Silver': +0.5,
                           'Shanghai': +0.6, 'Hang Seng': +0.4, 'S&P 500': +0.3},
        'description': 'Tracks copper & China demand; Gold/Silver for precious metals',
    },
    'Pharma & Healthcare': {
        'color': '#22c55e',
        'stocks': {'SUNPHARMA': 'SUNPHARMA.NS', 'DRREDDY': 'DRREDDY.NS',
                   'CIPLA': 'CIPLA.NS', 'DIVISLAB': 'DIVISLAB.NS'},
        'index': {'NIFTY PHARMA': '^CNXPHARMA'},
        'global_drivers': {'USD/INR': +0.5, 'NASDAQ': +0.2, 'S&P 500': +0.2,
                           'Gold': +0.3, 'Crude Oil (WTI)': -0.1},
        'description': 'Defensive sector; benefits from weak rupee; mild global correlation',
    },
    'Auto & EV': {
        'color': '#ec4899',
        'stocks': {'TATAMOTORS': 'TATAMOTORS.BO', 'M&M': 'M&M.NS',
                   'MARUTI': 'MARUTI.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS'},
        'index': {'NIFTY AUTO': '^CNXAUTO'},
        'global_drivers': {'Crude Oil (WTI)': -0.5, 'S&P 500': +0.3, 'Copper': +0.3,
                           'US 10Y Yield': -0.2, 'USD/INR': -0.3},
        'description': 'Hurt by crude spikes and rupee weakness; linked to consumer sentiment',
    },
    'FMCG & Consumer': {
        'color': '#06b6d4',
        'stocks': {'HINDUNILVR': 'HINDUNILVR.NS', 'ITC': 'ITC.NS',
                   'NESTLEIND': 'NESTLEIND.NS', 'BRITANNIA': 'BRITANNIA.NS'},
        'index': {'NIFTY FMCG': '^CNXFMCG'},
        'global_drivers': {'Gold': +0.3, 'Crude Oil (WTI)': -0.3, 'S&P 500': +0.15,
                           'USD/INR': -0.2, 'US VIX': +0.3},
        'description': 'Defensive safe haven; benefits when markets are fearful (high VIX)',
    },
}

BROAD_MARKET = {'NIFTY 50': '^NSEI', 'BANKNIFTY': '^NSEBANK'}

# Output paths (relative to project root)
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, 'reports')
MODEL_DIR = os.path.join(_BASE_DIR, 'models')
DATA_DIR = os.path.join(_BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
