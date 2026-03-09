"""
Feature Engineering Pipeline v4 — Enhanced Accuracy
=====================================================
Improvements over v3:
  1. VMD (Variational Mode Decomposition) for noise reduction
  2. FinBERT news sentiment analysis
  3. Multi-timeframe features (5, 10, 20, 60 day windows)
  4. Cross-asset correlation features
  5. Candlestick pattern recognition
  6. Enhanced volume profile features
  7. Regime detection features (bull/bear/sideways)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')

from config import VMD_MODES, VMD_ALPHA, VMD_TAU, USE_SENTIMENT


# ============================================================
# VMD DENOISING
# ============================================================
def vmd_denoise(series, n_modes=VMD_MODES, alpha=VMD_ALPHA, tau=VMD_TAU):
    """
    Apply Variational Mode Decomposition to denoise a price series.
    Removes the highest-frequency mode (noise) and reconstructs.
    Research shows VMD-LSTM reduces RMSE by ~70% vs raw LSTM.
    """
    try:
        from vmdpy import VMD
        signal = series.dropna().values
        if len(signal) < 50:
            return series

        u, u_hat, omega = VMD(signal, alpha, tau, n_modes, 0, 1, 1e-7)

        # Sort modes by center frequency (ascending)
        freq_order = np.argsort([np.mean(np.abs(np.diff(mode))) for mode in u])

        # Reconstruct WITHOUT the highest-frequency mode (noise)
        denoised = np.sum(u[freq_order[:-1]], axis=0)

        result = series.copy()
        result.iloc[result.notna()] = denoised[:result.notna().sum()]
        return result
    except Exception:
        return series  # Fallback to raw series


# ============================================================
# NEWS SENTIMENT (FinBERT)
# ============================================================
_sentiment_pipeline = None

def get_sentiment_pipeline():
    """Lazy-load FinBERT sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None and USE_SENTIMENT:
        try:
            from transformers import pipeline as hf_pipeline
            _sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )
        except Exception as e:
            print(f"    FinBERT load failed: {e}, continuing without sentiment")
            _sentiment_pipeline = None
    return _sentiment_pipeline


def fetch_news_sentiment(stock_name, days=5):
    """
    Fetch recent news for a stock and compute FinBERT sentiment score.
    Returns: float between -1 (very bearish) and +1 (very bullish)
    """
    pipe = get_sentiment_pipeline()
    if pipe is None:
        return 0.0

    try:
        ticker = yf.Ticker(stock_name)
        news = ticker.news
        if not news:
            return 0.0

        headlines = []
        for item in news[:15]:  # Last 15 articles
            title = item.get('title', '')
            if title:
                headlines.append(title)

        if not headlines:
            return 0.0

        results = pipe(headlines)
        score = 0.0
        for r in results:
            if r['label'] == 'positive':
                score += r['score']
            elif r['label'] == 'negative':
                score -= r['score']
            # neutral contributes 0

        return round(score / len(results), 4)
    except Exception:
        return 0.0


# ============================================================
# DATA FETCHING
# ============================================================
def _period_to_bars(period):
    """Convert yfinance-style period string to number of daily bars."""
    period = period.lower().strip()
    if period == 'max':
        return 5000  # tvdatafeed max
    num = int(''.join(c for c in period if c.isdigit()) or '5')
    if 'y' in period:
        return min(num * 252, 5000)  # ~252 trading days/year, max 5000
    elif 'mo' in period:
        return min(num * 21, 5000)
    elif 'd' in period:
        return min(num, 5000)
    return min(num * 252, 5000)


# ============================================================
# TVDATAFEED SYMBOL MAPPING (replaces yfinance for ALL markets)
# ============================================================
# Master mapping: yfinance symbol → (tvdatafeed_symbol, exchange)
_TV_SYMBOL_MAP = {
    # --- Indian Indices ---
    '^NSEI':        ('NIFTY', 'NSE'),
    '^NSEBANK':     ('BANKNIFTY', 'NSE'),
    '^CNXIT':       ('CNXIT', 'NSE'),
    '^CNXMETAL':    ('CNXMETAL', 'NSE'),
    '^CNXPHARMA':   ('CNXPHARMA', 'NSE'),
    '^CNXAUTO':     ('CNXAUTO', 'NSE'),
    '^CNXFMCG':     ('CNXFMCG', 'NSE'),
    '^INDIAVIX':    ('INDIAVIX', 'NSE'),
    # --- US Markets ---
    '^GSPC':        ('SPX', 'SP'),
    '^IXIC':        ('IXIC', 'NASDAQ'),
    '^DJI':         ('DJI', 'DJ'),
    '^RUT':         ('RUT', 'TVC'),
    # --- Europe ---
    '^FTSE':        ('UKX', 'TVC'),
    '^GDAXI':       ('DEU40', 'TVC'),
    '^FCHI':        ('CAC40', 'TVC'),
    # --- Asia ---
    '^N225':        ('NI225', 'TVC'),
    '^TOPX':        ('TOPIX', 'TSE'),
    '^HSI':         ('HSI', 'TVC'),
    '^HSTECH':      ('HSTECH', 'TVC'),
    '000001.SS':    ('000001', 'SSE'),
    '399001.SZ':    ('399001', 'SZSE'),
    '000300.SS':    ('000300', 'SSE'),
    '^KS11':        ('KOSPI', 'TVC'),
    # --- Japan stocks ---
    '7203.T':       ('7203', 'TSE'),   # Toyota
    '6758.T':       ('6758', 'TSE'),   # Sony
    '9984.T':       ('9984', 'TSE'),   # SoftBank
    '6861.T':       ('6861', 'TSE'),   # Keyence
    '7974.T':       ('7974', 'TSE'),   # Nintendo
    # --- China/HK stocks ---
    '9988.HK':      ('9988', 'HKEX'),  # Alibaba
    '0700.HK':      ('0700', 'HKEX'),  # Tencent
    '1211.HK':      ('1211', 'HKEX'),  # BYD
    '3690.HK':      ('3690', 'HKEX'),  # Meituan
    '9618.HK':      ('9618', 'HKEX'),  # JD.com
    '0857.HK':      ('0857', 'HKEX'),  # PetroChina
    # --- Commodities ---
    'CL=F':         ('USOIL', 'TVC'),      # WTI Crude
    'BZ=F':         ('UKOIL', 'TVC'),      # Brent Crude
    'GC=F':         ('GOLD', 'TVC'),       # Gold
    'SI=F':         ('SILVER', 'TVC'),     # Silver
    'HG=F':         ('HG1!', 'COMEX'),     # Copper
    'NG=F':         ('NG1!', 'NYMEX'),     # Natural Gas
    # --- Forex & Rates ---
    'INR=X':        ('USDINR', 'FX_IDC'),  # USD/INR
    '^TNX':         ('TNX', 'TVC'),        # US 10Y Yield
    'DX-Y.NYB':    ('DXY', 'TVC'),        # Dollar Index
    # --- Volatility ---
    '^VIX':         ('VIX', 'TVC'),        # US VIX
}

# Singleton TvDatafeed instance (reuse connection)
_tv_instance = None

def _get_tv():
    """Get or create singleton TvDatafeed instance."""
    global _tv_instance
    if _tv_instance is None:
        import logging
        logging.getLogger('tvDatafeed').setLevel(logging.ERROR)
        from tvDatafeed import TvDatafeed
        _tv_instance = TvDatafeed()
    return _tv_instance


def _yf_to_tv_symbol(yf_symbol):
    """
    Convert yfinance symbol to tvdatafeed (symbol, exchange) tuple.
    Covers ALL markets: Indian stocks/indices, global indices, commodities, forex.
    """
    # Check master mapping first
    if yf_symbol in _TV_SYMBOL_MAP:
        return _TV_SYMBOL_MAP[yf_symbol]

    # NSE stocks: SYMBOL.NS -> (SYMBOL, NSE)
    # tvdatafeed uses underscores for special chars (BAJAJ_AUTO, M_M)
    if yf_symbol.endswith('.NS'):
        tv_sym = yf_symbol.replace('.NS', '').replace('-', '_').replace('&', '_')
        return (tv_sym, 'NSE')

    # BSE stocks: SYMBOL.BO -> (SYMBOL, BSE)
    if yf_symbol.endswith('.BO'):
        tv_sym = yf_symbol.replace('.BO', '').replace('-', '_').replace('&', '_')
        return (tv_sym, 'BSE')

    return None  # Unknown symbol


def _tv_fetch(symbol, exchange, n_bars=10, retries=2):
    """Fetch data from tvdatafeed with retry on connection drops."""
    global _tv_instance
    from tvDatafeed import Interval
    import time

    for attempt in range(retries + 1):
        try:
            tv = _get_tv()
            data = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars
            )
            if data is not None and len(data) > 0:
                df = data[['open', 'high', 'low', 'close', 'volume']].copy()
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index.name = 'Date'
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df.dropna()
            elif attempt < retries:
                # Connection might have dropped — reset and retry
                _tv_instance = None
                time.sleep(1)
        except Exception:
            if attempt < retries:
                _tv_instance = None  # Force reconnect
                time.sleep(1)
    return None


def fetch_data(symbol, period='5y', interval='1d'):
    """
    Fetch historical OHLCV data using tvdatafeed for ALL markets.
    tvdatafeed uses TradingView websockets — no conflict with TensorFlow.
    Falls back to yfinance ONLY if tvdatafeed mapping is unknown.
    """
    tv_info = _yf_to_tv_symbol(symbol)
    if tv_info is not None:
        n_bars = _period_to_bars(period)
        df = _tv_fetch(tv_info[0], tv_info[1], n_bars)
        if df is not None and len(df) >= 60:
            print(f"    [tvdatafeed] {symbol} → {tv_info[0]}:{tv_info[1]} — {len(df)} bars")
            return df
        else:
            print(f"    [tvdatafeed] {symbol} ({tv_info[0]}:{tv_info[1]}): insufficient data, trying yfinance...")

    # Fallback to yfinance only for unmapped symbols
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty or len(df) < 60:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception as e:
        print(f"    Error fetching {symbol}: {e}")
        return None


def fetch_global_snapshot():
    """Fetch latest change % for all global instruments using tvdatafeed."""
    from config import US_MARKETS, EUROPE_MARKETS, ASIA_MARKETS, COMMODITIES_FX, VOLATILITY
    all_tickers = {}
    for group in [US_MARKETS, EUROPE_MARKETS, ASIA_MARKETS, COMMODITIES_FX, VOLATILITY]:
        all_tickers.update(group)

    result = {}
    for name, yf_sym in all_tickers.items():
        tv_info = _yf_to_tv_symbol(yf_sym)
        if tv_info is not None:
            df = _tv_fetch(tv_info[0], tv_info[1], n_bars=10)
            if df is not None and len(df) >= 2:
                close = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                result[name] = {
                    'close': round(float(close), 2),
                    'change_pct': round(((close - prev) / prev) * 100, 2),
                }
                continue

        # Fallback to yfinance for unmapped symbols
        try:
            h = yf.Ticker(yf_sym).history(period='5d', interval='1d')
            if len(h) >= 2:
                close = h['Close'].iloc[-1]
                prev = h['Close'].iloc[-2]
                result[name] = {
                    'close': round(float(close), 2),
                    'change_pct': round(((close - prev) / prev) * 100, 2),
                }
        except:
            pass
    return result


# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
def build_ta_features(df, apply_vmd=True):
    """
    Build 70+ technical indicator features with VMD denoising.
    """
    feat = pd.DataFrame(index=df.index)

    # --- VMD DENOISED CLOSE ---
    if apply_vmd:
        close_denoised = vmd_denoise(df['Close'])
        feat['close_denoised'] = close_denoised
        feat['close_noise'] = df['Close'] - close_denoised  # Noise component
        feat['noise_ratio'] = feat['close_noise'].abs() / df['Close'] * 100
    else:
        close_denoised = df['Close']

    # --- TREND INDICATORS ---
    # EMA family (on both raw and denoised)
    for window in [9, 21, 50, 200]:
        feat[f'ema_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
        if apply_vmd:
            feat[f'ema_{window}_dn'] = ta.trend.ema_indicator(close_denoised, window=window)

    # EMA crossover signals
    feat['ema_9_21_cross'] = (feat['ema_9'] - feat['ema_21']) / df['Close'] * 100
    feat['ema_9_50_cross'] = (feat['ema_9'] - feat['ema_50']) / df['Close'] * 100
    feat['ema_21_50_cross'] = (feat['ema_21'] - feat['ema_50']) / df['Close'] * 100
    feat['price_vs_ema50'] = (df['Close'] - feat['ema_50']) / df['Close'] * 100
    feat['price_vs_ema200'] = (df['Close'] - feat['ema_200']) / df['Close'] * 100

    # MACD
    macd = ta.trend.MACD(df['Close'])
    feat['macd'] = macd.macd()
    feat['macd_signal'] = macd.macd_signal()
    feat['macd_histogram'] = macd.macd_diff()
    # MACD trend (histogram slope)
    feat['macd_hist_slope'] = feat['macd_histogram'].diff(3)

    # ADX (trend strength)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    feat['adx'] = adx.adx()
    feat['adx_pos'] = adx.adx_pos()
    feat['adx_neg'] = adx.adx_neg()
    feat['adx_diff'] = feat['adx_pos'] - feat['adx_neg']  # DI+ minus DI-

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(df['High'], df['Low'])
    feat['ichi_a'] = ichi.ichimoku_a()
    feat['ichi_b'] = ichi.ichimoku_b()
    feat['ichi_diff'] = (feat['ichi_a'] - feat['ichi_b']) / df['Close'] * 100

    # SMA (Simple Moving Averages — different from EMA)
    feat['sma_10'] = df['Close'].rolling(10).mean()
    feat['sma_30'] = df['Close'].rolling(30).mean()
    feat['sma_10_30_cross'] = (feat['sma_10'] - feat['sma_30']) / df['Close'] * 100

    # --- MOMENTUM INDICATORS ---
    # RSI + multi-period
    feat['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    feat['rsi_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()
    feat['rsi_21'] = ta.momentum.RSIIndicator(df['Close'], window=21).rsi()
    feat['rsi_slope'] = feat['rsi'].diff(5)  # RSI momentum

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    feat['stoch_k'] = stoch.stoch()
    feat['stoch_d'] = stoch.stoch_signal()
    feat['stoch_diff'] = feat['stoch_k'] - feat['stoch_d']

    # Williams %R
    feat['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

    # ROC (Rate of Change) — multi-period
    for period in [3, 5, 10, 20]:
        feat[f'roc_{period}'] = ta.momentum.ROCIndicator(df['Close'], window=period).roc()

    # Awesome Oscillator
    feat['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()

    # CCI (Commodity Channel Index)
    feat['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()

    # --- VOLATILITY INDICATORS ---
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    feat['bb_upper'] = bb.bollinger_hband()
    feat['bb_lower'] = bb.bollinger_lband()
    feat['bb_width'] = bb.bollinger_wband()
    feat['bb_pct'] = bb.bollinger_pband()

    # ATR + multi-period
    feat['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    feat['atr_pct'] = feat['atr'] / df['Close'] * 100
    feat['atr_ratio'] = feat['atr'] / feat['atr'].rolling(20).mean()  # ATR expansion/contraction

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
    feat['kc_upper'] = kc.keltner_channel_hband()
    feat['kc_lower'] = kc.keltner_channel_lband()
    feat['kc_width'] = (feat['kc_upper'] - feat['kc_lower']) / df['Close'] * 100

    # --- VOLUME INDICATORS ---
    feat['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    feat['obv_change'] = feat['obv'].pct_change()
    feat['obv_slope'] = feat['obv'].diff(5) / feat['obv'].abs().clip(lower=1) * 100

    feat['vwap'] = ta.volume.VolumeWeightedAveragePrice(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).volume_weighted_average_price()
    feat['price_vs_vwap'] = (df['Close'] - feat['vwap']) / df['Close'] * 100

    feat['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    feat['volume_ratio_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
    feat['volume_spike'] = (feat['volume_ratio'] > 2).astype(int)

    feat['mfi'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()

    # Force Index
    feat['force_index'] = ta.volume.ForceIndexIndicator(df['Close'], df['Volume']).force_index()

    # --- CANDLESTICK PATTERNS ---
    feat['body_size'] = abs(df['Close'] - df['Open']) / df['Open'] * 100
    feat['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'] * 100
    feat['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close'] * 100
    feat['is_bullish_candle'] = (df['Close'] > df['Open']).astype(int)
    feat['bullish_streak'] = feat['is_bullish_candle'].rolling(5).sum()  # Bullish days in last 5

    # Doji detection (small body)
    feat['is_doji'] = (feat['body_size'] < 0.1).astype(int)

    # --- PRICE-DERIVED (Multi-timeframe) ---
    for period in [1, 3, 5, 10, 20, 60]:
        feat[f'return_{period}d'] = df['Close'].pct_change(period) * 100

    feat['high_low_range'] = (df['High'] - df['Low']) / df['Close'] * 100
    feat['close_open_range'] = (df['Close'] - df['Open']) / df['Open'] * 100

    # Volatility (rolling std of returns)
    feat['volatility_5'] = feat['return_1d'].rolling(5).std()
    feat['volatility_10'] = feat['return_1d'].rolling(10).std()
    feat['volatility_20'] = feat['return_1d'].rolling(20).std()
    feat['volatility_ratio'] = feat['volatility_5'] / feat['volatility_20'].clip(lower=0.01)

    # --- REGIME DETECTION ---
    # Trend regime: price above/below long-term EMA
    feat['regime_trend'] = np.where(df['Close'] > feat['ema_200'], 1, -1)
    feat['regime_momentum'] = np.where(feat['rsi'] > 50, 1, -1)
    feat['regime_volatility'] = np.where(feat['bb_width'] > feat['bb_width'].rolling(50).mean(), 1, 0)

    # Market phase (combination)
    feat['market_phase'] = feat['regime_trend'] + feat['regime_momentum']

    # Support/Resistance proximity
    feat['near_high_20'] = (df['Close'] / df['High'].rolling(20).max()) * 100
    feat['near_low_20'] = (df['Close'] / df['Low'].rolling(20).min()) * 100
    feat['range_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                              (df['High'].rolling(20).max() - df['Low'].rolling(20).min() + 0.001) * 100

    # --- GAP FEATURES (important for overnight effects) ---
    feat['gap_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    feat['gap_fill'] = np.where(
        (feat['gap_pct'] > 0) & (df['Low'] <= df['Close'].shift(1)), 1,
        np.where((feat['gap_pct'] < 0) & (df['High'] >= df['Close'].shift(1)), -1, 0)
    )

    # --- STATISTICAL FEATURES ---
    feat['skewness_20'] = feat['return_1d'].rolling(20).skew()
    feat['kurtosis_20'] = feat['return_1d'].rolling(20).kurt()
    feat['zscore_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()

    # --- DAY OF WEEK (cyclical encoding) ---
    feat['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    feat['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # --- MONTH (cyclical encoding) ---
    feat['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    return feat


# ============================================================
# TARGET CREATION
# ============================================================
def create_targets(df, horizon=1):
    """
    Create prediction targets:
    - direction: 1 (up) or 0 (down)
    - magnitude: % change next day
    """
    future_return = df['Close'].pct_change(horizon).shift(-horizon) * 100
    direction = (future_return > 0).astype(int)
    return direction, future_return


# ============================================================
# ML DATASET PIPELINE
# ============================================================
def prepare_ml_dataset(symbol, period='15y'):
    """
    Full pipeline: fetch data -> VMD denoise -> build features -> create targets.
    Uses period='15y' by default — optimal balance of data volume and relevance.
    Returns X (features), y_dir (direction), y_mag (magnitude), dates.
    """
    from config import HISTORY_PERIOD
    use_period = period if period != '15y' else HISTORY_PERIOD
    df = fetch_data(symbol, period=use_period)
    if df is None:
        return None, None, None, None, None

    features = build_ta_features(df, apply_vmd=True)
    y_dir, y_mag = create_targets(df)

    # Merge
    dataset = features.copy()
    dataset['target_dir'] = y_dir
    dataset['target_mag'] = y_mag

    # Drop NaN rows (from indicator warmup + future shift)
    dataset.dropna(inplace=True)

    X = dataset.drop(columns=['target_dir', 'target_mag'])
    y_direction = dataset['target_dir']
    y_magnitude = dataset['target_mag']

    return X, y_direction, y_magnitude, df, features
