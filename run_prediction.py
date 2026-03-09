#!/usr/bin/env python3
"""
MAIN ENTRY POINT — ML-Powered Market Prediction Bot v3
========================================================
Orchestrates: Data Fetch → Feature Engineering → ML Prediction → Global Cues → HTML Report

Usage:
  python run_prediction.py                  # Daily prediction (loads pre-trained models)
  python run_prediction.py --train          # Train/retrain models first, then predict
  python run_prediction.py --backtest       # Run backtester on all strategies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (SECTORS, BROAD_MARKET, US_MARKETS, EUROPE_MARKETS, ASIA_MARKETS,
                    COMMODITIES_FX, VOLATILITY, OUTPUT_DIR, MODEL_DIR, LOOKBACK,
                    CHINA_JAPAN_DETAIL)
from features import fetch_data, fetch_global_snapshot, build_ta_features, create_targets, prepare_ml_dataset
from lstm_model import MarketPredictor


def train_all_models():
    """Train LSTM+XGBoost models for each sector's lead stock."""
    print("\n" + "=" * 65)
    print("  TRAINING ML MODELS")
    print("=" * 65)

    all_metrics = {}
    stocks_to_train = {}

    # Collect unique stocks across all sectors
    for sector_name, sector_config in SECTORS.items():
        for stock_name, symbol in sector_config['stocks'].items():
            if stock_name not in stocks_to_train:
                stocks_to_train[stock_name] = symbol

    # Also add broad market
    for name, sym in BROAD_MARKET.items():
        stocks_to_train[name] = sym

    for stock_name, symbol in stocks_to_train.items():
        print(f"\n  Training: {stock_name} ({symbol})")

        X, y_dir, y_mag, df, features = prepare_ml_dataset(symbol, period='15y')
        if X is None or len(X) < 100:
            print(f"    SKIP — insufficient data for {stock_name}")
            continue

        print(f"    Data: {len(X)} samples, {len(X.columns)} features")

        model = MarketPredictor(name=stock_name)
        metrics = model.train(X, y_dir, y_mag)
        model.save()

        all_metrics[stock_name] = metrics

    # Save overall metrics summary
    with open(os.path.join(MODEL_DIR, 'training_summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE — Model Accuracy Summary")
    print("=" * 65)
    print(f"  {'Stock':<16} {'LSTM Acc':>10} {'XGB Acc':>10} {'Ensemble':>10} {'MAE':>8}")
    print("  " + "-" * 56)
    for name, m in all_metrics.items():
        print(f"  {name:<16} {m['lstm']['accuracy']:>9.1%} {m['xgboost']['accuracy']:>9.1%} "
              f"{m['ensemble']['accuracy']:>9.1%} {m['ensemble']['mae']:>7.3f}%")

    return all_metrics


def run_predictions():
    """Run predictions using trained models + global cues."""
    print("\n" + "=" * 65)
    print("  ML-POWERED MARKET PREDICTION")
    print(f"  {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}")
    print("=" * 65)

    # 1. Fetch global cues
    print("\n[1/5] Fetching Global Markets...")
    global_data = fetch_global_snapshot()
    for name, d in global_data.items():
        arrow = '▲' if d['change_pct'] >= 0 else '▼'
        print(f"  {name:<20}: {d['close']:>12,.2f} {arrow} {d['change_pct']:+.2f}%")

    # 1b. Fetch China & Japan detail stocks (using tvdatafeed)
    print("\n[2/5] Fetching China & Japan Major Stocks...")
    china_japan_data = {}
    from features import _yf_to_tv_symbol, _tv_fetch
    for name, sym in CHINA_JAPAN_DETAIL.items():
        try:
            tv_info = _yf_to_tv_symbol(sym)
            df = None
            if tv_info:
                df = _tv_fetch(tv_info[0], tv_info[1], n_bars=10)
            if df is not None and len(df) >= 2:
                close = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2])
                china_japan_data[name] = {
                    'close': round(close, 2),
                    'change_pct': round(((close - prev) / prev) * 100, 2),
                    'symbol': sym,
                }
                arrow = '▲' if china_japan_data[name]['change_pct'] >= 0 else '▼'
                print(f"  {name:<20}: {china_japan_data[name]['close']:>12,.2f} {arrow} {china_japan_data[name]['change_pct']:+.2f}%")
        except:
            pass

    # 3. Broad market data (using tvdatafeed)
    print("\n[3/5] Fetching Indian Broad Market...")
    broad_data = {}
    for name, sym in BROAD_MARKET.items():
        try:
            tv_info = _yf_to_tv_symbol(sym)
            df = None
            if tv_info:
                df = _tv_fetch(tv_info[0], tv_info[1], n_bars=10)
            if df is not None and len(df) >= 2:
                broad_data[name] = {
                    'close': round(float(df['Close'].iloc[-1]), 2),
                    'change_pct': round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2),
                }
                print(f"  {name}: {broad_data[name]['close']:,.2f} ({broad_data[name]['change_pct']:+.2f}%)")
        except:
            pass

    # 3. ML predictions for each stock
    print("\n[4/5] Running ML Predictions...")
    ml_predictions = {}

    for sector_name, sector_config in SECTORS.items():
        print(f"\n  --- {sector_name} ---")
        sector_ml = {}

        for stock_name, symbol in sector_config['stocks'].items():
            model = MarketPredictor(name=stock_name)
            loaded = model.load()

            if not loaded:
                print(f"    {stock_name}: No trained model found, skipping ML")
                sector_ml[stock_name] = None
                continue

            # Get latest features
            df = fetch_data(symbol, period='6mo')
            if df is None or len(df) < LOOKBACK + 10:
                print(f"    {stock_name}: Insufficient data")
                sector_ml[stock_name] = None
                continue

            features = build_ta_features(df)
            features.ffill(inplace=True)
            features.bfill(inplace=True)
            features.dropna(axis=1, how='all', inplace=True)

            # NOTE: feature alignment is now handled inside model.predict()
            # which aligns to feature_names, scales, then selects features

            if len(features) < LOOKBACK:
                print(f"    {stock_name}: Not enough feature data")
                sector_ml[stock_name] = None
                continue

            # Predict
            pred = model.predict(features)
            pred['close'] = round(float(df['Close'].iloc[-1]), 2)
            pred['prev_close'] = round(float(df['Close'].iloc[-2]), 2)
            pred['change_pct'] = round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2)
            pred['metrics'] = model.metrics

            sector_ml[stock_name] = pred

            icon = '🟢' if pred['direction'] == 'UP' else '🔴'
            print(f"    {icon} {stock_name:<14}: {pred['direction']} ({pred['probability']:.1%}) | "
                  f"Expected: {pred['magnitude_pct']:+.2f}% | Confidence: {pred['confidence']:.0f}%")

        ml_predictions[sector_name] = sector_ml

    # 4. Combine: Global Cues + ML → Sector Prediction
    print("\n[5/5] Computing Sector-Level Predictions...")
    sector_results = {}

    for sector_name, sector_config in SECTORS.items():
        # Global cue score (same as v2)
        drivers = sector_config['global_drivers']
        global_score = 0
        total_weight = 0
        driver_details = []

        for gname, corr in drivers.items():
            d = global_data.get(gname)
            if not d: continue
            capped = max(min(d['change_pct'], 3.0), -3.0)
            impact = capped * corr
            global_score += impact
            total_weight += abs(corr)
            driver_details.append({
                'driver': gname, 'change': d['change_pct'], 'correlation': corr,
                'impact': round(impact, 2),
                'direction': 'BULLISH' if impact > 0 else ('BEARISH' if impact < 0 else 'NEUTRAL'),
            })

        global_normalized = (global_score / total_weight * 3) if total_weight > 0 else 0
        global_normalized = max(min(global_normalized, 5), -5)

        # ML score (average of stock predictions)
        ml_scores = []
        ml_mags = []
        stock_preds = ml_predictions.get(sector_name, {})
        for stock_name, pred in stock_preds.items():
            if pred is None: continue
            # Convert probability to directional score: 0.5 = neutral, 1 = strong buy, 0 = strong sell
            ml_score = (pred['probability'] - 0.5) * 10  # Scale to -5 to +5
            ml_scores.append(ml_score)
            ml_mags.append(pred['magnitude_pct'])

        avg_ml_score = np.mean(ml_scores) if ml_scores else 0
        avg_ml_mag = np.mean(ml_mags) if ml_mags else 0

        # Ensemble: 40% Global + 60% ML (ML weighted higher since it's data-driven)
        composite = 0.4 * global_normalized + 0.6 * avg_ml_score

        # Determine prediction
        if composite > 2: prediction, action = 'STRONG BUY', 'BUY'
        elif composite > 0.8: prediction, action = 'BULLISH', 'BUY'
        elif composite > 0.2: prediction, action = 'MILDLY BULLISH', 'BUY (cautious)'
        elif composite < -2: prediction, action = 'STRONG SELL', 'SELL'
        elif composite < -0.8: prediction, action = 'BEARISH', 'SELL'
        elif composite < -0.2: prediction, action = 'MILDLY BEARISH', 'SELL (cautious)'
        else: prediction, action = 'NEUTRAL', 'WAIT'

        confidence = min(abs(composite) / 5 * 100, 95)

        # Average model accuracy for this sector
        model_accuracies = []
        for sname, pred in stock_preds.items():
            if pred and 'metrics' in pred:
                model_accuracies.append(pred['metrics'].get('ensemble', {}).get('accuracy', 0))
        avg_model_acc = np.mean(model_accuracies) if model_accuracies else 0

        sector_results[sector_name] = {
            'sector': sector_name,
            'prediction': prediction,
            'action': action,
            'composite_score': round(composite, 2),
            'global_score': round(global_normalized, 2),
            'ml_score': round(avg_ml_score, 2),
            'expected_magnitude': round(avg_ml_mag, 3),
            'confidence': round(confidence, 0),
            'model_accuracy': round(avg_model_acc * 100, 1),
            'global_details': sorted(driver_details, key=lambda x: abs(x['impact']), reverse=True),
            'stock_predictions': stock_preds,
            'color': sector_config['color'],
            'description': sector_config['description'],
        }

    # Print summary
    print("\n" + "=" * 65)
    print("  SECTOR-WISE PREDICTIONS (ML + Global Cues)")
    print("=" * 65)

    sorted_sectors = sorted(sector_results.values(), key=lambda x: x['composite_score'], reverse=True)
    for s in sorted_sectors:
        icon = '🟢' if 'BUY' in s['action'] else ('🔴' if 'SELL' in s['action'] else '🟡')
        mag = s['expected_magnitude']
        mag_str = f"{mag:+.2f}%" if mag != 0 else "N/A"
        print(f"  {icon} {s['sector']:<28} {s['action']:<18} Score: {s['composite_score']:>+6.2f} "
              f"| ML: {s['ml_score']:>+5.2f} | Global: {s['global_score']:>+5.2f} "
              f"| Exp: {mag_str} | Acc: {s['model_accuracy']:.0f}%")

    # Generate HTML report
    generate_html_report(sector_results, global_data, broad_data, china_japan_data)

    return sector_results


def generate_html_report(sectors_data, global_data, broad_data, china_japan_data=None):
    """Generate professional industry-standard HTML dashboard."""
    if china_japan_data is None:
        china_japan_data = {}

    now = datetime.now()
    report_date = now.strftime('%A, %B %d, %Y')
    report_time = now.strftime('%I:%M %p')

    all_composites = [s['composite_score'] for s in sectors_data.values()]
    avg_composite = np.mean(all_composites)
    bullish = sum(1 for s in sectors_data.values() if s['composite_score'] > 0.2)
    bearish = sum(1 for s in sectors_data.values() if s['composite_score'] < -0.2)
    neutral = len(sectors_data) - bullish - bearish

    if avg_composite > 1: mood, mood_color = 'BULLISH', '#0e7c6b'
    elif avg_composite > 0: mood, mood_color = 'MILDLY BULLISH', '#34b89a'
    elif avg_composite < -1: mood, mood_color = 'BEARISH', '#dc2646'
    elif avg_composite < 0: mood, mood_color = 'MILDLY BEARISH', '#e85d6f'
    else: mood, mood_color = 'NEUTRAL', '#b45309'

    # VIX data
    vix = global_data.get('India VIX', {})
    vix_val = vix.get('close', 0)
    vix_chg = vix.get('change_pct', 0)
    vix_status = 'EXTREME FEAR' if vix_val > 25 else ('HIGH FEAR' if vix_val > 20 else ('ELEVATED' if vix_val > 15 else 'CALM'))
    vix_color = '#dc2646' if vix_val > 20 else ('#b45309' if vix_val > 15 else '#0e7c6b')
    us_vix_val = global_data.get('US VIX', dict(close=0)).get('close', 0)
    us_vix_chg = global_data.get('US VIX', dict(change_pct=0)).get('change_pct', 0)

    # Gauge rotation helper: maps -5..+5 composite to -90..+90 degrees
    gauge_deg = max(min(avg_composite / 5 * 90, 90), -90)

    # SVG Icons (inline for no external dependencies)
    svg_refresh = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>'
    svg_activity = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'
    svg_trending_up = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
    svg_trending_down = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline><polyline points="17 18 23 18 23 12"></polyline></svg>'
    svg_bar_chart = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>'
    svg_globe = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>'
    svg_shield = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
    svg_alert = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    svg_cpu = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>'

    def arrow_icon(change):
        return svg_trending_up if change >= 0 else svg_trending_down

    def val_color(change):
        return '#0e7c6b' if change >= 0 else '#dc2646'

    # ---------- World Map: position markets geographically ----------
    map_pins = {
        'S&P 500': (18, 42), 'NASDAQ': (18, 48), 'Dow Jones': (18, 54), 'Russell 2000': (18, 60),
        'FTSE 100': (46, 32), 'DAX (Germany)': (50, 35), 'CAC 40 (France)': (47, 40),
        'Nikkei 225': (86, 38), 'TOPIX (Japan)': (86, 44),
        'Hang Seng': (80, 48), 'Hang Seng Tech': (80, 54),
        'Shanghai Composite': (77, 40), 'Shenzhen Component': (77, 46), 'CSI 300 (China)': (77, 52),
        'KOSPI': (83, 36),
        'NIFTY 50': (70, 52), 'BANKNIFTY': (70, 58),
        'Crude Oil (WTI)': (22, 70), 'Brent Crude': (22, 76), 'Gold': (28, 70), 'Silver': (28, 76),
        'Copper': (34, 70), 'Natural Gas': (34, 76),
        'USD/INR': (40, 70), 'US 10Y Yield': (40, 76), 'Dollar Index': (46, 70),
        'India VIX': (70, 64), 'US VIX': (22, 64),
    }

    def map_pin_html(name, x, y, data):
        if not data: return ''
        c = val_color(data['change_pct'])
        return f'<div class="map-pin" style="left:{x}%;top:{y}%" data-name="{name}" data-type="global"><span class="map-pin-name">{name}</span> <span class="map-pin-val" style="color:{c}" data-field="change">{data["change_pct"]:+.2f}%</span></div>'

    map_pins_html = ''
    all_map_data = {**global_data, **broad_data}
    for name, (x, y) in map_pins.items():
        d = all_map_data.get(name)
        if d:
            map_pins_html += map_pin_html(name, x, y, d)

    # ---------- Market table rows ----------
    def mkt_table_rows(markets_dict, source_data):
        rows = ''
        for name in markets_dict:
            d = source_data.get(name)
            if not d: continue
            c = val_color(d['change_pct'])
            rows += f'<tr data-name="{name}" data-type="global"><td class="td-name">{name}</td><td class="td-chg" style="color:{c}" data-field="change">{d["change_pct"]:+.2f}%</td><td class="td-price" data-field="price">{d["close"]:,.2f}</td></tr>'
        return rows

    us_rows = mkt_table_rows(US_MARKETS, global_data)
    eu_rows = mkt_table_rows(EUROPE_MARKETS, global_data)
    asia_rows = mkt_table_rows(ASIA_MARKETS, global_data)
    comm_rows = mkt_table_rows(COMMODITIES_FX, global_data)

    def intl_rows(stocks_dict):
        rows = ''
        for name, d in stocks_dict.items():
            c = val_color(d['change_pct'])
            rows += f'<tr data-name="{name}" data-type="china_japan"><td class="td-name">{name}</td><td class="td-chg" style="color:{c}" data-field="change">{d["change_pct"]:+.2f}%</td><td class="td-price" data-field="price">{d["close"]:,.2f}</td></tr>'
        return rows

    japan_stocks = {k: v for k, v in china_japan_data.items() if k in ['Toyota', 'Sony', 'SoftBank', 'Keyence', 'Nintendo']}
    china_stocks = {k: v for k, v in china_japan_data.items() if k in ['Alibaba', 'Tencent', 'BYD', 'Meituan', 'JD.com', 'PetroChina']}
    japan_rows = intl_rows(japan_stocks)
    china_rows = intl_rows(china_stocks)

    broad_cards = ''
    for name, d in broad_data.items():
        c = val_color(d['change_pct'])
        broad_cards += f'<div class="broad-card" data-name="{name}" data-type="broad"><div class="broad-label">{name}</div><div class="broad-price" data-field="price">{d["close"]:,.2f}</div><div class="broad-chg" style="color:{c}" data-field="change">{arrow_icon(d["change_pct"])} {d["change_pct"]:+.2f}%</div></div>'

    # Sector heatmap
    sorted_sectors = sorted(sectors_data.values(), key=lambda x: x['composite_score'], reverse=True)
    heatmap_html = ""
    for s in sorted_sectors:
        sc = s['composite_score']
        ac = '#0e7c6b' if 'BUY' in s['action'] else ('#dc2646' if 'SELL' in s['action'] else '#b45309')
        bar_w = min(abs(sc) / 3 * 100, 100)
        mag = s['expected_magnitude']
        mag_str = f"{mag:+.2f}%" if mag != 0 else "N/A"
        signal_icon = svg_trending_up if 'BUY' in s['action'] else (svg_trending_down if 'SELL' in s['action'] else svg_activity)
        heatmap_html += f"""
        <div class="hm-card" style="--sig:{ac}">
            <div class="hm-top">
                <span class="hm-sector" style="color:{s['color']}">{s['sector']}</span>
                <span class="hm-signal" style="color:{ac}">{signal_icon}</span>
            </div>
            <div class="hm-action" style="color:{ac}">{s['action']}</div>
            <div class="hm-metrics">
                <span>Exp: <strong style="color:{ac}">{mag_str}</strong></span>
                <span>Acc: {s['model_accuracy']:.0f}%</span>
            </div>
            <div class="hm-bar"><div class="hm-fill" style="width:{bar_w:.0f}%;background:{ac}"></div></div>
            <div class="hm-score">Score: {sc:+.2f}</div>
        </div>"""

    # Sector deep-dive cards
    sector_cards_html = ""
    for s in sorted_sectors:
        ac = '#0e7c6b' if 'BUY' in s['action'] else ('#dc2646' if 'SELL' in s['action'] else '#b45309')
        mag = s['expected_magnitude']

        gd_rows = ""
        for d in s['global_details'][:5]:
            ic = val_color(d['impact'])
            gc = val_color(d['change'])
            gd_rows += f"""<tr>
                <td class="td-name">{d["driver"]}</td>
                <td style="color:{gc}">{d["change"]:+.2f}%</td>
                <td class="td-muted">{d["correlation"]:+.1f}</td>
                <td style="color:{ic};font-weight:600">{d["impact"]:+.2f}</td>
            </tr>"""

        stock_rows = ""
        for sname, pred in s.get('stock_predictions', {}).items():
            if pred is None:
                stock_rows += f'<tr data-stock="{sname}"><td class="td-name">{sname}</td><td colspan="4" class="td-muted">No model</td></tr>'
                continue
            dc = val_color(1 if pred['direction'] == 'UP' else -1)
            stock_rows += f"""<tr data-stock="{sname}">
                <td class="td-name">{sname}</td>
                <td data-field="stock-price">{pred['close']:,.2f}</td>
                <td style="color:{dc}">{pred['direction']} ({pred['probability']:.0%})</td>
                <td style="color:{dc}">{pred['magnitude_pct']:+.2f}%</td>
                <td class="td-muted">L:{pred['lstm_prob']:.0%} X:{pred['xgb_prob']:.0%}</td>
            </tr>"""

        gs_c = val_color(s['global_score'])
        ml_c = val_color(s['ml_score'])
        mg_c = val_color(mag)

        sector_cards_html += f"""
        <div class="deep-card">
            <div class="deep-head">
                <div class="deep-title-wrap">
                    <div class="deep-dot" style="background:{s['color']}"></div>
                    <div>
                        <h3 class="deep-title">{s['sector']}</h3>
                        <p class="deep-desc">{s['description']}</p>
                    </div>
                </div>
                <div class="deep-badge" style="--sig:{ac}">
                    <span class="badge-action">{s['action']}</span>
                    <span class="badge-label">{s['prediction']}</span>
                </div>
            </div>
            <div class="kpi-row">
                <div class="kpi"><span class="kpi-label">Global</span><span class="kpi-val" style="color:{gs_c}">{s['global_score']:+.2f}</span></div>
                <div class="kpi"><span class="kpi-label">ML Score</span><span class="kpi-val" style="color:{ml_c}">{s['ml_score']:+.2f}</span></div>
                <div class="kpi kpi-main"><span class="kpi-label">Composite</span><span class="kpi-val" style="color:{ac};font-size:1.25rem">{s['composite_score']:+.2f}</span></div>
                <div class="kpi"><span class="kpi-label">Exp. Move</span><span class="kpi-val" style="color:{mg_c}">{mag:+.2f}%</span></div>
                <div class="kpi"><span class="kpi-label">Model Acc</span><span class="kpi-val">{s['model_accuracy']:.0f}%</span></div>
                <div class="kpi"><span class="kpi-label">Confidence</span><span class="kpi-val">{s['confidence']:.0f}%</span></div>
            </div>
            <div class="deep-tables">
                <div class="dtable">
                    <h4 class="dtable-title">{svg_globe} Global Drivers</h4>
                    <table><thead><tr><th>Driver</th><th>Change</th><th>Corr</th><th>Impact</th></tr></thead>
                    <tbody>{gd_rows}</tbody></table>
                </div>
                <div class="dtable">
                    <h4 class="dtable-title">{svg_cpu} ML Predictions (LSTM + XGBoost)</h4>
                    <table><thead><tr><th>Stock</th><th>CMP</th><th>Direction</th><th>Exp.</th><th>Split</th></tr></thead>
                    <tbody>{stock_rows}</tbody></table>
                </div>
            </div>
        </div>"""

    # Model accuracy stats
    all_accs = [s['model_accuracy'] for s in sectors_data.values() if s['model_accuracy'] > 0]
    avg_acc = np.mean(all_accs) if all_accs else 0
    max_acc = max(all_accs) if all_accs else 0

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Prediction &mdash; {report_date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300..800&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#ffffff;
  --bg-page:#f8fafb;
  --bg-card:#ffffff;
  --bg-hover:#f1f5f9;
  --border:#e2e8f0;
  --border-hover:#cbd5e1;
  --text:#0f172a;
  --text-secondary:#475569;
  --text-muted:#94a3b8;
  --accent:#0e7c6b;
  --accent-light:#ecfdf5;
  --green:#0e7c6b;
  --red:#dc2646;
  --amber:#b45309;
  --radius:14px;
  --radius-sm:10px;
  --shadow:0 1px 3px rgba(0,0,0,.04),0 1px 2px rgba(0,0,0,.02);
  --shadow-md:0 4px 12px rgba(0,0,0,.05);
  --shadow-lg:0 8px 24px rgba(0,0,0,.06);
}}
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
html{{font-size:16px;-webkit-font-smoothing:antialiased;scroll-behavior:smooth}}
body{{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg-page);color:var(--text);min-height:100vh;line-height:1.6;font-variant-numeric:tabular-nums}}
@media(prefers-reduced-motion:reduce){{*{{animation-duration:0s!important;transition-duration:0s!important}}}}
.app{{max-width:1480px;margin:0 auto;padding:0}}

/* Top bar */
.topbar{{display:flex;align-items:center;justify-content:space-between;padding:10px 32px;background:var(--bg);border-bottom:1px solid var(--border);font-size:.78rem;flex-wrap:wrap;gap:8px}}
.topbar-left{{display:flex;align-items:center;gap:20px;color:var(--text-secondary)}}
.topbar-ticker{{display:inline-flex;align-items:center;gap:6px;font-weight:500}}
.topbar-right{{display:flex;align-items:center;gap:12px}}
.live-dot{{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2.5s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}

/* Header */
.header{{display:flex;align-items:center;justify-content:space-between;padding:22px 32px;background:var(--bg);border-bottom:1px solid var(--border);flex-wrap:wrap;gap:12px}}
.header h1{{font-size:1.4rem;font-weight:700;letter-spacing:-.03em;color:var(--text)}}
.header-right{{display:flex;align-items:center;gap:10px}}
.btn{{display:inline-flex;align-items:center;gap:6px;padding:7px 16px;border-radius:var(--radius-sm);font-size:.78rem;font-weight:600;border:1px solid var(--border);cursor:pointer;transition:all .2s;font-family:inherit;background:var(--bg);color:var(--text-secondary)}}
.btn:hover{{border-color:var(--border-hover);color:var(--text);background:var(--bg-hover)}}
.btn:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.btn-primary{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.btn-primary:hover{{filter:brightness(1.08);color:#fff;background:var(--accent)}}
.btn-primary.loading{{opacity:.5;pointer-events:none}}
.auto-label{{display:flex;align-items:center;gap:5px;font-size:.72rem;color:var(--text-muted);cursor:pointer}}
.auto-label input{{accent-color:var(--accent)}}
.tag{{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:20px;font-size:.7rem;font-weight:500}}
.tag-ml{{background:rgba(14,124,107,.06);color:var(--accent)}}
.tag-live{{background:rgba(14,124,107,.06);color:var(--green)}}

/* Body */
.body{{padding:28px 32px}}
@media(max-width:768px){{.body{{padding:16px}}}}

/* Map */
.map-section{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:28px;margin-bottom:24px;position:relative;overflow:hidden;box-shadow:var(--shadow)}}
.map-section h2{{font-size:1.05rem;font-weight:700;margin-bottom:18px;letter-spacing:-.02em;color:var(--text)}}
.map-container{{position:relative;width:100%;aspect-ratio:2.4/1;background:linear-gradient(135deg,#f0faf7 0%,#eef5f2 50%,#f5f7fa 100%);border-radius:var(--radius-sm);overflow:hidden}}
.map-svg{{position:absolute;inset:0;width:100%;height:100%;opacity:.3}}
.map-pin{{position:absolute;display:flex;align-items:center;gap:4px;font-size:.66rem;font-weight:600;white-space:nowrap;z-index:2;cursor:default;transition:transform .2s,opacity .2s;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,.75);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);border:1px solid rgba(0,0,0,.04)}}
.map-pin:hover{{transform:scale(1.08);z-index:10;background:rgba(255,255,255,.95);box-shadow:var(--shadow-md)}}
.map-pin-name{{color:var(--text);font-weight:600;font-size:.62rem}}
.map-pin-val{{font-weight:700;font-size:.7rem}}
.map-legend{{display:flex;gap:20px;margin-top:14px;font-size:.7rem;color:var(--text-muted)}}
.map-legend span{{display:flex;align-items:center;gap:4px}}
.map-legend-dot{{width:8px;height:8px;border-radius:50%}}

/* Summary */
.summary-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
@media(max-width:900px){{.summary-row{{grid-template-columns:1fr 1fr}}}}
@media(max-width:500px){{.summary-row{{grid-template-columns:1fr}}}}
.summary-card{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow);transition:box-shadow .25s}}
.summary-card:hover{{box-shadow:var(--shadow-md)}}
.summary-label{{font-size:.62rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.12em;font-weight:600;margin-bottom:6px}}
.summary-value{{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;line-height:1.2}}
.summary-meta{{font-size:.75rem;color:var(--text-secondary);margin-top:4px}}
.summary-breakdown{{display:flex;gap:16px;margin-top:8px}}
.summary-stat-val{{font-size:1.1rem;font-weight:700}}
.summary-stat-label{{font-size:.6rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em}}

/* Broad */
.broad-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:24px}}
.broad-card{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:18px;text-align:center;box-shadow:var(--shadow);transition:all .25s}}
.broad-card:hover{{box-shadow:var(--shadow-md);transform:translateY(-1px)}}
.broad-label{{font-size:.68rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.1em;font-weight:600}}
.broad-price{{font-size:1.4rem;font-weight:700;margin:4px 0;color:var(--text)}}
.broad-chg{{display:inline-flex;align-items:center;gap:4px;font-size:.82rem;font-weight:600}}

/* Tables grid */
.tables-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-bottom:24px}}
.table-card{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow)}}
.table-card h3{{font-size:.82rem;font-weight:700;margin-bottom:14px;letter-spacing:-.01em;display:flex;align-items:center;gap:8px;color:var(--text)}}
.table-card h3 svg{{color:var(--accent);opacity:.6}}
.table-card table{{width:100%;border-collapse:collapse;font-size:.78rem}}
.table-card th{{text-align:left;font-size:.6rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600;padding:6px 0;border-bottom:1px solid var(--border)}}
.table-card th:last-child,.table-card td:last-child{{text-align:right}}
.table-card th:nth-child(2),.table-card td:nth-child(2){{text-align:right}}
.table-card td{{padding:8px 0;border-bottom:1px solid #f1f5f9;color:var(--text-secondary)}}
.table-card tbody tr{{transition:background .15s}}
.table-card tbody tr:hover{{background:var(--bg-hover)}}
.td-name{{font-weight:500;color:var(--text)}}
.td-chg{{font-weight:600}}
.td-price{{color:var(--text-secondary);font-variant-numeric:tabular-nums}}

/* Section headers */
.sec-head{{display:flex;align-items:center;justify-content:space-between;margin:30px 0 16px;padding-bottom:10px;border-bottom:2px solid var(--border)}}
.sec-head h2{{font-size:1rem;font-weight:700;color:var(--text);display:flex;align-items:center;gap:8px;letter-spacing:-.01em}}
.sec-head h2 svg{{color:var(--accent)}}

/* Heatmap */
.hm-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px;margin-bottom:28px}}
.hm-card{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:18px;box-shadow:var(--shadow);transition:all .25s;cursor:default;position:relative;overflow:hidden}}
.hm-card:hover{{box-shadow:var(--shadow-md);border-color:var(--border-hover)}}
.hm-card::before{{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:var(--sig,var(--border))}}
.hm-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
.hm-sector{{font-size:.8rem;font-weight:600;color:var(--text)}}
.hm-signal{{display:flex;align-items:center}}
.hm-action{{font-size:1.1rem;font-weight:800;margin:4px 0;letter-spacing:-.02em}}
.hm-metrics{{display:flex;justify-content:space-between;font-size:.72rem;color:var(--text-secondary);margin:8px 0 12px}}
.hm-bar{{height:3px;background:var(--border);border-radius:2px;overflow:hidden}}
.hm-fill{{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1)}}
.hm-score{{font-size:.62rem;color:var(--text-muted);margin-top:8px;text-align:right}}

/* Deep dive */
.deep-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px;margin-bottom:28px}}
@media(max-width:600px){{.deep-grid{{grid-template-columns:1fr}}}}
.deep-card{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:24px;box-shadow:var(--shadow);transition:all .25s}}
.deep-card:hover{{box-shadow:var(--shadow-md)}}
.deep-head{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;gap:16px}}
.deep-title-wrap{{display:flex;align-items:flex-start;gap:12px}}
.deep-dot{{width:10px;height:10px;border-radius:3px;margin-top:6px;flex-shrink:0}}
.deep-title{{font-size:1rem;font-weight:600;color:var(--text);letter-spacing:-.01em}}
.deep-desc{{font-size:.72rem;color:var(--text-muted);margin-top:3px;max-width:320px;line-height:1.5}}
.deep-badge{{padding:10px 16px;border-radius:var(--radius-sm);text-align:center;background:var(--bg-page);border:1.5px solid var(--sig)}}
.badge-action{{display:block;font-size:.88rem;font-weight:700;color:var(--sig)}}
.badge-label{{display:block;font-size:.6rem;color:var(--text-muted);margin-top:2px}}
.kpi-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:18px}}
@media(max-width:600px){{.kpi-row{{grid-template-columns:repeat(3,1fr)}}}}
.kpi{{background:var(--bg-page);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;text-align:center}}
.kpi-main{{border-color:var(--accent);background:var(--accent-light)}}
.kpi-label{{display:block;font-size:.56rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.1em;font-weight:600}}
.kpi-val{{display:block;font-size:.95rem;font-weight:700;margin-top:3px}}
.deep-tables{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:600px){{.deep-tables{{grid-template-columns:1fr}}}}
.dtable{{overflow-x:auto}}
.dtable-title{{font-size:.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.dtable-title svg{{color:var(--accent);opacity:.6}}
.dtable table{{width:100%;border-collapse:collapse;font-size:.76rem}}
.dtable th{{text-align:left;font-size:.6rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600;padding:6px 8px;border-bottom:1px solid var(--border)}}
.dtable td{{padding:7px 8px;border-bottom:1px solid #f1f5f9;color:var(--text-secondary)}}
.dtable tbody tr{{transition:background .15s}}
.dtable tbody tr:hover{{background:var(--bg-hover)}}
.td-muted{{color:var(--text-muted)}}

/* Footer */
.disclaimer{{background:var(--bg);border:1px solid #fde68a;border-radius:var(--radius-sm);padding:16px 20px;margin-top:28px;display:flex;align-items:flex-start;gap:10px;box-shadow:var(--shadow)}}
.disclaimer svg{{flex-shrink:0;color:var(--amber);margin-top:2px}}
.disclaimer p{{font-size:.75rem;color:var(--text-secondary);line-height:1.7}}
.disclaimer strong{{color:var(--amber)}}
.footer{{text-align:center;color:var(--text-muted);font-size:.62rem;margin-top:20px;padding-bottom:32px}}

/* Scrollbar */
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:3px}}
::-webkit-scrollbar-thumb:hover{{background:#94a3b8}}

/* Animations */
.reveal{{opacity:0;transform:translateY(10px);animation:reveal .45s cubic-bezier(.4,0,.2,1) forwards}}
@keyframes reveal{{to{{opacity:1;transform:translateY(0)}}}}
.reveal-d1{{animation-delay:.05s}}.reveal-d2{{animation-delay:.1s}}.reveal-d3{{animation-delay:.15s}}.reveal-d4{{animation-delay:.2s}}.reveal-d5{{animation-delay:.25s}}
</style></head><body>

<div class="app">

<!-- TOP BAR -->
<div class="topbar reveal">
    <div class="topbar-left">
        <span class="topbar-ticker" data-name="NIFTY 50" data-type="broad">NIFTY 50: <strong>{broad_data.get('NIFTY 50',{{}}).get('close',0):,.2f}</strong> <span style="color:{val_color(broad_data.get('NIFTY 50',{{}}).get('change_pct',0))}">{broad_data.get('NIFTY 50',{{}}).get('change_pct',0):+.2f}%</span></span>
        <span class="topbar-ticker" data-name="BANKNIFTY" data-type="broad">BANKNIFTY: <strong>{broad_data.get('BANKNIFTY',{{}}).get('close',0):,.2f}</strong> <span style="color:{val_color(broad_data.get('BANKNIFTY',{{}}).get('change_pct',0))}">{broad_data.get('BANKNIFTY',{{}}).get('change_pct',0):+.2f}%</span></span>
        <span class="topbar-ticker">India VIX: <strong style="color:{vix_color}">{vix_val:.2f}</strong></span>
    </div>
    <div class="topbar-right">
        <span id="connStatus" style="font-size:.72rem;color:var(--text-muted)">Connecting...</span>
        <span style="color:var(--border)">&middot;</span>
        <span style="font-size:.72rem;color:var(--text-muted)"><span id="lastUpdate">{report_time}</span></span>
    </div>
</div>

<!-- HEADER -->
<div class="header reveal">
    <div style="display:flex;align-items:center;gap:14px">
        <h1>The Market</h1>
        <span class="tag tag-ml">{svg_cpu} LSTM + XGBoost</span>
        <span class="tag tag-live"><span class="live-dot"></span> Live</span>
    </div>
    <div class="header-right">
        <label class="auto-label"><input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()"> Auto</label>
        <button class="btn" onclick="fetchPrices()">{svg_refresh} Prices</button>
        <button class="btn btn-primary" id="refreshAll" onclick="refreshAll()">{svg_refresh} Full Refresh</button>
    </div>
</div>

<div class="body">

<!-- WORLD MAP -->
<div class="map-section reveal reveal-d1">
    <h2>Global Markets &mdash; {report_date}</h2>
    <div class="map-container">
        <svg class="map-svg" viewBox="0 0 1200 500" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
            <path d="M80,80 L120,60 L180,55 L220,70 L280,65 L300,80 L280,120 L300,150 L280,180 L260,200 L240,230 L220,250 L200,280 L180,300 L170,280 L160,250 L140,230 L120,210 L100,190 L80,160 L70,130 L75,100Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
            <path d="M220,300 L250,310 L280,350 L290,390 L280,430 L260,450 L240,440 L220,420 L200,390 L190,360 L200,330Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
            <path d="M480,70 L520,60 L560,70 L600,65 L620,80 L610,100 L620,120 L600,140 L580,130 L560,140 L540,130 L520,140 L500,130 L490,110 L480,90Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
            <path d="M500,180 L540,170 L580,180 L600,210 L610,250 L600,300 L580,340 L560,360 L540,350 L520,320 L500,280 L490,240 L495,210Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
            <path d="M620,60 L680,50 L740,55 L800,60 L860,70 L920,80 L960,100 L940,140 L920,180 L880,200 L840,210 L800,220 L760,230 L720,220 L680,200 L660,170 L640,140 L630,110 L625,80Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
            <path d="M740,220 L760,210 L780,230 L790,260 L780,290 L770,310 L760,300 L750,270 L740,250Z" fill="#b0e0d0" stroke="#80ccb0" stroke-width="1.5"/>
            <path d="M960,120 L970,140 L965,170 L955,190 L950,170 L955,145Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
            <path d="M880,350 L920,340 L960,350 L990,370 L1000,400 L990,430 L960,440 L920,430 L890,410 L880,380Z" fill="#c4ede3" stroke="#a0ddd0" stroke-width="1"/>
        </svg>
        {map_pins_html}
    </div>
    <div class="map-legend">
        <span><span class="map-legend-dot" style="background:var(--green)"></span> Positive</span>
        <span><span class="map-legend-dot" style="background:var(--red)"></span> Negative</span>
    </div>
</div>

<!-- SUMMARY -->
<div class="summary-row reveal reveal-d2">
    <div class="summary-card">
        <div class="summary-label">Market Outlook</div>
        <div class="summary-value" style="color:{mood_color}">{mood}</div>
        <div class="summary-meta">Composite Score {avg_composite:+.2f}</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Sectors</div>
        <div class="summary-breakdown">
            <div><div class="summary-stat-val" style="color:var(--green)">{bullish}</div><div class="summary-stat-label">Bullish</div></div>
            <div><div class="summary-stat-val" style="color:var(--amber)">{neutral}</div><div class="summary-stat-label">Neutral</div></div>
            <div><div class="summary-stat-val" style="color:var(--red)">{bearish}</div><div class="summary-stat-label">Bearish</div></div>
        </div>
    </div>
    <div class="summary-card">
        <div class="summary-label">India VIX</div>
        <div class="summary-value" style="color:{vix_color}">{vix_val:.2f}</div>
        <div class="summary-meta" style="color:{vix_color}">{vix_status} ({vix_chg:+.1f}%)</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">ML Accuracy</div>
        <div class="summary-value" style="color:var(--accent)">{avg_acc:.0f}%</div>
        <div class="summary-meta">Best: {max_acc:.0f}% &middot; {len(all_accs)} models</div>
    </div>
</div>

<!-- BROAD MARKET -->
<div class="broad-row reveal reveal-d2">{broad_cards}</div>

<!-- MARKET TABLES -->
<div class="tables-grid reveal reveal-d3">
    <div class="table-card"><h3>{svg_globe} US Markets</h3><table><thead><tr><th>Index</th><th>+/-%</th><th>Last</th></tr></thead><tbody>{us_rows}</tbody></table></div>
    <div class="table-card"><h3>{svg_globe} Europe</h3><table><thead><tr><th>Index</th><th>+/-%</th><th>Last</th></tr></thead><tbody>{eu_rows}</tbody></table></div>
    <div class="table-card"><h3>{svg_globe} Asia Pacific</h3><table><thead><tr><th>Index</th><th>+/-%</th><th>Last</th></tr></thead><tbody>{asia_rows}</tbody></table></div>
    <div class="table-card"><h3>{svg_globe} Commodities &amp; Forex</h3><table><thead><tr><th>Item</th><th>+/-%</th><th>Last</th></tr></thead><tbody>{comm_rows}</tbody></table></div>
    <div class="table-card"><h3>{svg_globe} Japan</h3><table><thead><tr><th>Stock</th><th>+/-%</th><th>Last</th></tr></thead><tbody>{japan_rows}</tbody></table></div>
    <div class="table-card"><h3>{svg_globe} China &amp; Hong Kong</h3><table><thead><tr><th>Stock</th><th>+/-%</th><th>Last</th></tr></thead><tbody>{china_rows}</tbody></table></div>
</div>

<!-- SECTOR HEATMAP -->
<div class="sec-head"><h2>{svg_bar_chart} Sector Predictions</h2></div>
<div class="hm-grid reveal reveal-d4">{heatmap_html}</div>

<!-- SECTOR DEEP DIVE -->
<div class="sec-head"><h2>{svg_cpu} Sector Deep Dive</h2></div>
<div class="deep-grid reveal reveal-d5">{sector_cards_html}</div>

<!-- DISCLAIMER -->
<div class="disclaimer reveal">
    {svg_alert}
    <p><strong>Disclaimer:</strong> ML-based analysis for educational purposes only. Models trained on historical data cannot predict black-swan events. Consult a SEBI-registered advisor.</p>
</div>
<div class="footer">v5 &middot; LSTM + XGBoost Ensemble &middot; TradingView Data &middot; TensorFlow + scikit-learn</div>

</div></div>

<script>
const IS_LOCAL=location.hostname==='localhost'||location.hostname==='127.0.0.1';
const SERVER=IS_LOCAL?'http://localhost:8080':'';
let priceTimer=null;let autoTimer=null;let isLive=false;
const UP_SVG='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>';
const DN_SVG='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline><polyline points="17 18 23 18 23 12"></polyline></svg>';

function fmt(n){{return n.toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}
function pctFmt(n){{return(n>=0?'+':'')+n.toFixed(2)+'%'}}
function clr(n){{return n>=0?'#0e7c6b':'#dc2646'}}
function arrow(n){{return n>=0?UP_SVG:DN_SVG}}

function flashCard(el){{
    el.style.transition='none';
    el.style.boxShadow='0 0 16px rgba(14,124,107,.15)';
    setTimeout(()=>{{el.style.transition='box-shadow .8s ease';el.style.boxShadow='none'}},50);
}}

function updateCard(el,close,chg){{
    const priceEl=el.querySelector('[data-field="price"]');
    const changeEl=el.querySelector('[data-field="change"]');
    if(priceEl){{
        const oldVal=priceEl.textContent.replace(/,/g,'');
        const newVal=fmt(close);
        if(oldVal!==newVal.replace(/,/g,'')){{priceEl.textContent=newVal;flashCard(el);}}
    }}
    if(changeEl){{
        const tag=el.tagName.toLowerCase();
        if(tag==='tr'){{changeEl.innerHTML=pctFmt(chg);}}
        else{{changeEl.innerHTML=arrow(chg)+' '+pctFmt(chg);}}
        changeEl.style.color=clr(chg);
    }}
}}

async function fetchPrices(){{
    const statusEl=document.getElementById('connStatus');
    const timeEl=document.getElementById('lastUpdate');
    try{{
        const r=await fetch(SERVER+'/api/prices',{{signal:AbortSignal.timeout(60000)}});
        const data=await r.json();
        if(data.error)return;
        if(data.global){{document.querySelectorAll('[data-type="global"]').forEach(el=>{{const name=el.dataset.name;if(data.global[name])updateCard(el,data.global[name].close,data.global[name].change_pct)}})}}
        if(data.broad){{document.querySelectorAll('[data-type="broad"]').forEach(el=>{{const name=el.dataset.name;if(data.broad[name])updateCard(el,data.broad[name].close,data.broad[name].change_pct)}})}}
        if(data.china_japan){{document.querySelectorAll('[data-type="china_japan"]').forEach(el=>{{const name=el.dataset.name;if(data.china_japan[name])updateCard(el,data.china_japan[name].close,data.china_japan[name].change_pct)}})}}
        if(data.stocks){{document.querySelectorAll('[data-stock]').forEach(el=>{{const sname=el.dataset.stock;if(data.stocks[sname]){{const td=el.querySelector('[data-field="stock-price"]');if(td){{const old=td.textContent.replace(/,/g,'');const nv=fmt(data.stocks[sname].close);if(old!==nv.replace(/,/g,'')){{td.textContent=nv;td.style.transition='none';td.style.color='var(--accent)';setTimeout(()=>{{td.style.transition='color .8s';td.style.color='var(--text-secondary)'}},50)}}}}}}}})}}
        if(!isLive){{isLive=true;if(statusEl)statusEl.innerHTML='<span class="live-dot"></span> Connected'}}
        if(timeEl)timeEl.textContent=new Date().toLocaleTimeString('en-US',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
    }}catch(e){{
        if(statusEl&&isLive){{statusEl.innerHTML='<span style="color:var(--amber)">Offline</span>';isLive=false}}
    }}
}}

function refreshAll(){{
    if(!IS_LOCAL){{fetchPrices();return}}
    const btn=document.getElementById('refreshAll');
    btn.classList.add('loading');
    fetch(SERVER+'/api/refresh',{{signal:AbortSignal.timeout(180000)}}).then(r=>r.json()).then(()=>window.location.reload()).catch(()=>{{fetchPrices();btn.classList.remove('loading')}});
}}

function toggleAutoRefresh(){{
    const on=document.getElementById('autoRefresh').checked;
    if(on){{if(!priceTimer)priceTimer=setInterval(fetchPrices,30000);autoTimer=setInterval(refreshAll,300000)}}
    else{{if(priceTimer){{clearInterval(priceTimer);priceTimer=null}}if(autoTimer){{clearInterval(autoTimer);autoTimer=null}}}}
}}

document.getElementById('lastUpdate').textContent=new Date().toLocaleTimeString('en-US',{{hour:'2-digit',minute:'2-digit'}});
fetch(SERVER+'/api/status',{{signal:AbortSignal.timeout(3000)}}).then(r=>r.json()).then(d=>{{
    isLive=true;const s=document.getElementById('connStatus');if(s)s.innerHTML='<span class="live-dot"></span> Connected';
    fetchPrices();priceTimer=setInterval(fetchPrices,30000);
}}).catch(()=>{{
    if(!IS_LOCAL){{fetchPrices();priceTimer=setInterval(fetchPrices,30000);const s=document.getElementById('connStatus');if(s)s.innerHTML='<span class="live-dot"></span> Vercel Live'}}
    else{{const s=document.getElementById('connStatus');if(s)s.innerHTML='<span style="color:var(--text-muted)">Start live_server.py for real-time updates</span>'}}
}});
</script>
</body></html>"""

    date_str = now.strftime('%Y-%m-%d')
    dated = os.path.join(OUTPUT_DIR, f'prediction_{date_str}.html')
    latest = os.path.join(OUTPUT_DIR, 'prediction_latest.html')

    # Also save to public/index.html for Vercel deployment
    _base = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.join(_base, 'public')
    os.makedirs(public_dir, exist_ok=True)
    public_index = os.path.join(public_dir, 'index.html')

    for path in [dated, latest, public_index]:
        with open(path, 'w') as f:
            f.write(html)

    print(f"\n  Reports saved to {OUTPUT_DIR}/")
    print(f"  Vercel dashboard saved to public/index.html")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if '--train' in sys.argv:
        train_all_models()
        run_predictions()
    elif '--backtest' in sys.argv:
        from backtester import run_all_backtests
        from features import fetch_data
        print("\n  Running Backtrader Backtests...")
        instruments = {}
        for name, sym in BROAD_MARKET.items():
            df = fetch_data(sym, period='3y')
            if df is not None:
                instruments[name] = df
        for sector in SECTORS.values():
            for sname, sym in list(sector['stocks'].items())[:2]:
                df = fetch_data(sym, period='3y')
                if df is not None:
                    instruments[sname] = df
        run_all_backtests(instruments)
    else:
        run_predictions()
