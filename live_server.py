#!/usr/bin/env python3
"""
Live Dashboard Server — Real-Time Market Prediction
=====================================================
A lightweight HTTP server that:
  1. Serves the prediction HTML dashboard
  2. Provides /api/prices for fast real-time price updates (no ML)
  3. Provides /api/refresh for full ML re-prediction
  4. All data fetched live from TradingView via tvdatafeed

Usage:
  python live_server.py              # Start on port 8080
  python live_server.py --port 9090  # Custom port
"""

import sys
import os
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (OUTPUT_DIR, US_MARKETS, EUROPE_MARKETS, ASIA_MARKETS,
                    COMMODITIES_FX, VOLATILITY, BROAD_MARKET, SECTORS,
                    CHINA_JAPAN_DETAIL)

refresh_lock = threading.Lock()
price_lock = threading.Lock()
last_refresh_time = None


def fetch_live_prices():
    """Fast price-only fetch — no ML, no features. Returns in ~5-15 seconds."""
    from features import _yf_to_tv_symbol, _tv_fetch

    result = {
        'timestamp': datetime.now().isoformat(),
        'global': {},
        'broad': {},
        'stocks': {},
        'china_japan': {},
    }

    # 1. Global markets
    all_global = {}
    all_global.update(US_MARKETS)
    all_global.update(EUROPE_MARKETS)
    all_global.update(ASIA_MARKETS)
    all_global.update(COMMODITIES_FX)
    all_global.update(VOLATILITY)

    for name, sym in all_global.items():
        try:
            tv_info = _yf_to_tv_symbol(sym)
            if tv_info:
                df = _tv_fetch(tv_info[0], tv_info[1], n_bars=5)
                if df is not None and len(df) >= 2:
                    close = round(float(df['Close'].iloc[-1]), 2)
                    prev = round(float(df['Close'].iloc[-2]), 2)
                    chg = round(((close - prev) / prev) * 100, 2) if prev != 0 else 0
                    result['global'][name] = {'close': close, 'change_pct': chg}
        except Exception:
            pass

    # 2. Broad market
    for name, sym in BROAD_MARKET.items():
        try:
            tv_info = _yf_to_tv_symbol(sym)
            if tv_info:
                df = _tv_fetch(tv_info[0], tv_info[1], n_bars=5)
                if df is not None and len(df) >= 2:
                    close = round(float(df['Close'].iloc[-1]), 2)
                    prev = round(float(df['Close'].iloc[-2]), 2)
                    chg = round(((close - prev) / prev) * 100, 2) if prev != 0 else 0
                    result['broad'][name] = {'close': close, 'change_pct': chg}
        except Exception:
            pass

    # 3. Indian stocks (sector stocks)
    for sector_name, sector_config in SECTORS.items():
        for stock_name, symbol in sector_config['stocks'].items():
            try:
                tv_info = _yf_to_tv_symbol(symbol)
                if tv_info:
                    df = _tv_fetch(tv_info[0], tv_info[1], n_bars=5)
                    if df is not None and len(df) >= 2:
                        close = round(float(df['Close'].iloc[-1]), 2)
                        prev = round(float(df['Close'].iloc[-2]), 2)
                        chg = round(((close - prev) / prev) * 100, 2) if prev != 0 else 0
                        result['stocks'][stock_name] = {'close': close, 'change_pct': chg}
            except Exception:
                pass

    # 4. China & Japan
    for name, sym in CHINA_JAPAN_DETAIL.items():
        try:
            tv_info = _yf_to_tv_symbol(sym)
            if tv_info:
                df = _tv_fetch(tv_info[0], tv_info[1], n_bars=5)
                if df is not None and len(df) >= 2:
                    close = round(float(df['Close'].iloc[-1]), 2)
                    prev = round(float(df['Close'].iloc[-2]), 2)
                    chg = round(((close - prev) / prev) * 100, 2) if prev != 0 else 0
                    result['china_japan'][name] = {'close': close, 'change_pct': chg}
        except Exception:
            pass

    return result


def run_full_refresh():
    """Re-run the entire prediction pipeline with fresh data."""
    global last_refresh_time
    from run_prediction import run_predictions
    print(f"\n[SERVER] Full refresh at {datetime.now().strftime('%H:%M:%S')}...")
    start = time.time()
    run_predictions()
    elapsed = time.time() - start
    last_refresh_time = datetime.now().isoformat()
    print(f"[SERVER] Refresh complete in {elapsed:.1f}s")
    return {'status': 'ok', 'elapsed': round(elapsed, 1), 'timestamp': last_refresh_time}


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler: dashboard + price API + refresh API."""

    def do_GET(self):
        parsed = urlparse(self.path)

        # Fast price-only API (no ML)
        if parsed.path == '/api/prices':
            self.handle_prices()
            return

        # Full ML refresh API
        if parsed.path == '/api/refresh':
            self.handle_refresh()
            return

        # Status API
        if parsed.path == '/api/status':
            self.send_json({'status': 'running', 'last_refresh': last_refresh_time})
            return

        # Serve dashboard
        if parsed.path in ('/', '/index.html', '/dashboard'):
            self.serve_dashboard()
            return

        super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def handle_prices(self):
        """Fast price fetch — updates dashboard in seconds."""
        if not price_lock.acquire(blocking=False):
            self.send_json({'status': 'busy', 'message': 'Price fetch in progress'}, 429)
            return
        try:
            data = fetch_live_prices()
            data['status'] = 'ok'
            self.send_json(data)
        except Exception as e:
            self.send_json({'status': 'error', 'message': str(e)}, 500)
        finally:
            price_lock.release()

    def handle_refresh(self):
        """Full ML prediction refresh."""
        if not refresh_lock.acquire(blocking=False):
            self.send_json({'status': 'busy', 'message': 'Refresh already in progress'}, 429)
            return
        try:
            result = run_full_refresh()
            self.send_json(result)
        except Exception as e:
            self.send_json({'status': 'error', 'message': str(e)}, 500)
        finally:
            refresh_lock.release()

    def serve_dashboard(self):
        latest = os.path.join(OUTPUT_DIR, 'prediction_latest.html')
        if not os.path.exists(latest):
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>No report found. Click Refresh to generate.</h1>')
            return

        with open(latest, 'r') as f:
            content = f.read()

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {args[0]}")


def main():
    port = 8080
    if '--port' in sys.argv:
        idx = sys.argv.index('--port')
        port = int(sys.argv[idx + 1])

    os.chdir(OUTPUT_DIR)

    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print("=" * 60)
    print("  ML MARKET PREDICTION — LIVE DASHBOARD")
    print("=" * 60)
    print(f"  Dashboard:    http://localhost:{port}/")
    print(f"  Live Prices:  http://localhost:{port}/api/prices")
    print(f"  Full Refresh: http://localhost:{port}/api/refresh")
    print(f"  Status:       http://localhost:{port}/api/status")
    print("=" * 60)
    print("  Press Ctrl+C to stop\n")

    latest = os.path.join(OUTPUT_DIR, 'prediction_latest.html')
    if not os.path.exists(latest):
        print("[SERVER] No report — generating initial prediction...")
        try:
            run_full_refresh()
        except Exception as e:
            print(f"[SERVER] Initial generation failed: {e}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
