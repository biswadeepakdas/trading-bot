"""
Vercel Serverless Function — Live Price Fetcher
Uses Yahoo Finance direct API via urllib (zero dependencies).
Fetches all global markets, broad market, sector stocks, and China/Japan in parallel.
"""
from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Symbol Definitions (mirrors config.py) ──────────────────────────
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
COMMODITIES_FX = {
    'Crude Oil (WTI)': 'CL=F', 'Brent Crude': 'BZ=F',
    'Gold': 'GC=F', 'Silver': 'SI=F', 'Copper': 'HG=F',
    'Natural Gas': 'NG=F', 'USD/INR': 'INR=X',
    'US 10Y Yield': '^TNX', 'Dollar Index': 'DX-Y.NYB',
}
VOLATILITY = {'India VIX': '^INDIAVIX', 'US VIX': '^VIX'}
CHINA_JAPAN_DETAIL = {
    'Toyota': '7203.T', 'Sony': '6758.T', 'SoftBank': '9984.T',
    'Keyence': '6861.T', 'Nintendo': '7974.T',
    'Alibaba': '9988.HK', 'Tencent': '0700.HK', 'BYD': '1211.HK',
    'Meituan': '3690.HK', 'JD.com': '9618.HK', 'PetroChina': '0857.HK',
}
BROAD_MARKET = {'NIFTY 50': '^NSEI', 'BANKNIFTY': '^NSEBANK'}

SECTORS = {
    'Banking & Financials': {'HDFCBANK': 'HDFCBANK.NS', 'ICICIBANK': 'ICICIBANK.NS',
        'KOTAKBANK': 'KOTAKBANK.NS', 'SBIN': 'SBIN.NS', 'BAJFINANCE': 'BAJFINANCE.NS'},
    'IT & Technology': {'TCS': 'TCS.NS', 'INFY': 'INFY.NS', 'WIPRO': 'WIPRO.NS',
        'HCLTECH': 'HCLTECH.NS', 'TECHM': 'TECHM.NS'},
    'Oil Upstream (Producers)': {'ONGC': 'ONGC.NS', 'OIL INDIA': 'OIL.NS', 'RELIANCE': 'RELIANCE.NS'},
    'Oil Downstream (OMCs)': {'BPCL': 'BPCL.NS', 'IOC': 'IOC.NS', 'HINDPETRO': 'HINDPETRO.NS'},
    'Metals & Mining': {'TATASTEEL': 'TATASTEEL.NS', 'HINDALCO': 'HINDALCO.NS',
        'JSWSTEEL': 'JSWSTEEL.NS', 'COALINDIA': 'COALINDIA.NS'},
    'Pharma & Healthcare': {'SUNPHARMA': 'SUNPHARMA.NS', 'DRREDDY': 'DRREDDY.NS',
        'CIPLA': 'CIPLA.NS', 'DIVISLAB': 'DIVISLAB.NS'},
    'Auto & EV': {'TATAMOTORS': 'TATAMOTORS.BO', 'M&M': 'M&M.NS',
        'MARUTI': 'MARUTI.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS'},
    'FMCG & Consumer': {'HINDUNILVR': 'HINDUNILVR.NS', 'ITC': 'ITC.NS',
        'NESTLEIND': 'NESTLEIND.NS', 'BRITANNIA': 'BRITANNIA.NS'},
}


def _fetch_yahoo(symbol):
    """Fetch current price from Yahoo Finance direct API."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?interval=1d&range=2d"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read().decode())
        meta = data['chart']['result'][0]['meta']
        price = meta.get('regularMarketPrice', 0)
        prev = meta.get('chartPreviousClose', meta.get('previousClose', price))
        change_pct = round((price - prev) / prev * 100, 2) if prev else 0
        return {'close': round(price, 2), 'change_pct': change_pct}
    except Exception:
        return None


import urllib.parse

def _fetch_batch(name_symbol_dict, max_workers=15):
    """Fetch prices for a dict of {name: yahoo_symbol} in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_yahoo, sym): name
                   for name, sym in name_symbol_dict.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    results[name] = result
            except Exception:
                pass
    return results


def fetch_all_prices():
    """Fetch all prices across all categories."""
    # Merge all global markets into one dict for parallel fetch
    all_global = {}
    all_global.update(US_MARKETS)
    all_global.update(EUROPE_MARKETS)
    all_global.update(ASIA_MARKETS)
    all_global.update(COMMODITIES_FX)
    all_global.update(VOLATILITY)

    # Collect all stocks
    all_stocks = {}
    for sector_stocks in SECTORS.values():
        all_stocks.update(sector_stocks)

    # Fetch everything in parallel batches
    with ThreadPoolExecutor(max_workers=25) as executor:
        f_global = executor.submit(_fetch_batch, all_global, 20)
        f_broad = executor.submit(_fetch_batch, BROAD_MARKET, 5)
        f_stocks = executor.submit(_fetch_batch, all_stocks, 20)
        f_cj = executor.submit(_fetch_batch, CHINA_JAPAN_DETAIL, 15)

    return {
        'timestamp': datetime.now().isoformat(),
        'global': f_global.result(),
        'broad': f_broad.result(),
        'stocks': f_stocks.result(),
        'china_japan': f_cj.result(),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            data = fetch_all_prices()
            body = json.dumps(data).encode()
            self.send_response(200)
        except Exception as e:
            body = json.dumps({'error': str(e)}).encode()
            self.send_response(500)

        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
