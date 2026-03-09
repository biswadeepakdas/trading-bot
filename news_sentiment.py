#!/usr/bin/env python3
"""
News & Social Sentiment Analysis Module
========================================
Fetches news from multiple free sources, analyzes sentiment,
and maps impact to Indian market sectors.

Sources:
  - Google News RSS (free, no API key)
  - RSS feeds from major financial outlets
  - Social sentiment via Reddit (free API)

Sentiment Analysis:
  - Keyword-based financial sentiment scoring
  - Sector-specific keyword mapping
  - Geopolitical event detection
"""

import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FINANCIAL SENTIMENT LEXICON
# ============================================================
BULLISH_WORDS = {
    # Strong bullish
    'surge': 2, 'soar': 2, 'rally': 2, 'boom': 2, 'breakout': 2,
    'skyrocket': 2.5, 'record high': 2.5, 'all-time high': 2.5,
    'bull run': 2, 'bull market': 2, 'outperform': 1.5,
    # Medium bullish
    'rise': 1, 'gain': 1, 'jump': 1.2, 'climb': 1, 'advance': 1,
    'recover': 1, 'rebound': 1.2, 'upgrade': 1.5, 'buy': 1,
    'bullish': 1.5, 'optimistic': 1, 'growth': 1, 'profit': 1,
    'beat': 1.2, 'exceed': 1.2, 'strong': 0.8, 'positive': 0.8,
    'boost': 1, 'expand': 0.8, 'upbeat': 1, 'uptick': 0.8,
    'stimulus': 1.2, 'rate cut': 1.5, 'easing': 1,
    'investment': 0.5, 'inflow': 1, 'fii buying': 1.5, 'dii buying': 1.2,
    'green': 0.5, 'up': 0.3, 'higher': 0.5,
}

BEARISH_WORDS = {
    # Strong bearish
    'crash': -2.5, 'plunge': -2, 'collapse': -2.5, 'freefall': -2.5,
    'meltdown': -2, 'panic': -2, 'crisis': -2, 'recession': -2,
    'bear market': -2, 'bloodbath': -2.5,
    # Medium bearish
    'fall': -1, 'drop': -1, 'decline': -1, 'slip': -0.8, 'slide': -1,
    'tumble': -1.5, 'sink': -1.2, 'plummet': -1.8, 'sell-off': -1.5,
    'selloff': -1.5, 'downgrade': -1.5, 'sell': -1,
    'bearish': -1.5, 'pessimistic': -1, 'loss': -1, 'miss': -1.2,
    'weak': -0.8, 'negative': -0.8, 'concern': -0.8, 'fear': -1,
    'risk': -0.5, 'uncertainty': -0.8, 'volatile': -0.5,
    'tariff': -1.2, 'sanction': -1.2, 'war': -1.5, 'conflict': -1.2,
    'inflation': -0.8, 'rate hike': -1.2, 'tightening': -1,
    'outflow': -1, 'fii selling': -1.5, 'dii selling': -1.2,
    'red': -0.5, 'down': -0.3, 'lower': -0.5,
}

# Geopolitical / macro events with broad market impact
GEO_EVENTS = {
    'war': -2, 'invasion': -2.5, 'missile': -2, 'nuclear': -2.5,
    'ceasefire': 1.5, 'peace': 1.5, 'treaty': 1,
    'election': 0, 'coup': -2, 'protest': -0.8,
    'fed': 0, 'rbi': 0, 'central bank': 0,
    'rate cut': 1.5, 'rate hike': -1.2,
    'stimulus': 1.2, 'bailout': 0.5,
    'default': -2, 'debt ceiling': -1,
    'trade war': -1.5, 'trade deal': 1.5,
    'covid': -1, 'pandemic': -1.5, 'lockdown': -1.5,
    'earthquake': -0.5, 'flood': -0.5, 'hurricane': -0.5,
    'oil embargo': -1.5, 'opec cut': 1, 'opec': 0.5,
}

# Sector keyword mapping
SECTOR_KEYWORDS = {
    'Banking & Financials': [
        'bank', 'banking', 'npa', 'credit', 'loan', 'rbi', 'interest rate',
        'hdfc', 'icici', 'sbi', 'kotak', 'bajaj finance', 'nbfc',
        'lending', 'deposit', 'mortgage', 'fintech', 'upi', 'digital payment',
        'financial', 'fiscal', 'monetary policy',
    ],
    'IT & Technology': [
        'it sector', 'software', 'technology', 'tech', 'tcs', 'infosys', 'wipro',
        'hcl tech', 'tech mahindra', 'outsourcing', 'ai ', 'artificial intelligence',
        'cloud', 'saas', 'digital', 'semiconductor', 'chip', 'silicon',
        'nasdaq', 'h1b', 'visa', 'offshoring',
    ],
    'Oil Upstream (Producers)': [
        'crude oil', 'oil price', 'brent', 'wti', 'opec', 'ongc', 'oil india',
        'reliance', 'petroleum', 'drilling', 'exploration', 'oil production',
        'shale', 'refinery', 'pipeline', 'energy',
    ],
    'Oil Downstream (OMCs)': [
        'petrol price', 'diesel price', 'fuel price', 'bpcl', 'ioc', 'hpcl',
        'oil marketing', 'subsidy', 'deregulation', 'fuel tax',
        'hindpetro', 'gas price', 'lpg',
    ],
    'Metals & Mining': [
        'steel', 'metal', 'mining', 'iron ore', 'copper', 'aluminium', 'aluminum',
        'tata steel', 'hindalco', 'jsw', 'coal india', 'coal',
        'gold', 'silver', 'zinc', 'nickel', 'commodity',
        'china demand', 'infrastructure',
    ],
    'Pharma & Healthcare': [
        'pharma', 'drug', 'fda', 'usfda', 'medicine', 'hospital', 'healthcare',
        'sun pharma', 'dr reddy', 'cipla', 'divis lab', 'vaccine',
        'biotech', 'clinical trial', 'generic', 'api',
    ],
    'Auto & EV': [
        'auto', 'automobile', 'car', 'vehicle', 'ev ', 'electric vehicle',
        'tata motors', 'mahindra', 'maruti', 'bajaj auto',
        'tesla', 'battery', 'lithium', 'charging', 'emission',
        'sales number', 'dispatch',
    ],
    'FMCG & Consumer': [
        'fmcg', 'consumer', 'hindustan unilever', 'itc', 'nestle', 'britannia',
        'rural demand', 'urban consumption', 'retail', 'inflation',
        'food price', 'commodity price', 'monsoon', 'rainfall',
    ],
}

# India-specific keywords
INDIA_KEYWORDS = [
    'india', 'nifty', 'sensex', 'bse', 'nse', 'sebi', 'rbi',
    'rupee', 'inr', 'fii', 'dii', 'mutual fund', 'indian market',
    'modi', 'budget', 'gst', 'make in india',
]


# ============================================================
# NEWS FETCHING
# ============================================================

def _fetch_url(url, timeout=15):
    """Fetch URL content with error handling."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return None


def fetch_google_news(query, num_results=20):
    """Fetch news from Google News RSS."""
    articles = []
    q = urllib.parse.quote(query)
    url = f'https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en'
    raw = _fetch_url(url)
    if not raw:
        return articles

    try:
        root = ET.fromstring(raw)
        for item in root.findall('.//item')[:num_results]:
            title = item.findtext('title', '')
            pub_date = item.findtext('pubDate', '')
            source = item.findtext('source', '')
            desc = item.findtext('description', '')
            articles.append({
                'title': title,
                'source': source,
                'date': pub_date,
                'description': _strip_html(desc),
                'query': query,
            })
    except:
        pass
    return articles


def fetch_reddit_sentiment(subreddit='IndianStockMarket', limit=25):
    """Fetch recent posts from Reddit for social sentiment."""
    articles = []
    url = f'https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}'
    raw = _fetch_url(url)
    if not raw:
        # Try alternative subreddits
        for sub in ['IndianStreetBets', 'DalalStreetTalks', 'wallstreetbets']:
            url = f'https://www.reddit.com/r/{sub}/hot.json?limit={limit}'
            raw = _fetch_url(url)
            if raw:
                subreddit = sub
                break

    if not raw:
        return articles

    try:
        data = json.loads(raw)
        for post in data.get('data', {}).get('children', []):
            d = post.get('data', {})
            articles.append({
                'title': d.get('title', ''),
                'source': f'Reddit r/{subreddit}',
                'date': datetime.fromtimestamp(d.get('created_utc', 0)).strftime('%a, %d %b %Y'),
                'description': d.get('selftext', '')[:300],
                'score': d.get('score', 0),
                'comments': d.get('num_comments', 0),
                'query': 'social',
            })
    except:
        pass
    return articles


def _strip_html(text):
    """Remove HTML tags."""
    return re.sub(r'<[^>]+>', '', text or '')


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text):
    """Score sentiment of a text using financial lexicon. Returns float in [-5, +5]."""
    if not text:
        return 0.0
    text_lower = text.lower()
    score = 0.0

    for word, val in BULLISH_WORDS.items():
        if word in text_lower:
            score += val

    for word, val in BEARISH_WORDS.items():
        if word in text_lower:
            score += val  # val is already negative

    for word, val in GEO_EVENTS.items():
        if word in text_lower:
            score += val

    # Clamp to [-5, 5]
    return max(-5.0, min(5.0, score))


def map_to_sectors(article):
    """Determine which sectors a news article affects. Returns dict {sector: relevance}."""
    text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
    sector_hits = {}

    for sector, keywords in SECTOR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            sector_hits[sector] = min(hits / 3.0, 1.0)  # Normalize to 0-1

    return sector_hits


def is_india_relevant(article):
    """Check if article is relevant to Indian markets."""
    text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
    return any(kw in text for kw in INDIA_KEYWORDS)


# ============================================================
# MAIN PIPELINE
# ============================================================

def fetch_all_news():
    """Fetch news from all sources."""
    print("    Fetching Google News (India markets)...")
    all_articles = []

    # India-specific market news
    queries = [
        'Indian stock market today',
        'Nifty Sensex today',
        'RBI monetary policy',
        'India economy',
        'FII DII India',
    ]
    for q in queries:
        articles = fetch_google_news(q, num_results=10)
        all_articles.extend(articles)
        print(f"      {q}: {len(articles)} articles")

    # Global market news affecting India
    global_queries = [
        'US stock market',
        'Federal Reserve interest rate',
        'crude oil price',
        'China economy',
        'global recession',
        'trade war tariff',
    ]
    print("    Fetching Google News (global markets)...")
    for q in global_queries:
        articles = fetch_google_news(q, num_results=8)
        all_articles.extend(articles)
        print(f"      {q}: {len(articles)} articles")

    # Sector-specific news
    sector_queries = [
        'India banking sector HDFC ICICI',
        'India IT sector TCS Infosys',
        'crude oil ONGC BPCL India',
        'India pharma FDA approval',
        'India auto EV sales',
        'India FMCG rural demand',
        'India steel metals demand',
    ]
    print("    Fetching Google News (sectors)...")
    for q in sector_queries:
        articles = fetch_google_news(q, num_results=6)
        all_articles.extend(articles)
        print(f"      {q}: {len(articles)} articles")

    # Social sentiment from Reddit
    print("    Fetching Reddit sentiment...")
    reddit_posts = fetch_reddit_sentiment('IndianStockMarket', limit=25)
    all_articles.extend(reddit_posts)
    print(f"      Reddit: {len(reddit_posts)} posts")

    reddit_global = fetch_reddit_sentiment('wallstreetbets', limit=15)
    all_articles.extend(reddit_global)
    print(f"      WSB: {len(reddit_global)} posts")

    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        key = a.get('title', '').strip().lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    print(f"    Total unique articles: {len(unique)}")
    return unique


def compute_sentiment_scores(articles):
    """
    Compute overall market and sector-wise sentiment scores.

    Returns:
        dict with keys:
            'overall': float (-5 to +5)
            'overall_label': str (VERY BULLISH, BULLISH, NEUTRAL, BEARISH, VERY BEARISH)
            'sectors': {sector_name: {'score': float, 'label': str, 'articles': int}}
            'top_headlines': list of {title, source, sentiment, sentiment_label}
            'geo_events': list of detected geopolitical events
            'social_sentiment': float
    """
    if not articles:
        return {
            'overall': 0, 'overall_label': 'NO DATA',
            'sectors': {}, 'top_headlines': [],
            'geo_events': [], 'social_sentiment': 0,
        }

    # Score each article
    scored = []
    for a in articles:
        text = a.get('title', '') + ' ' + a.get('description', '')
        sent = analyze_sentiment(text)
        sectors = map_to_sectors(a)
        india_rel = is_india_relevant(a)
        scored.append({
            **a,
            'sentiment': sent,
            'sectors': sectors,
            'india_relevant': india_rel,
        })

    # Overall sentiment (weight India-relevant articles 2x)
    total_weight = 0
    weighted_sum = 0
    for s in scored:
        w = 2.0 if s['india_relevant'] else 1.0
        if s.get('query') == 'social':
            w *= 0.5  # Social posts weighted less
        weighted_sum += s['sentiment'] * w
        total_weight += w

    overall = weighted_sum / total_weight if total_weight > 0 else 0
    overall = max(-5, min(5, overall))

    # Overall label
    if overall > 2: label = 'VERY BULLISH'
    elif overall > 0.5: label = 'BULLISH'
    elif overall > -0.5: label = 'NEUTRAL'
    elif overall > -2: label = 'BEARISH'
    else: label = 'VERY BEARISH'

    # Sector scores
    sector_scores = defaultdict(lambda: {'sum': 0, 'weight': 0, 'count': 0})
    for s in scored:
        for sector, relevance in s['sectors'].items():
            sector_scores[sector]['sum'] += s['sentiment'] * relevance
            sector_scores[sector]['weight'] += relevance
            sector_scores[sector]['count'] += 1

    sectors = {}
    for sector in SECTOR_KEYWORDS.keys():
        ss = sector_scores.get(sector, {'sum': 0, 'weight': 0, 'count': 0})
        score = ss['sum'] / ss['weight'] if ss['weight'] > 0 else 0
        score = max(-5, min(5, score))
        if score > 1: sl = 'BULLISH'
        elif score > 0.2: sl = 'MILDLY BULLISH'
        elif score > -0.2: sl = 'NEUTRAL'
        elif score > -1: sl = 'MILDLY BEARISH'
        else: sl = 'BEARISH'
        sectors[sector] = {'score': round(score, 2), 'label': sl, 'articles': ss['count']}

    # Top headlines sorted by |sentiment|
    top = sorted(scored, key=lambda x: abs(x['sentiment']), reverse=True)[:15]
    top_headlines = []
    for t in top:
        s = t['sentiment']
        if s > 1: hl = 'BULLISH'
        elif s > 0: hl = 'MILDLY BULLISH'
        elif s < -1: hl = 'BEARISH'
        elif s < 0: hl = 'MILDLY BEARISH'
        else: hl = 'NEUTRAL'
        top_headlines.append({
            'title': t.get('title', ''),
            'source': t.get('source', ''),
            'sentiment': round(s, 2),
            'sentiment_label': hl,
        })

    # Detect geopolitical events
    geo_detected = []
    all_text = ' '.join(a.get('title', '') for a in articles).lower()
    for event, impact in GEO_EVENTS.items():
        if event in all_text and abs(impact) >= 1:
            geo_detected.append({
                'event': event,
                'impact': impact,
                'type': 'bullish' if impact > 0 else 'bearish',
            })

    # Social sentiment
    social = [s for s in scored if s.get('query') == 'social']
    social_avg = sum(s['sentiment'] for s in social) / len(social) if social else 0

    return {
        'overall': round(overall, 2),
        'overall_label': label,
        'sectors': sectors,
        'top_headlines': top_headlines,
        'geo_events': geo_detected,
        'social_sentiment': round(social_avg, 2),
        'total_articles': len(scored),
    }


def get_news_sentiment():
    """
    Main entry point. Fetches news + social media, returns sentiment analysis.
    Called by run_prediction.py.
    """
    print("\n  --- News & Social Sentiment Analysis ---")
    articles = fetch_all_news()
    results = compute_sentiment_scores(articles)

    # Print summary
    print(f"\n    Overall Sentiment: {results['overall']:+.2f} ({results['overall_label']})")
    print(f"    Social Sentiment:  {results['social_sentiment']:+.2f}")
    print(f"    Articles analyzed: {results['total_articles']}")

    if results['geo_events']:
        print(f"    ⚠ Geopolitical events detected:")
        for ge in results['geo_events']:
            print(f"      {'🔴' if ge['type']=='bearish' else '🟢'} {ge['event'].upper()} (impact: {ge['impact']:+.1f})")

    print(f"\n    Sector Sentiment:")
    for sector, data in results['sectors'].items():
        if data['articles'] > 0:
            emoji = '🟢' if data['score'] > 0.2 else ('🔴' if data['score'] < -0.2 else '🟡')
            print(f"      {emoji} {sector:30s} {data['score']:+.2f} ({data['label']}) [{data['articles']} articles]")

    print(f"\n    Top Headlines:")
    for h in results['top_headlines'][:8]:
        emoji = '🟢' if h['sentiment'] > 0.5 else ('🔴' if h['sentiment'] < -0.5 else '🟡')
        print(f"      {emoji} [{h['sentiment']:+.1f}] {h['title'][:80]}")

    return results


if __name__ == '__main__':
    results = get_news_sentiment()
    print(f"\n{'='*60}")
    print(json.dumps(results, indent=2, default=str))
