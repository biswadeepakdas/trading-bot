<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

// Yahoo Finance v8 API for live prices
function fetchQuotes($symbols) {
    $joined = implode(',', $symbols);
    $url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" . urlencode($joined);
    $ctx = stream_context_create([
        'http' => [
            'timeout' => 15,
            'header' => "User-Agent: Mozilla/5.0\r\n"
        ]
    ]);
    $raw = @file_get_contents($url, false, $ctx);
    if (!$raw) return [];
    $json = json_decode($raw, true);
    if (!isset($json['quoteResponse']['result'])) return [];
    $out = [];
    foreach ($json['quoteResponse']['result'] as $q) {
        $sym = $q['symbol'] ?? '';
        $out[$sym] = [
            'price' => $q['regularMarketPrice'] ?? 0,
            'change' => $q['regularMarketChange'] ?? 0,
            'changePct' => $q['regularMarketChangePercent'] ?? 0,
            'prevClose' => $q['regularMarketPreviousClose'] ?? 0,
            'name' => $q['shortName'] ?? $sym
        ];
    }
    return $out;
}

// All symbols grouped
$global = [
    '^GSPC' => 'S&P 500', '^IXIC' => 'NASDAQ', '^DJI' => 'Dow Jones', '^RUT' => 'Russell 2000',
    '^FTSE' => 'FTSE 100', '^GDAXI' => 'DAX', '^FCHI' => 'CAC 40',
    '^N225' => 'Nikkei 225', '^TOPX' => 'TOPIX', '^HSI' => 'Hang Seng',
    '000001.SS' => 'Shanghai', '399001.SZ' => 'Shenzhen', '000300.SS' => 'CSI 300', '^KS11' => 'KOSPI'
];

$commodities = [
    'CL=F' => 'Crude Oil WTI', 'BZ=F' => 'Brent Crude', 'GC=F' => 'Gold',
    'SI=F' => 'Silver', 'HG=F' => 'Copper', 'NG=F' => 'Natural Gas',
    'USDINR=X' => 'USD/INR', '^TNX' => 'US 10Y Yield', 'DX-Y.NYB' => 'Dollar Index'
];

$vix = ['^INDIAVIX' => 'India VIX', '^VIX' => 'US VIX'];

$broad = ['^NSEI' => 'NIFTY 50', '^NSEBANK' => 'BANKNIFTY'];

$sectors = [
    'Banking' => ['HDFCBANK.NS','ICICIBANK.NS','KOTAKBANK.NS','SBIN.NS','BAJFINANCE.NS'],
    'IT' => ['TCS.NS','INFY.NS','WIPRO.NS','HCLTECH.NS','TECHM.NS'],
    'Oil_Upstream' => ['ONGC.NS','OIL.NS','RELIANCE.NS'],
    'Oil_Downstream' => ['BPCL.NS','IOC.NS','HINDPETRO.NS'],
    'Metals' => ['TATASTEEL.NS','HINDALCO.NS','JSWSTEEL.NS','COALINDIA.NS'],
    'Pharma' => ['SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','DIVISLAB.NS'],
    'Auto' => ['TATAMOTORS.NS','M&M.NS','MARUTI.NS','BAJAJ-AUTO.NS'],
    'FMCG' => ['HINDUNILVR.NS','ITC.NS','NESTLEIND.NS','BRITANNIA.NS']
];

$china_japan = [
    '7203.T' => 'Toyota', '6758.T' => 'Sony', '9984.T' => 'SoftBank',
    '6861.T' => 'Keyence', '7974.T' => 'Nintendo',
    '9988.HK' => 'Alibaba', '1211.HK' => 'BYD', '3690.HK' => 'Meituan',
    '9618.HK' => 'JD.com', '0700.HK' => 'Tencent'
];

// Collect all symbols
$allSymbols = array_merge(
    array_keys($global), array_keys($commodities), array_keys($vix),
    array_keys($broad), array_keys($china_japan)
);
foreach ($sectors as $stocks) {
    $allSymbols = array_merge($allSymbols, $stocks);
}

// Batch fetch (20 per request)
$allData = [];
$batches = array_chunk($allSymbols, 20);
foreach ($batches as $batch) {
    $allData = array_merge($allData, fetchQuotes($batch));
}

// Build response
function mapData($mapping, &$allData) {
    $out = [];
    foreach ($mapping as $sym => $name) {
        $d = $allData[$sym] ?? null;
        $out[] = [
            'name' => $name, 'symbol' => $sym,
            'price' => $d ? $d['price'] : null,
            'change' => $d ? $d['change'] : null,
            'changePct' => $d ? $d['changePct'] : null
        ];
    }
    return $out;
}

$response = [
    'timestamp' => date('c'),
    'global' => mapData($global, $allData),
    'commodities' => mapData($commodities, $allData),
    'vix' => mapData($vix, $allData),
    'broad' => mapData($broad, $allData),
    'china_japan' => mapData($china_japan, $allData),
    'sectors' => []
];

$sectorNames = [
    'Banking' => 'Banking & Financials', 'IT' => 'IT & Technology',
    'Oil_Upstream' => 'Oil Upstream', 'Oil_Downstream' => 'Oil Downstream',
    'Metals' => 'Metals & Mining', 'Pharma' => 'Pharma & Healthcare',
    'Auto' => 'Auto & EV', 'FMCG' => 'FMCG & Consumer'
];

foreach ($sectors as $key => $stocks) {
    $sectorStocks = [];
    $totalChg = 0; $count = 0;
    foreach ($stocks as $sym) {
        $d = $allData[$sym] ?? null;
        $name = str_replace('.NS', '', $sym);
        $sectorStocks[] = [
            'name' => $name, 'symbol' => $sym,
            'price' => $d ? $d['price'] : null,
            'changePct' => $d ? $d['changePct'] : null
        ];
        if ($d) { $totalChg += $d['changePct']; $count++; }
    }
    $avgChg = $count > 0 ? $totalChg / $count : 0;
    $response['sectors'][] = [
        'name' => $sectorNames[$key] ?? $key,
        'key' => $key,
        'avgChange' => round($avgChg, 2),
        'stocks' => $sectorStocks
    ];
}

echo json_encode($response, JSON_PRETTY_PRINT);
