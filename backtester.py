"""
Backtrader-based Professional Backtesting Engine
=================================================
Uses backtrader framework for realistic simulation including:
- Commission, slippage, position sizing
- Multiple strategies
- Detailed trade logs and performance analytics
"""

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import INITIAL_CAPITAL, COMMISSION_PCT


# ============================================================
# STRATEGIES (backtrader compatible)
# ============================================================
class EMACrossStrategy(bt.Strategy):
    """EMA 9/21 Crossover — our best-performing strategy from backtest."""
    params = dict(fast=9, slow=21)

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(period=self.p.fast)
        self.ema_slow = bt.indicators.EMA(period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

    def next(self):
        if self.crossover > 0 and not self.position:
            self.buy(size=int(self.broker.getcash() * 0.95 / self.data.close[0]))
        elif self.crossover < 0 and self.position:
            self.close()


class MACDStrategy(bt.Strategy):
    """MACD Crossover Strategy."""
    def __init__(self):
        self.macd = bt.indicators.MACD()
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if self.crossover > 0 and not self.position:
            self.buy(size=int(self.broker.getcash() * 0.95 / self.data.close[0]))
        elif self.crossover < 0 and self.position:
            self.close()


class RSIStrategy(bt.Strategy):
    """RSI Mean Reversion Strategy."""
    params = dict(period=14, oversold=30, overbought=70)

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.p.period)

    def next(self):
        if self.rsi < self.p.oversold and not self.position:
            self.buy(size=int(self.broker.getcash() * 0.95 / self.data.close[0]))
        elif self.rsi > self.p.overbought and self.position:
            self.close()


class SupertrendStrategy(bt.Strategy):
    """Supertrend + ADX Strategy."""
    params = dict(atr_period=10, multiplier=3, adx_period=14, adx_threshold=25)

    def __init__(self):
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period=self.p.adx_period)
        self.hl2 = (self.data.high + self.data.low) / 2
        self.direction = 0  # 1=bullish, -1=bearish

    def next(self):
        hl2 = (self.data.high[0] + self.data.low[0]) / 2
        upper = hl2 + self.p.multiplier * self.atr[0]
        lower = hl2 - self.p.multiplier * self.atr[0]

        if self.data.close[0] > upper:
            new_dir = 1
        elif self.data.close[0] < lower:
            new_dir = -1
        else:
            new_dir = self.direction

        if new_dir == 1 and self.direction == -1 and self.adx[0] > self.p.adx_threshold:
            if not self.position:
                self.buy(size=int(self.broker.getcash() * 0.95 / self.data.close[0]))
        elif new_dir == -1 and self.direction == 1:
            if self.position:
                self.close()

        self.direction = new_dir


STRATEGIES = {
    'EMA Crossover (9/21)': EMACrossStrategy,
    'MACD': MACDStrategy,
    'RSI (14)': RSIStrategy,
    'Supertrend + ADX': SupertrendStrategy,
}


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_single_backtest(df, strategy_class, strategy_name, initial_capital=INITIAL_CAPITAL):
    """Run a single backtest using backtrader."""
    cerebro = bt.Cerebro()

    # Data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index
        open='Open', high='High', low='Low', close='Close', volume='Volume',
    )
    cerebro.adddata(data)

    # Strategy
    cerebro.addstrategy(strategy_class)

    # Broker config
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=COMMISSION_PCT / 100)

    # Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=0.06, annualize=True)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
    cerebro.addanalyzer(btanalyzers.VWR, _name='vwr')

    # Run
    results = cerebro.run()
    strat = results[0]

    # Extract results
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    # Trade stats
    total_trades = trades.get('total', {}).get('total', 0)
    won = trades.get('won', {}).get('total', 0)
    lost = trades.get('lost', {}).get('total', 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0

    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    profit_factor = abs(avg_win * won / (avg_loss * lost)) if lost > 0 and avg_loss != 0 else float('inf')

    # CAGR
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        'strategy': strategy_name,
        'initial_capital': initial_capital,
        'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'sharpe_ratio': round(sharpe.get('sharperatio', 0) or 0, 2),
        'max_drawdown_pct': round(dd.get('max', {}).get('drawdown', 0) or 0, 2),
        'total_trades': total_trades,
        'win_rate_pct': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'sqn': round(sqn.get('sqn', 0) or 0, 2),
        'buy_hold_return_pct': round(((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100, 2),
    }


def run_all_backtests(instruments_data):
    """
    Run all strategies across all instruments.
    instruments_data: dict of {name: DataFrame}
    Returns: dict of results.
    """
    all_results = {}

    for strat_name, strat_class in STRATEGIES.items():
        all_results[strat_name] = {}
        for inst_name, df in instruments_data.items():
            try:
                result = run_single_backtest(df, strat_class, strat_name)
                all_results[strat_name][inst_name] = result
                ret = result['total_return_pct']
                wr = result['win_rate_pct']
                print(f"    {strat_name:<22} | {inst_name:<14} | Return: {ret:+.1f}% | WR: {wr:.0f}% | Trades: {result['total_trades']}")
            except Exception as e:
                print(f"    {strat_name:<22} | {inst_name:<14} | ERROR: {e}")

    return all_results
