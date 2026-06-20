#!/usr/bin/env python3
"""
ETF Relative Strength Ranking & Backtest System
================================================
Strategy:
- Every Friday EOD: Calculate Relative Strength scores for all ETFs
- Select top 3 ETFs by RS score
- Buy at Monday closing price (proxy for Monday open)
- Hold until next rebalance
- Dynamic lookback period optimization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ============================================================
# ETF DEFINITIONS
# ============================================================

ETFS = [
    ("GOLDBEES.NS",   "GoldBees",      "GOLD"),
    ("SILVERBEES.NS", "SilverBees",    "SILV"),
    ("NIFTYBEES.NS",  "NiftyBees",     "NFTY"),
    ("JUNIORBEES.NS", "JuniorBees",    "JNBR"),
    ("MID150BEES.NS", "Midcap150",     "MIDM"),
    ("NIF100BEES.NS", "Nifty100",      "NF10"),
    ("BANKBEES.NS",   "BankBees",      "BANK"),
    ("ITBEES.NS",     "ITBees",        "ITMC"),
    ("PHARMABEES.NS", "PharmaBees",    "PHRM"),
    ("AUTOBEES.NS",   "AutoBees",      "AUTO"),
    ("INFRABEES.NS",  "InfraBees",     "INFR"),
    ("CONSUMBEES.NS", "ConsumeBees",   "CNSM"),
    ("PSUBNKBEES.NS", "PSUBankBees",   "PSUB"),
    ("CPSEETF.NS",    "CPSE ETF",      "CETF"),
    ("LTGILTBEES.NS", "LT Gilt",       "GSCP"),
    ("GILT5YBEES.NS", "GSec 5Y",       "GS5Y"),
    ("LIQUIDBEES.NS", "LiquidBees",    "LIQD"),
    ("MOM100.NS",     "Momentum100",   "MOM" ),
    ("MOMENTUM30.NS", "Momentum30",    "MOM3"),
    ("NV20BEES.NS",   "Value20",       "NV20"),
    ("DIVOPPBEES.NS", "DivOpp",        "DIVO"),
    ("HNGSNGBEES.NS", "HangSeng",      "HNGS"),
    ("MAFANG.NS",     "FANGPlus",      "FANG"),
    ("MON100.NS",     "Nasdaq100",     "NSDQ"),
]

TICKERS = [e[0] for e in ETFS]
NAMES = {e[0]: e[1] for e in ETFS}
CODES = {e[0]: e[2] for e in ETFS}

# ============================================================
# DATA MANAGEMENT
# ============================================================

def download_data(cache_file='etf_prices.csv', force_download=False):
    if os.path.exists(cache_file) and not force_download:
        print('Loading cached data from ' + cache_file + '...')
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print('Loaded data shape: ' + str(prices.shape))
        dr = str(prices.index[0].date()) + ' to ' + str(prices.index[-1].date())
        print('Date range: ' + dr)
        return prices
    
    print('Downloading historical data for all ETFs...')
    print('This may take 2-3 minutes...')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=12*365)
    
    data = yf.download(TICKERS, start=start_date, end=end_date, progress=True, auto_adjust=True, threads=True)
    
    if hasattr(data.columns, 'levels'):
        close_prices = data['Close']
    else:
        close_prices = data
    
    close_prices = close_prices.ffill().bfill()
    close_prices = close_prices.dropna(axis=1, how='all')
    close_prices = close_prices.dropna(how='all')
    
    print('Downloaded data shape: ' + str(close_prices.shape))
    dr = str(close_prices.index[0].date()) + ' to ' + str(close_prices.index[-1].date())
    print('Date range: ' + dr)
    print('Available ETFs: ' + str(len(close_prices.columns)))
    
    close_prices.to_csv(cache_file)
    print('Data cached to ' + cache_file)
    
    return close_prices


def download_single_etf(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if len(df) > 0:
                return df['Close']
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                print('Failed to download ' + ticker + ': ' + str(e))
    return None


def download_data_individual(cache_file='etf_prices.csv', force_download=False):
    if os.path.exists(cache_file) and not force_download:
        print('Loading cached data from ' + cache_file + '...')
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices
    
    print('Downloading data one ETF at a time...')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=12*365)
    
    all_data = {}
    for i, ticker in enumerate(TICKERS):
        msg = '  [' + str(i+1) + '/' + str(len(TICKERS)) + '] Downloading ' + ticker + '...'
        print(msg, end=' ')
        series = download_single_etf(ticker, start_date, end_date)
        if series is not None and len(series) > 0:
            all_data[ticker] = series
            print('OK (' + str(len(series)) + ' rows)')
        else:
            print('FAILED')
        import time
        time.sleep(0.3)
    
    if not all_data:
        raise ValueError('No data downloaded! Check internet connection.')
    
    prices = pd.DataFrame(all_data)
    prices = prices.ffill().bfill()
    prices = prices.dropna(how='all')
    
    print('Downloaded data shape: ' + str(prices.shape))
    prices.to_csv(cache_file)
    print('Data cached to ' + cache_file)
    
    return prices


# ============================================================
# RELATIVE STRENGTH CALCULATION
# ============================================================

def calculate_relative_strength(prices, lookback_period):
    rs_scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    
    for ticker in prices.columns:
        price = prices[ticker]
        if price.isna().sum() > len(price) * 0.1:
            continue
        price = price.ffill()
        
        momentum = price.pct_change(lookback_period)
        daily_returns = price.pct_change()
        rolling_vol = daily_returns.rolling(lookback_period).std() * np.sqrt(252)
        vol_adj_return = momentum / rolling_vol.replace(0, np.nan)
        positive_days = (daily_returns > 0).rolling(lookback_period).mean()
        
        rs_scores[ticker] = (
            0.50 * momentum.fillna(0) + 
            0.30 * vol_adj_return.fillna(0) + 
            0.20 * positive_days.fillna(0.5)
        )
    
    return rs_scores


def get_top_n_etfs(rs_scores, n=3, date=None):
    if date is None:
        date = rs_scores.index[-1]
    
    if date not in rs_scores.index:
        valid_dates = rs_scores.index[rs_scores.index <= date]
        if len(valid_dates) == 0:
            return [], np.array([])
        date = valid_dates[-1]
    
    scores = rs_scores.loc[date].sort_values(ascending=False)
    scores = scores.dropna()
    
    return scores.head(n).index.tolist(), scores.head(n).values


# ============================================================
# BACKTEST ENGINE
# ============================================================

class ETFBacktest:
    def __init__(self, prices, initial_capital=1000000, top_n=3, 
                 lookback_period=63, rebalance_freq='W-FRI', transaction_cost=0.001):
        self.prices = prices.copy()
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.lookback_period = lookback_period
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        
        print('Calculating RS scores (lookback=' + str(lookback_period) + ' days)...')
        self.rs_scores = calculate_relative_strength(prices, lookback_period)
        print('RS scores calculated.')
    
    def run_backtest(self, start_date=None, end_date=None):
        if start_date is None:
            start_idx = self.lookback_period + 20
            start_date = self.prices.index[start_idx]
        
        if end_date is None:
            end_date = self.prices.index[-1]
        
        mask = (self.prices.index >= start_date) & (self.prices.index <= end_date)
        prices = self.prices.loc[mask].copy()
        rs_scores = self.rs_scores.loc[mask].copy()
        
        if len(prices) == 0:
            raise ValueError('No data in specified date range!')
        
        all_dates = prices.index
        fridays = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        fridays = [f for f in fridays if f in all_dates]
        
        mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        mondays = [m for m in mondays if m in all_dates]
        
        print('Running backtest: ' + str(start_date.date()) + ' to ' + str(end_date.date()))
        print('Rebalance dates: ' + str(len(fridays)) + ' Fridays')
        print('Execution dates: ' + str(len(mondays)) + ' Mondays')
        
        cash = self.initial_capital
        holdings = {}
        value_history = []
        trades = []
        
        for i, friday in enumerate(fridays):
            if friday not in rs_scores.index:
                continue
            
            next_monday = None
            for m in mondays:
                if m > friday:
                    next_monday = m
                    break
            
            if next_monday is None or next_monday not in prices.index:
                continue
            
            top_etfs, top_scores = get_top_n_etfs(rs_scores, n=self.top_n, date=friday)
            
            if len(top_etfs) == 0:
                continue
            
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in prices.columns and friday in prices.index:
                    price = prices.loc[friday, ticker]
                    if pd.notna(price):
                        portfolio_value += shares * price
            
            for ticker, shares in list(holdings.items()):
                if ticker in prices.columns and friday in prices.index:
                    sell_price = prices.loc[friday, ticker]
                    if pd.notna(sell_price) and sell_price > 0:
                        proceeds = shares * sell_price * (1 - self.transaction_cost)
                        cash += proceeds
                        trades.append({
                            'date': friday,
                            'action': 'SELL',
                            'ticker': ticker,
                            'shares': shares,
                            'price': sell_price,
                            'proceeds': proceeds
                        })
            
            holdings = {}
            allocation = cash / len(top_etfs)
            
            for ticker in top_etfs:
                if ticker in prices.columns and next_monday in prices.index:
                    buy_price = prices.loc[next_monday, ticker]
                    if pd.notna(buy_price) and buy_price > 0:
                        cost = allocation * (1 - self.transaction_cost)
                        shares = cost / buy_price
                        holdings[ticker] = shares
                        cash -= allocation
                        trades.append({
                            'date': next_monday,
                            'action': 'BUY',
                            'ticker': ticker,
                            'shares': shares,
                            'price': buy_price,
                            'cost': cost
                        })
            
            current_value = cash
            for ticker, shares in holdings.items():
                if ticker in prices.columns and next_monday in prices.index:
                    price = prices.loc[next_monday, ticker]
                    if pd.notna(price):
                        current_value += shares * price
            
            value_history.append({
                'date': next_monday,
                'value': current_value,
                'holdings': list(top_etfs),
                'cash': cash
            })
            
            if i % 52 == 0 and i > 0:
                msg = '  Year ' + str(i//52) + ': Portfolio = Rs ' + '{:.0f}'.format(current_value)
                print(msg)
        
        return {
            'cash': cash,
            'holdings': holdings,
            'value_history': value_history,
            'trades': trades,
            'initial_capital': self.initial_capital
        }
    
    def calculate_metrics(self, portfolio):
        values = pd.DataFrame(portfolio['value_history'])
        if len(values) == 0:
            return {}
        
        values = values.set_index('date')
        values['returns'] = values['value'].pct_change()
        
        total_return = (values['value'].iloc[-1] / self.initial_capital - 1) * 100
        
        years = (values.index[-1] - values.index[0]).days / 365.25
        if years <= 0:
            years = 1
        
        ann_return = ((values['value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        
        weekly_returns = values['returns'].dropna()
        ann_vol = weekly_returns.std() * np.sqrt(52) * 100
        
        sharpe = (ann_return - 6) / ann_vol if ann_vol > 0 else 0
        
        downside_returns = weekly_returns[weekly_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else 0.0001
        sortino = (ann_return - 6) / downside_vol if downside_vol > 0 else 0
        
        cummax = values['value'].cummax()
        drawdown = (values['value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        win_rate = (weekly_returns > 0).mean() * 100
        
        avg_win = weekly_returns[weekly_returns > 0].mean() * 100 if (weekly_returns > 0).any() else 0
        avg_loss = weekly_returns[weekly_returns < 0].mean() * 100 if (weekly_returns < 0).any() else 0
        
        gross_profit = weekly_returns[weekly_returns > 0].sum()
        gross_loss = abs(weekly_returns[weekly_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        num_buys = len([t for t in portfolio['trades'] if t['action'] == 'BUY'])
        num_sells = len([t for t in portfolio['trades'] if t['action'] == 'SELL'])
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_value': values['value'].iloc[-1],
            'num_buys': num_buys,
            'num_sells': num_sells,
            'years': years,
            'values': values,
            'drawdown': drawdown
        }


# ============================================================
# DYNAMIC LOOKBACK OPTIMIZATION
# ============================================================

def optimize_lookback(prices, lookback_options=None, test_periods=None,
                      initial_capital=1000000, top_n=3):
    if lookback_options is None:
        lookback_options = [21, 42, 63, 126, 252]
    
    if test_periods is None:
        end = prices.index[-1]
        test_periods = {
            '10Y': (end - timedelta(days=10*365), end),
            '5Y': (end - timedelta(days=5*365), end),
            '3Y': (end - timedelta(days=3*365), end),
            '1Y': (end - timedelta(days=365), end),
        }
    
    results = {}
    
    for period_name, (start, end) in test_periods.items():
        if start < prices.index[0]:
            start = prices.index[0]
            print('')
            print(period_name + ': Adjusted start to ' + str(start.date()) + ' (data limit)')
        else:
            print('')
            print('='*60)
            print('PERIOD: ' + period_name + ' (' + str(start.date()) + ' to ' + str(end.date()) + ')')
            print('='*60)
        
        period_results = {}
        
        for lookback in lookback_options:
            min_required_days = lookback + 30
            if (end - start).days < min_required_days:
                print('  Lookback ' + str(lookback) + ': Skipped (insufficient data)')
                continue
            
            print('')
            msg = '  Testing lookback: ' + str(lookback) + ' days (~' + str(int(lookback/21)) + ' months)'
            print(msg)
            
            try:
                bt = ETFBacktest(prices, lookback_period=lookback, initial_capital=initial_capital, top_n=top_n)
                portfolio = bt.run_backtest(start_date=start, end_date=end)
                metrics = bt.calculate_metrics(portfolio)
                
                period_results[lookback] = metrics
                
                if metrics:
                    print('    Ann Return: ' + '{:>8.2f}'.format(metrics['annualized_return']) + '%')
                    print('    Sharpe:     ' + '{:>8.2f}'.format(metrics['sharpe_ratio']))
                    print('    Max DD:     ' + '{:>8.2f}'.format(metrics['max_drawdown']) + '%')
                    print('    Final Val:  Rs ' + '{:>12,.0f}'.format(metrics['final_value']))
            
            except Exception as e:
                print('    Error: ' + str(e))
                period_results[lookback] = None
        
        valid_results = {k: v for k, v in period_results.items() if v and 'sharpe_ratio' in v}
        
        if valid_results:
            best_lookback = max(valid_results.items(),
                               key=lambda x: (x[1]['sharpe_ratio'], x[1]['annualized_return']))
            
            results[period_name] = {
                'best_lookback': best_lookback[0],
                'best_metrics': best_lookback[1],
                'all_results': period_results
            }
            
            print('')
            print('  ' + '*'*50)
            print('  BEST LOOKBACK for ' + period_name + ': ' + str(best_lookback[0]) + ' days')
            print('  Sharpe Ratio: ' + '{:.2f}'.format(best_lookback[1]['sharpe_ratio']))
            print('  Ann Return:   ' + '{:.2f}'.format(best_lookback[1]['annualized_return']) + '%')
            print('  Max Drawdown: ' + '{:.2f}'.format(best_lookback[1]['max_drawdown']) + '%')
            print('  ' + '*'*50)
        else:
            print('')
            print('  No valid results for ' + period_name)
    
    return results


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

def download_benchmark(ticker='^NSEI', start_date=None, end_date=None):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        return data['Close']
    except Exception as e:
        print('Could not download benchmark ' + ticker + ': ' + str(e))
        return None


def compare_with_benchmark(strategy_metrics, benchmark_prices):
    if benchmark_prices is None or len(benchmark_prices) == 0:
        print('Benchmark data not available for comparison.')
        return
    
    strategy_values = strategy_metrics['values']['value']
    benchmark_aligned = benchmark_prices.reindex(strategy_values.index, method='ffill')
    
    benchmark_returns = benchmark_aligned.pct_change().dropna()
    
    years = strategy_metrics['years']
    bench_total = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1) * 100
    bench_ann = ((benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) ** (1/years) - 1) * 100
    bench_vol = benchmark_returns.std() * np.sqrt(52) * 100
    bench_sharpe = (bench_ann - 6) / bench_vol if bench_vol > 0 else 0
    
    bench_cummax = benchmark_aligned.cummax()
    bench_dd = (benchmark_aligned - bench_cummax) / bench_cummax
    bench_max_dd = bench_dd.min() * 100
    
    ann_ret = strategy_metrics['annualized_return']
    ann_vol = strategy_metrics['annualized_volatility']
    sharpe = strategy_metrics['sharpe_ratio']
    max_dd = strategy_metrics['max_drawdown']
    
    print('')
    print('='*60)
    print('STRATEGY vs BENCHMARK (Nifty 50)')
    print('='*60)
    print('{:25} {:>12} {:>12} {:>12}'.format('Metric', 'Strategy', 'Benchmark', 'Diff'))
    print('-'*60)
    print('{:25} {:>12.2f} {:>12.2f} {:>+12.2f}'.format('Ann Return (%)', ann_ret, bench_ann, ann_ret-bench_ann))
    print('{:25} {:>12.2f} {:>12.2f} {:>+12.2f}'.format('Volatility (%)', ann_vol, bench_vol, ann_vol-bench_vol))
    print('{:25} {:>12.2f} {:>12.2f} {:>+12.2f}'.format('Sharpe Ratio', sharpe, bench_sharpe, sharpe-bench_sharpe))
    print('{:25} {:>12.2f} {:>12.2f} {:>+12.2f}'.format('Max Drawdown (%)', max_dd, bench_max_dd, max_dd-bench_max_dd))
    print('-'*60)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(results, benchmark_prices=None, save_path='etf_backtest_equity.png'):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print('matplotlib not installed. Skipping charts.')
        print('Install with: pip install matplotlib')
        return
    
    n_periods = len(results)
    if n_periods == 0:
        print('No results to plot.')
        return
    
    fig = plt.figure(figsize=(18, 4 * n_periods + 4))
    
    for idx, (period_name, result) in enumerate(results.items()):
        metrics = result['best_metrics']
        values = metrics['values']
        
        normalized = values['value'] / values['value'].iloc[0] * 100
        
        ax1 = plt.subplot(n_periods + 1, 2, 2*idx + 1)
        ax1.plot(normalized.index, normalized.values, linewidth=2, color='#2E86AB', label='Strategy')
        
        if benchmark_prices is not None:
            bench_aligned = benchmark_prices.reindex(values.index, method='ffill')
            bench_norm = bench_aligned / bench_aligned.iloc[0] * 100
            ax1.plot(bench_norm.index, bench_norm.values, linewidth=1.5,
                    color='#A23B72', linestyle='--', alpha=0.7, label='Nifty 50')
        
        title_str = (period_name + ' - Lookback: ' + str(result['best_lookback']) + 'd | ' +
                    'CAGR: ' + '{:.1f}'.format(metrics['annualized_return']) + '% | ' +
                    'Sharpe: ' + '{:.2f}'.format(metrics['sharpe_ratio']))
        ax1.set_title(title_str, fontsize=11, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Base 100)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        ax2 = plt.subplot(n_periods + 1, 2, 2*idx + 2)
        dd_pct = metrics['drawdown'] * 100
        ax2.fill_between(dd_pct.index, dd_pct.values, 0, color='#F18F01', alpha=0.3)
        ax2.plot(dd_pct.index, dd_pct.values, color='#C73E1D', linewidth=1)
        title2 = 'Drawdown | Max: ' + '{:.1f}'.format(metrics['max_drawdown']) + '%'
        ax2.set_title(title2, fontsize=11)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax3 = plt.subplot(n_periods + 1, 1, n_periods + 1)
    ax3.axis('off')
    
    table_data = []
    headers = ['Period', 'Lookback', 'CAGR%', 'Sharpe', 'MaxDD%', 'Win%', 'Final Rs']
    table_data.append(headers)
    
    for period_name, result in results.items():
        m = result['best_metrics']
        row = [
            period_name,
            str(result['best_lookback']) + 'd',
            '{:.1f}'.format(m['annualized_return']),
            '{:.2f}'.format(m['sharpe_ratio']),
            '{:.1f}'.format(m['max_drawdown']),
            '{:.1f}'.format(m['win_rate']),
            'Rs ' + '{:,.0f}'.format(m['final_value'])
        ]
        table_data.append(row)
    
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colColours=['#2E86AB']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(7):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.suptitle('ETF Relative Strength Momentum Backtest Results',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print('Charts saved to: ' + save_path)
    plt.close()


def plot_lookback_comparison(results, save_path='lookback_comparison.png'):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (period_name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        lookbacks = []
        returns = []
        sharpes = []
        
        for lb, metrics in result['all_results'].items():
            if metrics and 'annualized_return' in metrics:
                lookbacks.append(lb)
                returns.append(metrics['annualized_return'])
                sharpes.append(metrics['sharpe_ratio'])
        
        if not lookbacks:
            continue
        
        x = np.arange(len(lookbacks))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, returns, width, label='CAGR %', color='#2E86AB')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, sharpes, width, label='Sharpe', color='#F18F01')
        
        ax.set_xlabel('Lookback Period (days)')
        ax.set_ylabel('CAGR (%)', color='#2E86AB')
        ax2.set_ylabel('Sharpe Ratio', color='#F18F01')
        ax.set_title(period_name + ': Lookback Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([str(lb) + 'd' for lb in lookbacks])
        ax.grid(True, alpha=0.3, axis='y')
        
        best_idx = lookbacks.index(result['best_lookback'])
        bars1[best_idx].set_color('#1B4965')
        bars2[best_idx].set_color('#C73E1D')
    
    plt.suptitle('Lookback Period Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print('Lookback comparison saved to: ' + save_path)
    plt.close()


# ============================================================
# EXPORT RESULTS
# ============================================================

def export_results(results, filename='backtest_results.json'):
    export_data = {}
    
    for period_name, result in results.items():
        export_data[period_name] = {
            'best_lookback': result['best_lookback'],
            'metrics': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in result['best_metrics'].items()
                if k not in ['values', 'drawdown']
            }
        }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print('Results exported to: ' + filename)


def print_detailed_results(results):
    print('')
    print('='*80)
    print('COMPLETE BACKTEST RESULTS SUMMARY')
    print('='*80)
    
    for period_name, result in results.items():
        m = result['best_metrics']
        lb = result['best_lookback']
        
        print('')
        print('-'*80)
        header = 'PERIOD: ' + period_name + ' | OPTIMAL LOOKBACK: ' + str(lb) + ' days (~' + str(int(lb/21)) + ' months)'
        print(header)
        print('-'*80)
        print('  {:30} {:>20}'.format('Metric', 'Value'))
        print('  ' + '-'*50)
        print('  {:30} {:>20}'.format('Initial Capital', 'Rs 1,000,000'))
        print('  {:30} {:>20}'.format('Final Portfolio Value', 'Rs ' + '{:,.0f}'.format(m['final_value'])))
        print('  {:30} {:>20}'.format('Total Return', '{:.2f}'.format(m['total_return']) + '%'))
        print('  {:30} {:>20}'.format('CAGR (Annualized Return)', '{:.2f}'.format(m['annualized_return']) + '%'))
        print('  {:30} {:>20}'.format('Annualized Volatility', '{:.2f}'.format(m['annualized_volatility']) + '%'))
        print('  {:30} {:>20}'.format('Sharpe Ratio', '{:.2f}'.format(m['sharpe_ratio'])))
        print('  {:30} {:>20}'.format('Sortino Ratio', '{:.2f}'.format(m['sortino_ratio'])))
        print('  {:30} {:>20}'.format('Calmar Ratio', '{:.2f}'.format(m['calmar_ratio'])))
        print('  {:30} {:>20}'.format('Maximum Drawdown', '{:.2f}'.format(m['max_drawdown']) + '%'))
        print('  {:30} {:>20}'.format('Win Rate (Weekly)', '{:.1f}'.format(m['win_rate']) + '%'))
        print('  {:30} {:>20}'.format('Average Winning Week', '{:.2f}'.format(m['avg_win']) + '%'))
        print('  {:30} {:>20}'.format('Average Losing Week', '{:.2f}'.format(m['avg_loss']) + '%'))
        print('  {:30} {:>20}'.format('Profit Factor', '{:.2f}'.format(m['profit_factor'])))
        print('  {:30} {:>20}'.format('Number of Buy Trades', str(m['num_buys'])))
        print('  {:30} {:>20}'.format('Number of Sell Trades', str(m['num_sells'])))
        print('  {:30} {:>20}'.format('Backtest Duration', '{:.1f}'.format(m['years']) + ' years'))


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print('='*80)
    print('ETF RELATIVE STRENGTH MOMENTUM BACKTEST SYSTEM')
    print('='*80)
    print('Start Time: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('ETFs: ' + str(len(TICKERS)) + ' instruments')
    print('Strategy: Top 3 ETFs by RS Score | Friday Signal | Monday Execution')
    print('Capital: Rs 10,00,000')
    print('='*80)
    
    try:
        prices = download_data(cache_file='etf_prices.csv')
    except Exception as e:
        print('Batch download failed: ' + str(e))
        print('Trying individual downloads...')
        prices = download_data_individual(cache_file='etf_prices.csv')
    
    print('')
    print('Data Summary:')
    print('  ETFs with data: ' + str(len(prices.columns)))
    dr = str(prices.index[0].date()) + ' to ' + str(prices.index[-1].date())
    print('  Date range: ' + dr)
    print('  Total trading days: ' + str(len(prices)))
    
    lookback_options = [21, 42, 63, 126, 252]
    
    results = optimize_lookback(
        prices,
        lookback_options=lookback_options,
        initial_capital=1000000,
        top_n=3
    )
    
    print_detailed_results(results)
    
    print('')
    print('='*80)
    print('BENCHMARK COMPARISON')
    print('='*80)
    
    benchmark = download_benchmark('^NSEI')
    
    for period_name, result in results.items():
        print('')
        print('--- ' + period_name + ' ---')
        compare_with_benchmark(result['best_metrics'], benchmark)
    
    print('')
    print('='*80)
    print('GENERATING CHARTS')
    print('='*80)
    
    plot_results(results, benchmark_prices=benchmark, save_path='etf_backtest_equity.png')
    plot_lookback_comparison(results, save_path='lookback_comparison.png')
    
    export_results(results, 'backtest_results.json')
    
    print('')
    print('='*80)
    print('CURRENT SIGNAL (Latest Available Data)')
    print('='*80)
    
    if '1Y' in results:
        best_lb = results['1Y']['best_lookback']
    else:
        best_lb = 63
    
    final_rs = calculate_relative_strength(prices, best_lb)
    top3, scores = get_top_n_etfs(final_rs, n=3)
    
    print('')
    print('Using lookback: ' + str(best_lb) + ' days')
    print('')
    print('{:6} {:15} {:20} {:>12}'.format('Rank', 'Ticker', 'Name', 'RS Score'))
    print('-' * 55)
    for i, (ticker, score) in enumerate(zip(top3, scores), 1):
        name = NAMES.get(ticker, ticker)
        print('{:6} {:15} {:20} {:>12.4f}'.format(i, ticker, name, score))
    
    print('')
    print('='*80)
    print('End Time: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('='*80)
    
    return results


if __name__ == '__main__':
    results = main()