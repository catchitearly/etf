#!/usr/bin/env python3
"""
ETF Relative Strength Ranking & Backtest System
================================================
Strategy:
- Every Friday EOD: Calculate Relative Strength scores for all ETFs
- Select top 3 ETFs by RS score
- Buy at Monday's closing price (proxy for Monday open)
- Hold until next rebalance
- Dynamic lookback period optimization

Author: Generated for ETF Momentum Strategy
Date: 2026-06-20
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

def download_data(cache_file="etf_prices.csv", force_download=False):
    """
    Download historical data for all ETFs.
    Uses caching to avoid repeated downloads.
    """
    if os.path.exists(cache_file) and not force_download:
        print(f"Loading cached data from {cache_file}...")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded data shape: {prices.shape}")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        return prices

    print("Downloading historical data for all ETFs...")
    print("This may take 2-3 minutes...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=12*365)  # 12 years back

    # Download all at once
    data = yf.download(
        TICKERS, 
        start=start_date, 
        end=end_date, 
        progress=True, 
        auto_adjust=True,
        threads=True
    )

    # Extract Close prices
    if hasattr(data.columns, 'levels'):
        close_prices = data['Close']
    else:
        close_prices = data

    # Forward fill missing values, then backward fill any remaining
    close_prices = close_prices.ffill().bfill()

    # Remove columns with all NaN
    close_prices = close_prices.dropna(axis=1, how='all')

    # Remove rows with all NaN
    close_prices = close_prices.dropna(how='all')

    print(f"\nDownloaded data shape: {close_prices.shape}")
    print(f"Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print(f"Available ETFs: {len(close_prices.columns)}")

    # Save to cache
    close_prices.to_csv(cache_file)
    print(f"Data cached to {cache_file}")

    return close_prices


def download_single_etf(ticker, start_date, end_date, max_retries=3):
    """Download data for a single ETF with retry logic."""
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                auto_adjust=True
            )
            if len(df) > 0:
                return df['Close']
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {ticker}: {e}")
    return None


def download_data_individual(cache_file="etf_prices.csv", force_download=False):
    """
    Download data one ETF at a time (slower but avoids rate limits).
    """
    if os.path.exists(cache_file) and not force_download:
        print(f"Loading cached data from {cache_file}...")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices

    print("Downloading data one ETF at a time...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=12*365)

    all_data = {}
    for i, ticker in enumerate(TICKERS):
        print(f"  [{i+1}/{len(TICKERS)}] Downloading {ticker}...", end=" ")
        series = download_single_etf(ticker, start_date, end_date)
        if series is not None and len(series) > 0:
            all_data[ticker] = series
            print(f"✓ ({len(series)} rows)")
        else:
            print("✗ (failed)")

        import time
        time.sleep(0.3)  # Small delay

    if not all_data:
        raise ValueError("No data downloaded! Check internet connection.")

    # Combine into DataFrame
    prices = pd.DataFrame(all_data)
    prices = prices.ffill().bfill()
    prices = prices.dropna(how='all')

    print(f"\nDownloaded data shape: {prices.shape}")
    prices.to_csv(cache_file)
    print(f"Data cached to {cache_file}")

    return prices


# ============================================================
# RELATIVE STRENGTH CALCULATION
# ============================================================

def calculate_relative_strength(prices, lookback_period):
    """
    Calculate composite Relative Strength score for each ETF.

    Components:
    1. Momentum (50%): Raw price return over lookback period
    2. Risk-Adjusted Return (30%): Return / Volatility
    3. Consistency (20%): % of positive days

    Parameters:
    -----------
    prices : DataFrame
        Historical close prices
    lookback_period : int
        Number of trading days to look back

    Returns:
    --------
    DataFrame : RS scores for each ETF on each date
    """
    rs_scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ticker in prices.columns:
        price = prices[ticker]

        # Skip if insufficient data (>10% missing)
        if price.isna().sum() > len(price) * 0.1:
            continue

        # Fill any remaining NaN with forward fill
        price = price.ffill()

        # 1. Momentum: Price return over lookback
        momentum = price.pct_change(lookback_period)

        # 2. Risk-Adjusted Return (Sharpe-like)
        daily_returns = price.pct_change()
        rolling_vol = daily_returns.rolling(lookback_period).std() * np.sqrt(252)
        vol_adj_return = momentum / rolling_vol.replace(0, np.nan)

        # 3. Consistency: % of positive days
        positive_days = (daily_returns > 0).rolling(lookback_period).mean()

        # Composite score with weights
        rs_scores[ticker] = (
            0.50 * momentum.fillna(0) + 
            0.30 * vol_adj_return.fillna(0) + 
            0.20 * positive_days.fillna(0.5)
        )

    return rs_scores


def get_top_n_etfs(rs_scores, n=3, date=None):
    """
    Get top N ETFs by relative strength score on a given date.

    Parameters:
    -----------
    rs_scores : DataFrame
        RS scores for all ETFs
    n : int
        Number of top ETFs to select
    date : datetime or None
        Date to check. If None, uses last available date.

    Returns:
    --------
    tuple : (list of top N tickers, array of their scores)
    """
    if date is None:
        date = rs_scores.index[-1]

    if date not in rs_scores.index:
        # Find nearest previous date
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
    """
    ETF Relative Strength Momentum Backtest Engine.

    Strategy Rules:
    1. Every Friday EOD: Calculate RS scores, rank ETFs
    2. Select top N ETFs
    3. Execute buy at Monday's close price
    4. Hold until next rebalance
    5. Equal weight allocation among top N
    """

    def __init__(self, prices, initial_capital=1000000, top_n=3, 
                 lookback_period=63, rebalance_freq='W-FRI',
                 transaction_cost=0.001):
        """
        Initialize backtest parameters.

        Parameters:
        -----------
        prices : DataFrame
            Historical close prices for all ETFs
        initial_capital : float
            Starting capital in INR (default: 10 Lakhs)
        top_n : int
            Number of ETFs to hold in portfolio
        lookback_period : int
            Trading days to look back for RS calculation
        rebalance_freq : str
            Rebalancing frequency. 'W-FRI' = every Friday
        transaction_cost : float
            Cost per trade as fraction (0.001 = 0.1%)
        """
        self.prices = prices.copy()
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.lookback_period = lookback_period
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

        # Pre-calculate RS scores
        print(f"Calculating RS scores (lookback={lookback_period} days)...")
        self.rs_scores = calculate_relative_strength(prices, lookback_period)
        print("RS scores calculated.")

    def run_backtest(self, start_date=None, end_date=None):
        """
        Run the complete backtest.

        Parameters:
        -----------
        start_date : datetime or None
            Backtest start date
        end_date : datetime or None
            Backtest end date

        Returns:
        --------
        dict : Portfolio history and trade records
        """
        if start_date is None:
            # Start after lookback period + buffer
            start_idx = self.lookback_period + 20
            start_date = self.prices.index[start_idx]

        if end_date is None:
            end_date = self.prices.index[-1]

        # Filter to date range
        mask = (self.prices.index >= start_date) & (self.prices.index <= end_date)
        prices = self.prices.loc[mask].copy()
        rs_scores = self.rs_scores.loc[mask].copy()

        if len(prices) == 0:
            raise ValueError("No data in specified date range!")

        # Generate rebalance dates (Fridays)
        all_dates = prices.index
        fridays = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        fridays = [f for f in fridays if f in all_dates]

        # Generate execution dates (Mondays)
        mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        mondays = [m for m in mondays if m in all_dates]

        print(f"\nRunning backtest: {start_date.date()} to {end_date.date()}")
        print(f"Rebalance dates: {len(fridays)} Fridays")
        print(f"Execution dates: {len(mondays)} Mondays")

        # Portfolio state
        cash = self.initial_capital
        holdings = {}  # ticker -> shares
        value_history = []
        trades = []

        for i, friday in enumerate(fridays):
            if friday not in rs_scores.index:
                continue

            # Find next Monday for execution
            next_monday = None
            for m in mondays:
                if m > friday:
                    next_monday = m
                    break

            if next_monday is None or next_monday not in prices.index:
                continue

            # Get top N ETFs by RS score
            top_etfs, top_scores = get_top_n_etfs(
                rs_scores, n=self.top_n, date=friday
            )

            if len(top_etfs) == 0:
                continue

            # Calculate portfolio value before rebalance (at Friday close)
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in prices.columns and friday in prices.index:
                    price = prices.loc[friday, ticker]
                    if pd.notna(price):
                        portfolio_value += shares * price

            # Sell all current holdings at Friday close
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

            # Buy new top N at Monday's price
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

            # Record portfolio value at Monday close
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
                print(f"  Year {i//52}: Portfolio = ₹{current_value:,.0f}")

        return {
            'cash': cash,
            'holdings': holdings,
            'value_history': value_history,
            'trades': trades,
            'initial_capital': self.initial_capital
        }

    def calculate_metrics(self, portfolio):
        """
        Calculate comprehensive performance metrics.

        Parameters:
        -----------
        portfolio : dict
            Output from run_backtest()

        Returns:
        --------
        dict : Performance metrics
        """
        values = pd.DataFrame(portfolio['value_history'])
        if len(values) == 0:
            return {}

        values = values.set_index('date')
        values['returns'] = values['value'].pct_change()

        # Total return
        total_return = (values['value'].iloc[-1] / self.initial_capital - 1) * 100

        # Time period in years
        years = (values.index[-1] - values.index[0]).days / 365.25
        if years <= 0:
            years = 1

        # Annualized return (CAGR)
        ann_return = ((values['value'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100

        # Annualized volatility (weekly returns -> annual)
        weekly_returns = values['returns'].dropna()
        ann_vol = weekly_returns.std() * np.sqrt(52) * 100

        # Sharpe ratio (assuming 6% risk-free rate)
        sharpe = (ann_return - 6) / ann_vol if ann_vol > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = weekly_returns[weekly_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else 0.0001
        sortino = (ann_return - 6) / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cummax = values['value'].cummax()
        drawdown = (values['value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate (positive weeks)
        win_rate = (weekly_returns > 0).mean() * 100

        # Average win / loss
        avg_win = weekly_returns[weekly_returns > 0].mean() * 100 if (weekly_returns > 0).any() else 0
        avg_loss = weekly_returns[weekly_returns < 0].mean() * 100 if (weekly_returns < 0).any() else 0

        # Profit factor
        gross_profit = weekly_returns[weekly_returns > 0].sum()
        gross_loss = abs(weekly_returns[weekly_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Number of trades
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
    """
    Find optimal lookback period for each time frame.

    Parameters:
    -----------
    prices : DataFrame
        Historical close prices
    lookback_options : list or None
        Days to test. Default: [21, 42, 63, 126, 252]
    test_periods : dict or None
        Periods to test. Default: 10Y, 5Y, 3Y, 1Y
    initial_capital : float
        Starting capital
    top_n : int
        Number of ETFs to hold

    Returns:
    --------
    dict : Results for each period with best lookback
    """
    if lookback_options is None:
        lookback_options = [21, 42, 63, 126, 252]  # 1M, 2M, 3M, 6M, 1Y

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
        # Adjust start if before data availability
        if start < prices.index[0]:
            start = prices.index[0]
            print(f"\n{period_name}: Adjusted start to {start.date()} (data limit)")
        else:
            print(f"\n{'='*60}")
            print(f"PERIOD: {period_name} ({start.date()} to {end.date()})")
            print(f"{'='*60}")

        period_results = {}

        for lookback in lookback_options:
            # Skip if lookback too long for period
            min_required_days = lookback + 30
            if (end - start).days < min_required_days:
                print(f"  Lookback {lookback}: Skipped (insufficient data)")
                continue

            print(f"\n  Testing lookback: {lookback} days (~{lookback/21:.0f} months)")

            try:
                bt = ETFBacktest(
                    prices, 
                    lookback_period=lookback,
                    initial_capital=initial_capital,
                    top_n=top_n
                )

                portfolio = bt.run_backtest(start_date=start, end_date=end)
                metrics = bt.calculate_metrics(portfolio)

                period_results[lookback] = metrics

                if metrics:
                    print(f"    Ann Return: {metrics['annualized_return']:>8.2f}%")
                    print(f"    Sharpe:     {metrics['sharpe_ratio']:>8.2f}")
                    print(f"    Max DD:     {metrics['max_drawdown']:>8.2f}%")
                    print(f"    Final Val:  ₹{metrics['final_value']:>12,.0f}")

            except Exception as e:
                print(f"    Error: {e}")
                period_results[lookback] = None

        # Find best lookback for this period
        valid_results = {
            k: v for k, v in period_results.items() 
            if v and 'sharpe_ratio' in v
        }

        if valid_results:
            # Primary: Sharpe ratio. Secondary: Annualized return
            best_lookback = max(
                valid_results.items(),
                key=lambda x: (x[1]['sharpe_ratio'], x[1]['annualized_return'])
            )

            results[period_name] = {
                'best_lookback': best_lookback[0],
                'best_metrics': best_lookback[1],
                'all_results': period_results
            }

            print(f"\n  {'*'*50}")
            print(f"  BEST LOOKBACK for {period_name}: {best_lookback[0]} days")
            print(f"  Sharpe Ratio: {best_lookback[1]['sharpe_ratio']:.2f}")
            print(f"  Ann Return:   {best_lookback[1]['annualized_return']:.2f}%")
            print(f"  Max Drawdown: {best_lookback[1]['max_drawdown']:.2f}%")
            print(f"  {'*'*50}")
        else:
            print(f"\n  No valid results for {period_name}")

    return results


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

def download_benchmark(ticker="^NSEI", start_date=None, end_date=None):
    """Download benchmark data (Nifty 50)."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, 
                          progress=False, auto_adjust=True)
        return data['Close']
    except Exception as e:
        print(f"Could not download benchmark {ticker}: {e}")
        return None


def compare_with_benchmark(strategy_metrics, benchmark_prices):
    """
    Compare strategy performance with benchmark.

    Parameters:
    -----------
    strategy_metrics : dict
        Strategy performance metrics
    benchmark_prices : Series
        Benchmark closing prices
    """
    if benchmark_prices is None or len(benchmark_prices) == 0:
        print("Benchmark data not available for comparison.")
        return

    # Align dates
    strategy_values = strategy_metrics['values']['value']
    benchmark_aligned = benchmark_prices.reindex(strategy_values.index, method='ffill')

    # Calculate benchmark returns
    benchmark_returns = benchmark_aligned.pct_change().dropna()

    # Benchmark metrics
    years = strategy_metrics['years']
    bench_total = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1) * 100
    bench_ann = ((benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) ** (1/years) - 1) * 100
    bench_vol = benchmark_returns.std() * np.sqrt(52) * 100
    bench_sharpe = (bench_ann - 6) / bench_vol if bench_vol > 0 else 0

    # Benchmark drawdown
    bench_cummax = benchmark_aligned.cummax()
    bench_dd = (benchmark_aligned - bench_cummax) / bench_cummax
    bench_max_dd = bench_dd.min() * 100

    print(f"\n{'='*60}")
    print("STRATEGY vs BENCHMARK (Nifty 50)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Strategy':>12} {'Benchmark':>12} {'Diff':>12}")
    print(f"{'-'*60}")
    print(f"{'Ann Return (%)':<25} {strategy_metrics['annualized_return']:>12.2f} {bench_ann:>12.2f} {strategy_metrics['annualized_return']-bench_ann:>+12.2f}")
    print(f"{'Volatility (%)':<25} {strategy_metrics['annualized_volatility']:>12.2f} {bench_vol:>12.2f} {strategy_metrics['annualized_volatility']-bench_vol:>+12.2f}")
    print(f"{'Sharpe Ratio':<25} {strategy_metrics['sharpe_ratio']:>12.2f} {bench_sharpe:>12.2f} {strategy_metrics['sharpe_ratio']-bench_sharpe:>+12.2f}")
    print(f"{'Max Drawdown (%)':<25} {strategy_metrics['max_drawdown']:>12.2f} {bench_max_dd:>12.2f} {strategy_metrics['max_drawdown']-bench_max_dd:>+12.2f}")
    print(f"{'Final Value (₹)':<25} {strategy_metrics['final_value']:>12,.0f} {'-':>12} {'-':>12}")
    print(f"{'-'*60}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(results, benchmark_prices=None, save_path="etf_backtest_charts.png"):
    """
    Create comprehensive visualization of backtest results.

    Parameters:
    -----------
    results : dict
        Output from optimize_lookback()
    benchmark_prices : Series or None
        Benchmark prices for comparison
    save_path : str
        File path to save charts
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed. Skipping charts.")
        print("Install with: pip install matplotlib")
        return

    n_periods = len(results)
    if n_periods == 0:
        print("No results to plot.")
        return

    fig = plt.figure(figsize=(18, 4 * n_periods + 4))

    for idx, (period_name, result) in enumerate(results.items()):
        metrics = result['best_metrics']
        values = metrics['values']

        # Normalize to base 100
        normalized = values['value'] / values['value'].iloc[0] * 100

        # Equity curve subplot
        ax1 = plt.subplot(n_periods + 1, 2, 2*idx + 1)
        ax1.plot(normalized.index, normalized.values, linewidth=2, 
                color='#2E86AB', label='Strategy')

        # Add benchmark if available
        if benchmark_prices is not None:
            bench_aligned = benchmark_prices.reindex(values.index, method='ffill')
            bench_norm = bench_aligned / bench_aligned.iloc[0] * 100
            ax1.plot(bench_norm.index, bench_norm.values, linewidth=1.5,
                    color='#A23B72', linestyle='--', alpha=0.7, label='Nifty 50')

        ax1.set_title(f'{period_name} - Lookback: {result["best_lookback"]}d | '
                     f'CAGR: {metrics["annualized_return"]:.1f}% | '
                     f'Sharpe: {metrics["sharpe_ratio"]:.2f}',
                     fontsize=11, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Base 100)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Drawdown subplot
        ax2 = plt.subplot(n_periods + 1, 2, 2*idx + 2)
        dd_pct = metrics['drawdown'] * 100
        ax2.fill_between(dd_pct.index, dd_pct.values, 0, 
                        color='#F18F01', alpha=0.3)
        ax2.plot(dd_pct.index, dd_pct.values, color='#C73E1D', linewidth=1)
        ax2.set_title(f'Drawdown | Max: {metrics["max_drawdown"]:.1f}%',
                     fontsize=11)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Summary table
    ax3 = plt.subplot(n_periods + 1, 1, n_periods + 1)
    ax3.axis('off')

    table_data = []
    headers = ['Period', 'Lookback', 'CAGR%', 'Sharpe', 'MaxDD%', 'Win%', 'Final ₹']
    table_data.append(headers)

    for period_name, result in results.items():
        m = result['best_metrics']
        row = [
            period_name,
            f"{result['best_lookback']}d",
            f"{m['annualized_return']:.1f}",
            f"{m['sharpe_ratio']:.2f}",
            f"{m['max_drawdown']:.1f}",
            f"{m['win_rate']:.1f}",
            f"₹{m['final_value']:,.0f}"
        ]
        table_data.append(row)

    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colColours=['#2E86AB']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(7):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.suptitle('ETF Relative Strength Momentum Backtest Results',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCharts saved to: {save_path}")
    plt.close()


def plot_lookback_comparison(results, save_path="lookback_comparison.png"):
    """
    Plot comparison of all lookback periods for each time frame.
    """
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
        ax.set_title(f'{period_name}: Lookback Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lb}d' for lb in lookbacks])
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best
        best_idx = lookbacks.index(result['best_lookback'])
        bars1[best_idx].set_color('#1B4965')
        bars2[best_idx].set_color('#C73E1D')

    plt.suptitle('Lookback Period Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Lookback comparison saved to: {save_path}")
    plt.close()


# ============================================================
# EXPORT RESULTS
# ============================================================

def export_results(results, filename="backtest_results.json"):
    """Export results to JSON file."""
    export_data = {}

    for period_name, result in results.items():
        export_data[period_name] = {
            'best_lookback': result['best_lookback'],
            'metrics': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in result['best_metrics'].items()
                if k not in ['values', 'drawdown']  # Skip DataFrames
            }
        }

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Results exported to: {filename}")


def print_detailed_results(results):
    """Print comprehensive results table."""
    print(f"\n{'='*80}")
    print("COMPLETE BACKTEST RESULTS SUMMARY")
    print(f"{'='*80}")

    for period_name, result in results.items():
        m = result['best_metrics']
        lb = result['best_lookback']

        print(f"\n{'─'*80}")
        print(f"PERIOD: {period_name} | OPTIMAL LOOKBACK: {lb} days (~{lb/21:.0f} months)")
        print(f"{'─'*80}")
        print(f"  {'Metric':<30} {'Value':>20}")
        print(f"  {'-'*50}")
        print(f"  {'Initial Capital':<30} {'₹1,000,000':>20}")
        print(f"  {'Final Portfolio Value':<30} {f'₹{m["final_value"]:,.0f}':>20}")
        print(f"  {'Total Return':<30} {f'{m["total_return"]:.2f}%':>20}")
        print(f"  {'CAGR (Annualized Return)':<30} {f'{m["annualized_return"]:.2f}%':>20}")
        print(f"  {'Annualized Volatility':<30} {f'{m["annualized_volatility"]:.2f}%':>20}")
        print(f"  {'Sharpe Ratio':<30} {f'{m["sharpe_ratio"]:.2f}':>20}")
        print(f"  {'Sortino Ratio':<30} {f'{m["sortino_ratio"]:.2f}':>20}")
        print(f"  {'Calmar Ratio':<30} {f'{m["calmar_ratio"]:.2f}':>20}")
        print(f"  {'Maximum Drawdown':<30} {f'{m["max_drawdown"]:.2f}%':>20}")
        print(f"  {'Win Rate (Weekly)':<30} {f'{m["win_rate"]:.1f}%':>20}")
        print(f"  {'Average Winning Week':<30} {f'{m["avg_win"]:.2f}%':>20}")
        print(f"  {'Average Losing Week':<30} {f'{m["avg_loss"]:.2f}%':>20}")
        print(f"  {'Profit Factor':<30} {f'{m["profit_factor"]:.2f}':>20}")
        print(f"  {'Number of Buy Trades':<30} {f'{m["num_buys"]}':>20}")
        print(f"  {'Number of Sell Trades':<30} {f'{m["num_sells"]}':>20}")
        print(f"  {'Backtest Duration':<30} {f'{m["years"]:.1f} years':>20}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function."""
    print("="*80)
    print("ETF RELATIVE STRENGTH MOMENTUM BACKTEST SYSTEM")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ETFs: {len(TICKERS)} instruments")
    print(f"Strategy: Top {3} ETFs by RS Score | Friday Signal | Monday Execution")
    print(f"Capital: ₹10,00,000")
    print("="*80)

    # Step 1: Download data
    try:
        prices = download_data(cache_file="etf_prices.csv")
    except Exception as e:
        print(f"Batch download failed: {e}")
        print("Trying individual downloads...")
        prices = download_data_individual(cache_file="etf_prices.csv")

    print(f"\nData Summary:")
    print(f"  ETFs with data: {len(prices.columns)}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Total trading days: {len(prices)}")

    # Step 2: Run optimization
    lookback_options = [21, 42, 63, 126, 252]

    results = optimize_lookback(
        prices,
        lookback_options=lookback_options,
        initial_capital=1000000,
        top_n=3
    )

    # Step 3: Print detailed results
    print_detailed_results(results)

    # Step 4: Benchmark comparison for best period
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}")

    benchmark = download_benchmark("^NSEI")

    for period_name, result in results.items():
        print(f"\n--- {period_name} ---")
        compare_with_benchmark(result['best_metrics'], benchmark)

    # Step 5: Visualization
    print(f"\n{'='*80}")
    print("GENERATING CHARTS")
    print(f"{'='*80}")

    plot_results(results, benchmark_prices=benchmark, 
                save_path="etf_backtest_equity.png")
    plot_lookback_comparison(results, save_path="lookback_comparison.png")

    # Step 6: Export
    export_results(results, "backtest_results.json")

    # Step 7: Current signal
    print(f"\n{'='*80}")
    print("CURRENT SIGNAL (Latest Available Data)")
    print(f"{'='*80}")

    # Use best lookback from 1Y period (most recent)
    if '1Y' in results:
        best_lb = results['1Y']['best_lookback']
    else:
        best_lb = 63  # Default

    final_rs = calculate_relative_strength(prices, best_lb)
    top3, scores = get_top_n_etfs(final_rs, n=3)

    print(f"\nUsing lookback: {best_lb} days")
    print(f"\n{'Rank':<6} {'Ticker':<15} {'Name':<20} {'RS Score':>12}")
    print(f"{'-'*55}")
    for i, (ticker, score) in enumerate(zip(top3, scores), 1):
        name = NAMES.get(ticker, ticker)
        print(f"{i:<6} {ticker:<15} {name:<20} {score:>12.4f}")

    print(f"\n{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return results


if __name__ == "__main__":
    results = main()
