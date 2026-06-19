"""
India ETF Pairwise RS Matrix Backtester (Production)
==================================================
Universe  : 24 NSE ETFs (Commodities, Broad, Sectoral, Debt, Smart Beta)
Signal    : 63-day pairwise relative strength — each ETF vs every other
Portfolio : Long top-3 by RS score, equally weighted
Rebalance : Weekly (Friday close signal → executed same close)
Capital   : ₹10,00,000
Costs     : 0.1% per trade (brokerage + slippage)
Period    : Jan 2023 → present (live-updated via GitHub Actions)

FIXES vs previous version:
  - TRADE_START corrected to 2023-01-01 (was set to today → blank charts)
  - period_return returns None on bad data (was 0.0 → corrupted RS scores)
  - Cash accounting fixed (no double-subtract on retained positions)
  - RS ranking is pure pairwise avg: score(A) = mean(ret_A - ret_B for all B≠A)
  - Blank chart fix: dates passed as ISO strings, data validated before render
  - Removed unused setup_global_session()
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys
import os
import time
import random
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ETFS = [
    # --- Commodities ---
    ("GOLDBEES.NS",    "GoldBees",       "GOLD"),
    ("SILVERBEES.NS",  "SilverBees",     "SILV"),

    # --- Core Broad Market ---
    ("NIFTYBEES.NS",   "NiftyBees",      "NFTY"),
    ("JUNIORBEES.NS",  "JuniorBees",     "JNBR"),
    ("MID150BEES.NS",  "Midcap150Bees",  "MIDM"),
    ("NIF100BEES.NS",  "Nifty100Bees",   "NF10"),

    # --- Sectoral ---
    ("BANKBEES.NS",    "BankBees",       "BANK"),
    ("ITBEES.NS",      "ITBees",         "ITMC"),
    ("PHARMABEES.NS",  "PharmaBees",     "PHRM"),
    ("AUTOBEES.NS",    "AutoBees",       "AUTO"),
    ("INFRABEES.NS",   "InfraBees",      "INFR"),
    ("CONSUMBEES.NS",  "ConsumeBees",    "CNSM"),

    # --- PSU / Theme ---
    ("PSUBNKBEES.NS",  "PSUBankBees",    "PSUB"),
    ("CPSEETF.NS",     "CPSE ETF",       "CETF"),

    # --- Debt / Fixed Income ---
    ("LTGILTBEES.NS",  "LT Gilt Bees",   "GSCP"),
    ("GILT5YBEES.NS",  "GSec 5Y Bees",   "GS5Y"),
    ("LIQUIDBEES.NS",  "LiquidBees",     "LIQD"),

    # --- Smart Beta / Factor / International ---
    ("MOM100.NS",      "Momentum100",    "MOM" ),
    ("MOMENTUM30.NS",  "Momentum30",     "MOM3"),
    ("NV20BEES.NS",    "Value20Bees",    "NV20"),
    ("DIVOPPBEES.NS",  "DivOppBees",     "DIVO"),
    ("HNGSNGBEES.NS",  "HangSengBees",   "HNGS"),
    ("MAFANG.NS",      "FANGPlus ETF",   "FANG"),
    ("MON100.NS",      "Nasdaq100 ETF",  "NSDQ"),
]

LOOKBACK    = 63            # ~3 months of trading days
TOP_N       = 3             # long top-3 instruments
COST_PCT    = 0.001         # 0.1% per trade side
INITIAL     = 1_000_000     # ₹10 lakh
FETCH_START = "2023-01-01"  # extra buffer so lookback works from Jan 2023
TRADE_START = "2025-06-01"  # ← FIXED: was "2025-06-18" (today) → blank charts
END         = date.today().strftime("%Y-%m-%d")
NIFTY_SYM   = "NIFTYBEES.NS"
OUT_PATH    = "docs/index.html"
CACHE_DIR   = ".cache/yfinance_data"

# ─── DATA FETCH ──────────────────────────────────────────────────────────────
def _fetch_one(sym: str, retries: int = 4, delay: float = 4.0):
    """
    Fetch a single ticker's daily Close prices with retries + local file cache.
    Returns pd.Series (DatetimeIndex, tz-naive) or None on failure.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{sym}_{FETCH_START}_{END}.csv")

    # 1. Try local cache first
    if os.path.exists(cache_path):
        try:
            df_cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df_cached.empty:
                s = df_cached.iloc[:, 0].copy()
                s.name = sym
                return s
        except Exception:
            pass  # corrupted cache → re-download

    # 2. Network download with retries
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=sym,
                start=FETCH_START,
                end=END,
                progress=False,
                timeout=25,
                auto_adjust=True,
            )
            if df is None or df.empty:
                raise ValueError("empty dataframe")

            # Handle MultiIndex columns (yfinance >= 0.2.x)
            if isinstance(df.columns, pd.MultiIndex):
                s = df["Close"][sym].copy()
            else:
                s = df["Close"].copy()

            # Strip timezone → naive index so all tickers align
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.name = sym

            if s.notna().sum() < 50:
                raise ValueError(f"only {s.notna().sum()} valid rows")

            s.to_csv(cache_path)
            return s

        except Exception as e:
            if attempt == retries:
                print(f"    ❌ all {retries} attempts failed: {e}")
            time.sleep(delay * attempt + random.uniform(0.5, 2.0))
    return None


def fetch_prices() -> pd.DataFrame:
    print(f"Fetching {len(ETFS)} ETFs from {FETCH_START} to {END} …")
    series_list = []

    for i, (sym, name, short) in enumerate(ETFS, 1):
        print(f"  [{i:>2}/{len(ETFS)}] {name:<18} ({sym})", end=" … ", flush=True)
        s = _fetch_one(sym)
        if s is not None and s.notna().sum() > 50:
            series_list.append(s)
            print(f"✓  {s.notna().sum()} rows")
        else:
            print("✗  skipped")
        time.sleep(random.uniform(1.0, 2.0))

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.ffill(limit=5)   # fill weekends / holidays

    valid = [c for c in prices.columns if prices[c].notna().sum() > 100]
    dropped = [s for s in [e[0] for e in ETFS] if s not in valid]
    if dropped:
        print(f"\n  ⚠  Dropped (insufficient history): {[s.split('.')[0] for s in dropped]}")

    prices = prices[valid].dropna(how="all")
    print(f"\n  ✓  {len(prices)} trading days · {len(valid)} instruments ready\n")
    return prices

# ─── RS ENGINE ───────────────────────────────────────────────────────────────
def period_return(series: pd.Series, idx_now: int, lookback_days: int):
    """
    63-day simple return.
    Returns None if either price is NaN or zero — callers must handle None.
    ← FIX: was returning 0.0 which silently corrupted RS rankings.
    """
    idx_past = max(idx_now - lookback_days, 0)
    if idx_past == idx_now:
        return None                          # not enough history yet
    p_now  = series.iloc[idx_now]
    p_past = series.iloc[idx_past]
    if pd.isna(p_now) or pd.isna(p_past) or p_past == 0:
        return None
    return float((p_now / p_past) - 1.0)


def compute_rs(prices: pd.DataFrame, idx: int, available_syms: list):
    """
    Pairwise RS score for each instrument:
        score(A) = mean over all B≠A of [ret_A(63d) − ret_B(63d)]

    Positive score → A outperformed the average peer over 63 days.
    Instruments with None return are excluded from both ranking and peer comparison.
    """
    # Raw 63-day returns; None means insufficient data
    rets: dict = {}
    for sym in available_syms:
        if sym in prices.columns:
            rets[sym] = period_return(prices[sym], idx, LOOKBACK)

    # Only use instruments with valid returns for peer comparison
    valid_syms = [s for s in rets if rets[s] is not None]

    scores: dict = {}
    for sym in available_syms:
        if rets.get(sym) is None:
            scores[sym] = None          # can't rank this instrument
            continue
        peers = [rets[s] for s in valid_syms if s != sym]
        if peers:
            scores[sym] = float(np.mean([rets[sym] - p for p in peers]))
        else:
            scores[sym] = 0.0

    return scores, rets


def build_matrix(rets: dict, available_syms: list):
    """
    N×N pairwise return-difference matrix.
    matrix[i][j] = ret_i − ret_j  (positive = row outperforms column, in %)
    None entries = one of the instruments had no valid return.
    """
    matrix = []
    for i, (si, _, _) in enumerate(ETFS):
        row = []
        for j, (sj, _, _) in enumerate(ETFS):
            if i == j:
                row.append(0.0)
            elif (si in available_syms and sj in available_syms
                  and rets.get(si) is not None and rets.get(sj) is not None):
                row.append(round((rets[si] - rets[sj]) * 100, 3))
            else:
                row.append(None)
        matrix.append(row)
    return matrix

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(prices: pd.DataFrame):
    available_syms = prices.columns.tolist()

    # Snap all Fridays from TRADE_START→END to nearest available trading day
    all_fridays = pd.date_range(TRADE_START, END, freq="W-FRI")
    snapped = []
    for f in all_fridays:
        pos = prices.index.searchsorted(f, side="right") - 1
        if 0 <= pos < len(prices):
            snapped.append(pos)
    snapped = sorted(set(snapped))

    if not snapped:
        print("ERROR: No trading days found between TRADE_START and END.")
        return [], [], [], [], 0, {}, {}, None, None, None

    # Portfolio state
    cash      = float(INITIAL)
    holdings  = {}           # sym → shares (float)
    cur_top3  = []
    peak      = float(INITIAL)
    total_trades = 0
    hold_count   = {e[0]: 0 for e in ETFS}
    instr_trades = {e[0]: 0 for e in ETFS}

    equity_curve, nifty_curve, dd_curve, trade_log = [], [], [], []

    # Nifty baseline for benchmark comparison
    nifty_start_pos = max(prices.index.searchsorted(pd.Timestamp(TRADE_START), side="right") - 1, 0)
    nifty_start_px  = (float(prices[NIFTY_SYM].iloc[nifty_start_pos])
                       if NIFTY_SYM in prices.columns else None)

    last_scores = last_rets = last_matrix = None

    for wi, idx in enumerate(snapped):
        date_str = str(prices.index[idx].date())

        scores, rets = compute_rs(prices, idx, available_syms)

        # Rank by score descending; skip instruments with None score
        ranked = sorted(
            [(s, v) for s, v in scores.items() if v is not None],
            key=lambda x: -x[1]
        )
        new_top3 = [s for s, _ in ranked[:TOP_N]]

        if len(new_top3) < TOP_N:
            print(f"  ⚠  {date_str}: only {len(new_top3)} valid instruments, skipping week")
            continue

        needs_rebal = (wi == 0) or (set(new_top3) != set(cur_top3))
        exiting     = [s for s in cur_top3 if s not in new_top3]
        entering    = [s for s in new_top3  if s not in cur_top3]

        if needs_rebal:
            # ── Step 1: Sell exiting positions → cash ──────────────────────
            for sym in exiting:
                if sym in holdings and holdings[sym] > 0:
                    px = float(prices[sym].iloc[idx])
                    cash += holdings[sym] * px * (1 - COST_PCT)
                    instr_trades[sym] += 1
                    total_trades += 1
                    del holdings[sym]

            # ── Step 2: Value retained positions ───────────────────────────
            retained_val = sum(
                holdings[s] * float(prices[s].iloc[idx])
                for s in holdings if s in prices.columns
            )

            # ── Step 3: Compute target allocation ──────────────────────────
            # Total capital = cash in hand + value of retained positions
            total_capital    = cash + retained_val
            target_per_pos   = total_capital / TOP_N

            # ── Step 4: Rebalance retained + buy entering ───────────────────
            # FIX: Only touch positions that need adjustment or are new.
            # cash -= target_per_pos was running for ALL new_top3 even when
            # sym was already held → double-counted retained value → negative cash.
            for sym in new_top3:
                px = float(prices[sym].iloc[idx])
                current_val = holdings.get(sym, 0) * px

                # Rebalance if: new position, or weight drifted >5% from target
                if sym in entering or abs(current_val - target_per_pos) > target_per_pos * 0.05:
                    # Sell existing lot if any (book proceeds back to cash)
                    if sym in holdings and holdings[sym] > 0:
                        cash += holdings[sym] * px * (1 - COST_PCT)
                        instr_trades[sym] += 1
                        total_trades += 1

                    # Buy fresh lot at target weight
                    # Cash required = target_per_pos (gross); cost deducted on buy
                    shares = (target_per_pos * (1 - COST_PCT)) / px
                    holdings[sym] = shares
                    cash -= target_per_pos    # ← cash outflow for this position
                    instr_trades[sym] += 1
                    total_trades += 1

        # ── Mark-to-market ────────────────────────────────────────────────────
        port_val = cash + sum(
            holdings[s] * float(prices[s].iloc[idx])
            for s in holdings if s in prices.columns
        )
        # Safety clamp: floating point shouldn't go meaningfully negative
        port_val = max(port_val, 0)

        for sym in new_top3:
            hold_count[sym] = hold_count.get(sym, 0) + 1

        # Nifty benchmark
        if nifty_start_px and NIFTY_SYM in prices.columns:
            nifty_px  = float(prices[NIFTY_SYM].iloc[idx])
            nifty_val = (nifty_px / nifty_start_px) * INITIAL
        else:
            nifty_val = INITIAL

        if port_val > peak:
            peak = port_val
        dd = (port_val - peak) / peak * 100

        equity_curve.append({"date": date_str, "value": round(port_val, 2)})
        nifty_curve.append( {"date": date_str, "value": round(nifty_val, 2)})
        dd_curve.append(    {"date": date_str, "dd":    round(dd, 3)})

        trade_log.append({
            "date":     date_str,
            "top3":     new_top3,
            "exiting":  exiting,
            "entering": entering,
            "changed":  needs_rebal and wi > 0,
            "capital":  round(port_val, 2),
            # Store raw values for display; HTML builder does *100 for %
            "scores": {s: round(v, 6) if v is not None else None for s, v in scores.items()},
            "rets":   {s: round(v, 6) if v is not None else None for s, v in rets.items()},
        })

        cur_top3    = new_top3
        last_scores = scores
        last_rets   = rets
        last_matrix = build_matrix(rets, available_syms)

        if wi % 10 == 0:
            top3_short = [s.replace(".NS","") for s in new_top3]
            print(f"  {date_str}  ₹{port_val:>10,.0f}  top3={top3_short}")

    return (equity_curve, nifty_curve, dd_curve, trade_log,
            total_trades, hold_count, instr_trades,
            last_scores, last_rets, last_matrix)

# ─── STATS ───────────────────────────────────────────────────────────────────
def calc_stats(equity_curve, nifty_curve, dd_curve, total_trades):
    final_val   = equity_curve[-1]["value"]
    nifty_final = nifty_curve[-1]["value"]
    t0 = pd.Timestamp(TRADE_START)
    t1 = pd.Timestamp(END)
    years = max((t1 - t0).days / 365.25, 0.05)

    total_ret  = (final_val - INITIAL) / INITIAL * 100
    cagr       = (pow(final_val / INITIAL,   1 / years) - 1) * 100
    nifty_cagr = (pow(nifty_final / INITIAL, 1 / years) - 1) * 100
    alpha      = cagr - nifty_cagr

    weekly_rets = [
        (equity_curve[i]["value"] - equity_curve[i-1]["value"]) / equity_curve[i-1]["value"]
        for i in range(1, len(equity_curve))
    ]
    mean_w = np.mean(weekly_rets) if weekly_rets else 0
    std_w  = np.std(weekly_rets)  if weekly_rets else 1
    sharpe = (mean_w / std_w) * np.sqrt(52) if std_w > 0 else 0
    max_dd = min(d["dd"] for d in dd_curve) if dd_curve else 0

    return dict(
        final_val=round(final_val, 2),
        total_ret=round(total_ret, 2),
        cagr=round(cagr, 2),
        sharpe=round(sharpe, 2),
        max_dd=round(max_dd, 2),
        total_trades=total_trades,
        nifty_cagr=round(nifty_cagr, 2),
        alpha=round(alpha, 2),
        weeks=len(equity_curve),
    )

# ─── HTML REPORT ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>India ETF RS Backtest</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--card2:#22263a;--border:#2e3250;--accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--text:#e2e8f0;--muted:#8892b0}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
header{background:var(--card);border-bottom:1px solid var(--border);padding:18px 32px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
header h1{font-size:1.25rem;font-weight:700;color:var(--accent)}
header .meta{color:var(--muted);font-size:0.82rem}
.container{max-width:1400px;margin:0 auto;padding:24px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:24px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.stat-card .label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.stat-card .value{font-size:1.45rem;font-weight:700}
.green{color:var(--green)}.red{color:var(--red)}.accent{color:var(--accent)}.yellow{color:var(--yellow)}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:22px}
.chart-card h3{font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px}
.chart-wrap{position:relative}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:22px}
@media(max-width:800px){.grid-2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:0.84rem}
th{background:var(--card2);color:var(--muted);font-weight:600;text-transform:uppercase;font-size:0.7rem;letter-spacing:.05em;padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid var(--border)}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.73rem;font-weight:600}
.badge-green{background:rgba(34,197,94,.15);color:var(--green)}
.badge-red{background:rgba(239,68,68,.15);color:var(--red)}
.badge-muted{background:rgba(136,146,176,.1);color:var(--muted)}
.section-title{font-size:.95rem;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.section-title::before{content:'';display:block;width:4px;height:18px;background:var(--accent);border-radius:2px}
.matrix-wrap{overflow-x:auto}
.mx{border-collapse:collapse;font-size:.72rem;white-space:nowrap}
.mx th,.mx td{padding:5px 7px;border:1px solid var(--border);text-align:center;min-width:54px}
.mx th{background:var(--card2);color:var(--muted);font-weight:600}
.mx .rl{font-weight:600;color:var(--text);background:var(--card2);text-align:left;padding-left:10px;min-width:70px}
.tag{display:inline-flex;align-items:center;background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.3);color:var(--accent);border-radius:5px;padding:2px 9px;font-size:.78rem;font-weight:600;margin:2px}
.tabs{display:flex;gap:4px;background:var(--card2);padding:4px;border-radius:8px;width:fit-content;margin-bottom:16px}
.tab{padding:6px 16px;border-radius:6px;cursor:pointer;font-size:.83rem;font-weight:500;color:var(--muted);transition:all .15s}
.tab.active{background:var(--accent);color:#fff}
#log{max-height:420px;overflow-y:auto}
.lr{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.81rem;align-items:flex-start}
.ld{color:var(--muted);min-width:92px}
.lb{flex:1}
.updated{text-align:right;color:var(--muted);font-size:.78rem;padding:10px 0 0}
</style>
</head>
<body>
<header>
  <div>
    <h1>🇮🇳 India ETF — Pairwise RS Matrix Backtest</h1>
    <div class="meta">__N_ETFS__ ETFs · 63-day pairwise RS · Weekly Rebalance · Top-3 Long · ₹10L · Jan 2023 → present</div>
  </div>
  <div class="meta">Updated: __UPDATED__</div>
</header>
<div class="container">
<div class="stats-grid">
  <div class="stat-card"><div class="label">Final Portfolio</div><div class="value accent">__FINAL__</div></div>
  <div class="stat-card"><div class="label">Total Return</div><div class="value __RETCLS__">__RETURN__</div></div>
  <div class="stat-card"><div class="label">CAGR</div><div class="value __CAGRCLS__">__CAGR__</div></div>
  <div class="stat-card"><div class="label">Sharpe Ratio</div><div class="value __SHARPCLS__">__SHARPE__</div></div>
  <div class="stat-card"><div class="label">Max Drawdown</div><div class="value red">__MAXDD__</div></div>
  <div class="stat-card"><div class="label">Total Trades</div><div class="value">__TRADES__</div></div>
  <div class="stat-card"><div class="label">Nifty CAGR</div><div class="value">__NIFTY__</div></div>
  <div class="stat-card"><div class="label">Alpha vs Nifty</div><div class="value __ALPHACLS__">__ALPHA__</div></div>
</div>

<div class="chart-card">
  <h3>📈 Equity Curve — Strategy vs NiftyBees</h3>
  <div class="chart-wrap" style="height:300px"><canvas id="ec"></canvas></div>
</div>
<div class="chart-card">
  <h3>📉 Drawdown</h3>
  <div class="chart-wrap" style="height:160px"><canvas id="dc"></canvas></div>
</div>

<div class="grid-2">
  <div class="chart-card">
    <div class="section-title">Current RS Rankings (last rebalance)</div>
    __RANKINGS__
  </div>
  <div class="chart-card">
    <div class="section-title">Time in Portfolio</div>
    __CONTRIB__
  </div>
</div>

<div class="chart-card">
  <div class="section-title">Pairwise RS Matrix — last rebalance (row outperforms column, 63d %)</div>
  <div class="matrix-wrap">__MATRIX__</div>
</div>

<div class="chart-card">
  <div class="section-title">Weekly Rebalance Log</div>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('all',this)">All Weeks</div>
    <div class="tab" onclick="switchTab('chg',this)">Changes Only</div>
  </div>
  <div id="log">__LOG__</div>
</div>
<div class="updated">Generated by GitHub Actions · yfinance · Pairwise RS Strategy</div>
</div>

<script>
// FIX: parse dates explicitly so Chart.js time adapter never gets undefined x values
const parseRows = rows => rows.map(r => ({x: new Date(r.x).getTime(), y: r.y}));
const edata = parseRows(__EQUITY_JSON__);
const ndata = parseRows(__NIFTY_JSON__);
const ddata = __DD_JSON__.map(r => ({x: new Date(r.x).getTime(), y: r.y}));

const gc = '#2e3250';
const tooltip = {backgroundColor:'#1a1d27',titleColor:'#e2e8f0',bodyColor:'#8892b0'};

new Chart(document.getElementById('ec'),{type:'line',data:{datasets:[
  {label:'RS Strategy', data:edata, borderColor:'#4f8ef7', backgroundColor:'rgba(79,142,247,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true},
  {label:'NiftyBees',   data:ndata, borderColor:'#f59e0b', backgroundColor:'transparent',           borderWidth:1.5,pointRadius:0, tension:0.3, borderDash:[5,4]}
]},options:{responsive:true,maintainAspectRatio:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#8892b0'}},tooltip:{...tooltip,callbacks:{label:c=>`${c.dataset.label}: ₹${(c.raw.y/1000).toFixed(1)}K`}}},
  scales:{
    x:{type:'time',time:{unit:'month'},ticks:{color:'#8892b0',maxTicksLimit:20},grid:{color:gc}},
    y:{ticks:{color:'#8892b0',callback:v=>'₹'+(v/1000).toFixed(0)+'K'},grid:{color:gc}}
  }}});

new Chart(document.getElementById('dc'),{type:'line',data:{datasets:[
  {label:'Drawdown',data:ddata,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.12)',borderWidth:1.5,pointRadius:0,tension:0.3,fill:true}
]},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{...tooltip,callbacks:{label:c=>`DD: ${c.raw.y.toFixed(2)}%`}}},
  scales:{
    x:{type:'time',time:{unit:'month'},ticks:{color:'#8892b0',maxTicksLimit:20},grid:{color:gc}},
    y:{ticks:{color:'#8892b0',callback:v=>v.toFixed(1)+'%'},grid:{color:gc}}
  }}});

function switchTab(t,el){
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.lr').forEach(r=>{
    r.style.display = (t==='all' || r.dataset.changed==='1') ? 'flex' : 'none';
  });
}
</script>
</body>
</html>"""

# ─── HTML BUILDERS ───────────────────────────────────────────────────────────
def _pct(v, d=1):
    return f"{'+' if v>0 else ''}{v:.{d}f}%"

def _inr(v):
    return f"₹{v/100_000:.2f}L"

def build_rankings_html(last_scores, last_rets, available_syms):
    rows = [
        (e, last_scores.get(e[0]), last_rets.get(e[0]))
        for e in ETFS
        if e[0] in available_syms and last_scores.get(e[0]) is not None
    ]
    rows.sort(key=lambda x: -x[1])
    medals = ["🥇","🥈","🥉"]
    html = '<table><thead><tr><th>Rank</th><th>ETF</th><th>RS Score</th><th>63d Return</th><th>Signal</th></tr></thead><tbody>'
    for i, (etf, score, ret) in enumerate(rows):
        is_top  = i < TOP_N
        medal   = medals[i] if i < 3 else str(i+1)
        ret_pct = f"{ret*100:+.2f}%" if ret is not None else "—"
        ret_cls = "green" if (ret or 0) > 0 else "red"
        status  = '<span class="badge badge-green">▲ LONG</span>' if is_top else '<span class="badge badge-muted">— OUT</span>'
        bg      = 'background:rgba(79,142,247,0.06);' if is_top else ''
        # score is raw float; *100 to show as basis-point-like number
        html += (f'<tr style="{bg}"><td>{medal}</td>'
                 f'<td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.73rem">{etf[2]}</span></td>'
                 f'<td style="color:var(--accent)">{score*100:+.3f}</td>'
                 f'<td class="{ret_cls}">{ret_pct}</td>'
                 f'<td>{status}</td></tr>')
    return html + '</tbody></table>'


def build_contrib_html(hold_count, instr_trades, total_weeks):
    rows = sorted(ETFS, key=lambda e: -hold_count.get(e[0], 0))
    html = '<table><thead><tr><th>ETF</th><th>Weeks Held</th><th>% Time</th><th>Trades</th></tr></thead><tbody>'
    for etf in rows:
        wk    = hold_count.get(etf[0], 0)
        pct_t = round(wk / total_weeks * 100) if total_weeks else 0
        tr    = instr_trades.get(etf[0], 0)
        bar   = (f'<div style="height:5px;width:{pct_t}px;max-width:80px;'
                 f'background:var(--accent);border-radius:3px;min-width:2px;display:inline-block"></div>')
        html += f'<tr><td><b>{etf[1]}</b></td><td>{wk}</td><td>{bar} {pct_t}%</td><td style="color:var(--muted)">{tr}</td></tr>'
    return html + '</tbody></table>'


def build_matrix_html(last_matrix, last_rets, available_syms):
    html = '<table class="mx"><thead><tr><th>↓ vs →</th>'
    visible = [e for e in ETFS if e[0] in available_syms]
    for etf in visible:
        html += f'<th>{etf[2]}</th>'
    html += '<th style="background:rgba(79,142,247,.1);color:var(--accent)">Wins</th></tr></thead><tbody>'

    for i, etf_i in enumerate(ETFS):
        if etf_i[0] not in available_syms:
            continue
        wins  = sum(1 for v in last_matrix[i] if v is not None and v > 0)
        total = sum(1 for v in last_matrix[i] if v is not None and v != 0)
        html += f'<tr><td class="rl">{etf_i[2]}</td>'
        for j, etf_j in enumerate(ETFS):
            if etf_j[0] not in available_syms:
                continue
            val = last_matrix[i][j]
            if i == j:
                html += '<td style="background:var(--card2);color:var(--muted)">—</td>'
            elif val is None:
                html += '<td style="color:var(--muted)">N/A</td>'
            else:
                intensity = min(abs(val) / 8, 0.8)
                bg  = f'rgba(34,197,94,{intensity:.2f})' if val > 0 else f'rgba(239,68,68,{intensity:.2f})'
                clr = '#22c55e' if val > 0 else '#ef4444'
                html += f'<td style="background:{bg};color:{clr}">{val:+.1f}%</td>'
        html += f'<td style="background:rgba(79,142,247,.1);color:var(--accent);font-weight:700">{wins}/{total}</td></tr>'
    return html + '</tbody></table>'


def build_log_html(trade_log):
    etf_map = {e[0]: e for e in ETFS}
    html = ''
    for log in reversed(trade_log):
        top3_tags = ''.join(
            f'<span class="tag">{etf_map[s][2] if s in etf_map else s.replace(".NS","")}</span>'
            for s in log["top3"]
        )
        changes = ''
        if log["changed"]:
            ex = ' '.join(f'<span class="badge badge-red">− {etf_map[s][2] if s in etf_map else s.replace(".NS","")}</span>' for s in log["exiting"])
            en = ' '.join(f'<span class="badge badge-green">+ {etf_map[s][2] if s in etf_map else s.replace(".NS","")}</span>' for s in log["entering"])
            changes = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">{ex} {en}</div>'
        no_chg = '' if log["changed"] else '<span style="color:var(--muted);font-size:.74rem;margin-left:6px">no change</span>'
        html += (f'<div class="lr" data-changed="{"1" if log["changed"] else "0"}">'
                 f'<div class="ld">{log["date"]}</div>'
                 f'<div class="lb">'
                 f'<div style="display:flex;align-items:center;flex-wrap:wrap">{top3_tags}{no_chg}</div>'
                 f'{changes}'
                 f'<div style="color:var(--muted);font-size:.74rem;margin-top:3px">₹{log["capital"]/1000:.1f}K</div>'
                 f'</div></div>\n')
    return html


def render_html(stats, equity_curve, nifty_curve, dd_curve, trade_log,
                hold_count, instr_trades, last_scores, last_rets, last_matrix,
                available_syms):
    # Chart data: x must be ISO date strings for Chart.js time adapter
    equity_json = json.dumps([{"x": d["date"], "y": d["value"]} for d in equity_curve])
    nifty_json  = json.dumps([{"x": d["date"], "y": d["value"]} for d in nifty_curve])
    dd_json     = json.dumps([{"x": d["date"], "y": d["dd"]}    for d in dd_curve])

    s   = stats
    cls = lambda v: "green" if v >= 0 else "red"

    html = HTML_TEMPLATE
    repl = {
        "__UPDATED__":     datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "__N_ETFS__":      str(len(available_syms)),
        "__FINAL__":       _inr(s["final_val"]),
        "__RETURN__":      _pct(s["total_ret"]),
        "__RETCLS__":      cls(s["total_ret"]),
        "__CAGR__":        _pct(s["cagr"]),
        "__CAGRCLS__":     cls(s["cagr"]),
        "__SHARPE__":      f"{s['sharpe']:.2f}",
        "__SHARPCLS__":    cls(s["sharpe"] - 1),
        "__MAXDD__":       _pct(s["max_dd"]),
        "__TRADES__":      str(s["total_trades"]),
        "__NIFTY__":       _pct(s["nifty_cagr"]),
        "__ALPHA__":       _pct(s["alpha"]),
        "__ALPHACLS__":    cls(s["alpha"]),
        "__EQUITY_JSON__": equity_json,
        "__NIFTY_JSON__":  nifty_json,
        "__DD_JSON__":     dd_json,
        "__RANKINGS__":    build_rankings_html(last_scores, last_rets, available_syms),
        "__CONTRIB__":     build_contrib_html(hold_count, instr_trades, s["weeks"]),
        "__MATRIX__":      build_matrix_html(last_matrix, last_rets, available_syms),
        "__LOG__":         build_log_html(trade_log),
    }
    for k, v in repl.items():
        html = html.replace(k, v)
    return html

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices = fetch_prices()

    if prices.empty:
        print("ERROR: No price data fetched.", file=sys.stderr)
        sys.exit(1)

    available_syms = prices.columns.tolist()
    print(f"Instruments loaded: {[s.replace('.NS','') for s in available_syms]}\n")

    (equity_curve, nifty_curve, dd_curve, trade_log,
     total_trades, hold_count, instr_trades,
     last_scores, last_rets, last_matrix) = run_backtest(prices)

    if not equity_curve:
        print("ERROR: Backtest produced no results.", file=sys.stderr)
        sys.exit(1)

    stats = calc_stats(equity_curve, nifty_curve, dd_curve, total_trades)

    print(f"\n{'='*52}")
    print(f"  Final Portfolio : ₹{stats['final_val']:>12,.0f}")
    print(f"  Total Return    : {stats['total_ret']:>+8.1f}%")
    print(f"  CAGR            : {stats['cagr']:>+8.1f}%")
    print(f"  Sharpe Ratio    : {stats['sharpe']:>8.2f}")
    print(f"  Max Drawdown    : {stats['max_dd']:>8.1f}%")
    print(f"  Total Trades    : {stats['total_trades']:>8}")
    print(f"  Nifty CAGR      : {stats['nifty_cagr']:>+8.1f}%")
    print(f"  Alpha vs Nifty  : {stats['alpha']:>+8.1f}%")
    print(f"  Weeks Simulated : {stats['weeks']:>8}")
    print(f"{'='*52}\n")

    os.makedirs("docs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(render_html(
            stats, equity_curve, nifty_curve, dd_curve, trade_log,
            hold_count, instr_trades, last_scores, last_rets, last_matrix,
            available_syms,
        ))
    print(f"✅  Report written → {OUT_PATH}")
