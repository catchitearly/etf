"""
India ETF Pairwise RS Matrix Backtester (Production)
==================================================
Universe  : 12 NSE ETFs
Signal    : 63-day pairwise relative strength
Portfolio : Long top-3 by RS score, equally weighted
Rebalance : Weekly (Signals on Friday Close -> Executed on Next Available Close)
Capital   : ₹10,00,000
Costs     : 0.1% per trade (brokerage + slippage)
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
import requests_cache
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ETFS = [
    # --- Commodities ---
    ("GOLDBEES.NS",    "GoldBees",      "GOLD"),
    ("SILVERBEES.NS",  "SilverBees",    "SILV"),

    # --- Core Broad Market Indices ---
    ("NIFTYBEES.NS",   "NiftyBees",      "NFTY"),
    ("JUNIORBEES.NS",  "JuniorBees",     "JNBR"),
    ("MID150BEES.NS",  "Midcap150Bees",  "MIDM"),
    ("NIF100BEES.NS",  "Nifty100Bees",   "NF10"),

    # --- Sectoral Indices ---
    ("BANKBEES.NS",    "BankBees",       "BANK"),
    ("ITBEES.NS",      "ITBees",         "ITMC"),
    ("PHARMABEES.NS",  "PharmaBees",     "PHRM"),
    ("AUTOBEES.NS",    "AutoBees",       "AUTO"),
    ("INFRABEES.NS",   "InfraBees",      "INFR"),
    ("CONSUMBEES.NS",  "ConsumeBees",    "CNSM"),

    # --- PSU, Government & Theme-Based ---
    ("PSUBNKBEES.NS",  "PSUBankBees",    "PSUB"),
    ("CPSEBEES.NS",    "CPSEBees",       "CPSE"),
    ("CPSEETF.NS",     "CPSE_ETF",       "CETF"),

    # --- Debt & Fixed Income (G-Sec) ---
    ("LTGILTBEES.NS",  "GS Composite",   "GSCP"),  # Formatted with .NS suffix for yfinance
    ("GILT5YBEES.NS",  "GSec5YearBees",  "GS5Y"),
    ("LIQUIDBEES.NS",  "LiquidBees",     "LIQD"),

    # --- Smart Beta, Factor & International ---
    ("MOM100.NS",      "Momentum100",    "MOM" ),
    ("MOMENTUM30.NS",  "Momentum30Bees", "MOM3"),
    ("NV20BEES.NS",    "Value20Bees",    "NV20"),
    ("DIVOPPBEES.NS",  "DivOppBees",     "DIVO"),
    ("HNGSNGBEES.NS",  "HangSengBees",   "HNGS"),
    ("MAFANG.NS",      "FANGPlusETF",    "FANG"),
    ("MON100.NS",      "Nasdaq100ETF",   "NSDQ"),
]

LOOKBACK     = 63
TOP_N        = 3
COST_PCT     = 0.001       # 0.1%
INITIAL      = 1_000_000   # ₹10 Lakh
FETCH_START  = "2022-01-01"
TRADE_START  = "2026-06-18"
END          = date.today().strftime("%Y-%m-%d")
NIFTY_SYM    = "NIFTYBEES.NS"
OUT_PATH     = "docs/index.html"

# ─── DATA ACQUISITION WITH RATELIMIT BYPASS ──────────────────────────────────
def setup_global_session():
    """Configures a clean global session to avoid GitHub runner IP bans."""
    session = requests_cache.CachedSession('yfinance_net.cache', expire_after=3600)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session

# ─── DATA ACQUISITION WITH LOCAL STORAGE CACHING ────────────────────────────
CACHE_DIR = ".cache/yfinance_data"

def _fetch_one(sym: str, retries: int = 4, delay: float = 4.0):
    """Download single ticker closing array with a direct local file system cache layer."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{sym}_{FETCH_START}_{END}.csv")
    
    # 1. Look in local file system cache first
    if os.path.exists(cache_path):
        try:
            # Verify file isn't empty or corrupted
            df_cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df_cached.empty:
                s = df_cached.iloc[:, 0].copy()
                s.name = sym
                return s
        except Exception:
            pass # Stale or broken file, fallback to network download

    # 2. Network Download via yfinance native engine (No custom session overrides)
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=sym, 
                start=FETCH_START, 
                end=END, 
                progress=False, 
                timeout=25,
                auto_adjust=True
            )
            if df is None or df.empty:
                raise ValueError("Empty dataframe returned from API")
            
            # Extract closing array structural variations cleanly
            if "Close" in df.columns and isinstance(df.columns, pd.MultiIndex):
                s = df["Close"][sym].copy()
            elif "Close" in df.columns:
                s = df["Close"].copy()
            else:
                s = df.iloc[:, 0].copy()
                
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.name = sym

            # Save valid data straight to the workflow run cache directory
            s.to_csv(cache_path)
            return s

        except Exception as e:
            if attempt == retries:
                print(f"    ❌ final attempt failed: {e}")
            time.sleep(delay * attempt + random.uniform(0.5, 2.0))
    return None

def fetch_prices() -> pd.DataFrame:
    print(f"Fetching {len(ETFS)} ETFs individually from {FETCH_START} to {END} ...")
    series_list = []

    for i, (sym, name, short) in enumerate(ETFS, 1):
        print(f"  [{i:>2}/{len(ETFS)}] {name:<18} ({sym})", end=" ... ", flush=True)
        s = _fetch_one(sym) # No session object passed downstream
        if s is not None and s.notna().sum() > 50:
            series_list.append(s)
            print(f"✓ {s.notna().sum()} rows")
        else:
            print("✗ skipped")
        time.sleep(random.uniform(1.0, 2.0))

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1)
    prices = prices.sort_index().ffill(limit=5)
    valid_cols = [c for c in prices.columns if prices[c].notna().sum() > 100]
    return prices[valid_cols].dropna(how="all")

# ─── ENGINE HELPERS ──────────────────────────────────────────────────────────
def period_return(series: pd.Series, idx_now: int, lookback_days: int):
    idx_past = max(idx_now - lookback_days, 0)
    p_now = series.iloc[idx_now]
    p_past = series.iloc[idx_past]
    if pd.isna(p_now) or pd.isna(p_past) or p_past == 0:
        return 0.0
    return (p_now / p_past) - 1.0

def compute_rs(prices: pd.DataFrame, idx: int, available_syms: list):
    rets = {sym: period_return(prices[sym], idx, LOOKBACK) for sym in available_syms if sym in prices.columns}
    scores = {}
    valid = [s for s, r in rets.items() if r is not None]
    
    for sym in available_syms:
        if sym not in rets or rets[sym] is None:
            scores[sym] = None
            continue
        others = [rets[s] for s in valid if s != sym]
        scores[sym] = float(np.mean([rets[sym] - o for o in others])) if others else 0.0
    return scores, rets

def build_matrix(rets: dict, available_syms: list):
    matrix = []
    for i, (si, _, _) in enumerate(ETFS):
        row = []
        for j, (sj, _, _) in enumerate(ETFS):
            if i == j:
                row.append(0.0)
            elif si in available_syms and sj in available_syms:
                row.append(round((rets.get(si, 0) - rets.get(sj, 0)) * 100, 3))
            else:
                row.append(None)
        matrix.append(row)
    return matrix

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(prices: pd.DataFrame):
    available_syms = prices.columns.tolist()
    all_fridays = pd.date_range(TRADE_START, END, freq="W-FRI")
    
    snapped = []
    for f in all_fridays:
        pos = prices.index.searchsorted(f, side="right") - 1
        if 0 <= pos < len(prices):
            snapped.append(pos)
    snapped = sorted(set(snapped))

    cash = float(INITIAL)
    holdings = {}  # sym -> shares
    cur_top3 = []
    peak = float(INITIAL)
    total_trades = 0
    hold_count = {e[0]: 0 for e in ETFS}
    instr_trades = {e[0]: 0 for e in ETFS}

    equity_curve, nifty_curve, dd_curve, trade_log = [], [], [], []
    nifty_start_idx = max(prices.index.searchsorted(pd.Timestamp(TRADE_START)) - 1, 0)
    nifty_start_px = prices[NIFTY_SYM].iloc[nifty_start_idx] if NIFTY_SYM in prices.columns else 1.0

    last_scores = last_rets = last_matrix = None

    for wi, idx in enumerate(snapped):
        date_str = str(prices.index[idx].date())
        scores, rets = compute_rs(prices, idx, available_syms)
        
        valid_rank = [(s, v) for s, v in scores.items() if v is not None]
        valid_rank.sort(key=lambda x: -x[1])
        new_top3 = [s for s, _ in valid_rank[:TOP_N]]

        if len(new_top3) < TOP_N:
            continue

        needs_rebal = (wi == 0) or (set(new_top3) != set(cur_top3))
        exiting = [s for s in cur_top3 if s not in new_top3]
        entering = [s for s in new_top3 if s not in cur_top3]

        if needs_rebal:
            # Smart Sell: Drop structural exits to preserve frictional drag drops
            for sym in exiting:
                if sym in holdings:
                    px = float(prices[sym].iloc[idx])
                    cash += holdings[sym] * px * (1 - COST_PCT)
                    instr_trades[sym] += 1
                    total_trades += 1
                    del holdings[sym]

            # Re-verify dynamic cash values
            retained_val = sum(holdings[s] * float(prices[s].iloc[idx]) for s in holdings if s in prices.columns)
            total_portfolio_value = cash + retained_val
            target_per_position = total_portfolio_value / TOP_N

            # Smart Balanced Layer deployment
            for sym in new_top3:
                px = float(prices[sym].iloc[idx])
                current_val = holdings.get(sym, 0) * px
                
                if sym in entering or abs(current_val - target_per_position) > (target_per_position * 0.05):
                    if sym in holdings:
                        cash += holdings[sym] * px * (1 - COST_PCT)
                        del holdings[sym]
                    
                    holdings[sym] = (target_per_position * (1 - COST_PCT)) / px
                    cash -= target_per_position
                    instr_trades[sym] += 1
                    total_trades += 1

        # Mark-to-market valuations
        port_val = cash + sum(shares * float(prices[sym].iloc[idx]) for sym, shares in holdings.items())
        
        for sym in new_top3:
            hold_count[sym] += 1

        nifty_px = float(prices[NIFTY_SYM].iloc[idx]) if NIFTY_SYM in prices.columns else nifty_start_px
        nifty_val = (nifty_px / float(nifty_start_px)) * INITIAL

        if port_val > peak: peak = port_val
        dd = (port_val - peak) / peak * 100

        equity_curve.append({"date": date_str, "value": round(port_val, 2)})
        nifty_curve.append({"date": date_str, "value": round(nifty_val, 2)})
        dd_curve.append({"date": date_str, "dd": round(dd, 3)})

        trade_log.append({
            "date": date_str, "top3": new_top3, "exiting": exiting, "entering": entering,
            "changed": needs_rebal and wi > 0, "capital": round(port_val, 2),
            "scores": {s: round(v * 100, 3) if v is not None else None for s, v in scores.items()},
            "rets": {s: round(v * 100, 3) if v is not None else None for s, v in rets.items()}
        })

        cur_top3 = new_top3
        last_scores = scores
        last_rets = rets
        last_matrix = build_matrix(rets, available_syms)

    return equity_curve, nifty_curve, dd_curve, trade_log, total_trades, hold_count, instr_trades, last_scores, last_rets, last_matrix

# ─── PERFORMANCE METRICS ─────────────────────────────────────────────────────
def calc_stats(equity_curve, nifty_curve, total_trades, dd_curve):
    final_val = equity_curve[-1]["value"]
    nifty_final = nifty_curve[-1]["value"]
    
    t0, t1 = pd.Timestamp(TRADE_START), pd.Timestamp(END)
    years = max((t1 - t0).days / 365.25, 0.05)

    total_ret = (final_val - INITIAL) / INITIAL * 100
    cagr = (pow(final_val / INITIAL, 1 / years) - 1) * 100
    nifty_cagr = (pow(nifty_final / INITIAL, 1 / years) - 1) * 100
    
    weekly_rets = [(equity_curve[i]["value"] - equity_curve[i-1]["value"]) / equity_curve[i-1]["value"] for i in range(1, len(equity_curve))]
    sharpe = (np.mean(weekly_rets) / np.std(weekly_rets)) * np.sqrt(52) if np.std(weekly_rets) > 0 else 0
    max_dd = min([d["dd"] for d in dd_curve]) if dd_curve else 0

    return dict(
        final_val=round(final_val, 2), total_ret=round(total_ret, 2), cagr=round(cagr, 2),
        sharpe=round(sharpe, 2), max_dd=round(max_dd, 2), total_trades=total_trades,
        nifty_cagr=round(nifty_cagr, 2), alpha=round(cagr - nifty_cagr, 2), weeks=len(equity_curve)
    )

# ─── HTML DASHBOARD DESIGN TEMPLATE ──────────────────────────────────────────
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
.mx .rl{font-weight:600;color:var(--text);background:var(--card2);text-align:left;padding-left:10px}
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
    <div class="meta">12 ETFs · 63-day RS · Weekly Rebalance · Top-3 Long · ₹10L · Jan 2023 → present</div>
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
  <div class="section-title">Current RS Rankings</div>
  __RANKINGS__
</div>
<div class="chart-card">
  <div class="section-title">Time in Portfolio</div>
  __CONTRIB__
</div>
</div>

<div class="chart-card">
  <div class="section-title">Pairwise RS Matrix — last week (row outperforms column, 63d)</div>
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

<div class="updated">Generated by GitHub Actions · yfinance · Strategy by RS Matrix</div>
</div>

<script>
const edata = __EQUITY_JSON__;
const ndata = __NIFTY_JSON__;
const ddata = __DD_JSON__;

const gridColor = '#2e3250';
const eCtx = document.getElementById('ec').getContext('2d');
new Chart(eCtx,{type:'line',data:{datasets:[
  {label:'RS Strategy',data:edata,borderColor:'#4f8ef7',backgroundColor:'rgba(79,142,247,0.08)',borderWidth:2,pointRadius:0,tension:0.3,fill:true},
  {label:'NiftyBees',data:ndata,borderColor:'#f59e0b',backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,borderDash:[5,4],tension:0.3}
]},options:{responsive:true,maintainAspectRatio:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#8892b0',font:{size:12}}},
    tooltip:{backgroundColor:'#1a1d27',titleColor:'#e2e8f0',bodyColor:'#8892b0',
      callbacks:{label:c=>`${c.dataset.label}: ₹${(c.raw.y/1000).toFixed(1)}K`}}},
  scales:{
    x:{type:'time',time:{unit:'month'},ticks:{color:'#8892b0',maxTicksLimit:20},grid:{color:gridColor}},
    y:{ticks:{color:'#8892b0',callback:v=>'₹'+(v/1000).toFixed(0)+'K'},grid:{color:gridColor}}
  }}});

const dCtx = document.getElementById('dc').getContext('2d');
new Chart(dCtx,{type:'line',data:{datasets:[
  {label:'Drawdown',data:ddata,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.12)',borderWidth:1.5,pointRadius:0,tension:0.3,fill:true}
]},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{backgroundColor:'#1a1d27',callbacks:{label:c=>`DD: ${c.raw.y.toFixed(2)}%`}}},
  scales:{
    x:{type:'time',time:{unit:'month'},ticks:{color:'#8892b0',maxTicksLimit:20},grid:{color:gridColor}},
    y:{ticks:{color:'#8892b0',callback:v=>v.toFixed(1)+'%'},grid:{color:gridColor}}
  }}});

function switchTab(t,el){
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.lr').forEach(r=>{
    if(t==='all') r.style.display='flex';
    else r.style.display = r.dataset.changed==='1' ? 'flex' : 'none';
  });
}
</script>
</body>
</html>
"""

def pct(v, decimals=1):
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def inr(v):
    return f"₹{v/100_000:.2f}L"

def build_rankings_html(last_scores, last_rets, available_syms):
    rows = [(e, last_scores.get(e[0]), last_rets.get(e[0])) for e in ETFS if e[0] in available_syms and last_scores.get(e[0]) is not None]
    rows.sort(key=lambda x: -x[1])
    medals = ["🥇","🥈","🥉"]
    html = '<table><thead><tr><th>Rank</th><th>ETF</th><th>RS Score</th><th>63d Return</th><th>Signal</th></tr></thead><tbody>'
    for i, (etf, score, ret) in enumerate(rows):
        is_top = i < TOP_N
        medal  = medals[i] if i < 3 else str(i+1)
        ret_s  = f"{ret*100:+.1f}%" if ret is not None else "—"
        ret_cls= "green" if (ret or 0) > 0 else "red"
        status = '<span class="badge badge-green">▲ LONG</span>' if is_top else '<span class="badge badge-muted">— OUT</span>'
        row_bg = 'background:rgba(79,142,247,0.05);' if is_top else ''
        html += f'<tr style="{row_bg}"><td>{medal}</td><td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.73rem">{etf[2]}</span></td><td style="color:var(--accent)">{score*100:.3f}</td><td class="{ret_cls}">{ret_s}</td><td>{status}</td></tr>'
    return html + '</tbody></table>'

def build_contrib_html(hold_count, instr_trades, total_weeks):
    rows = sorted(ETFS, key=lambda e: -hold_count.get(e[0], 0))
    html = '<table><thead><tr><th>ETF</th><th>Weeks Held</th><th>% Time</th><th>Trades</th></tr></thead><tbody>'
    for etf in rows:
        wk   = hold_count.get(etf[0], 0)
        pct_t = round(wk / total_weeks * 100) if total_weeks else 0
        tr   = instr_trades.get(etf[0], 0)
        bar = f'<div style="height:5px;width:{pct_t}px;max-width:80px;background:var(--accent);border-radius:3px;min-width:2px;display:inline-block"></div>'
        html += f'<tr><td><b>{etf[1]}</b></td><td>{wk}</td><td>{bar} {pct_t}%</td><td style="color:var(--muted)">{tr}</td></tr>'
    return html + '</tbody></table>'

def build_matrix_html(last_matrix, last_rets, available_syms):
    html = '<table class="mx"><thead><tr><th>↓ vs →</th>'
    for etf in ETFS:
        html += f'<th>{etf[2]}</th>'
    html += '<th style="background:rgba(79,142,247,.1);color:var(--accent)">Wins</th></tr></thead><tbody>'

    for i, etf_i in enumerate(ETFS):
        if etf_i[0] not in available_syms:
            continue
        wins = sum(1 for v in last_matrix[i] if v is not None and v > 0)
        html += f'<tr><td class="rl">{etf_i[2]}</td>'
        for j, etf_j in enumerate(ETFS):
            val = last_matrix[i][j]
            if i == j:
                html += '<td style="background:var(--card2);color:var(--muted)">—</td>'
            elif val is None:
                html += '<td style="color:var(--muted)">N/A</td>'
            else:
                intensity = min(abs(val) / 10, 0.75)
                bg  = f'rgba(34,197,94,{intensity:.2f})' if val > 0 else f'rgba(239,68,68,{intensity:.2f})'
                clr = '#22c55e' if val > 0 else '#ef4444'
                html += f'<td style="background:{bg};color:{clr}">{val:+.1f}%</td>'
        total = sum(1 for v in last_matrix[i] if v is not None and v != 0)
        html += f'<td style="background:rgba(79,142,247,.1);color:var(--accent);font-weight:700">{wins}/{total}</td></tr>'
    return html + '</tbody></table>'

def build_log_html(trade_log):
    etf_map = {e[0]: e for e in ETFS}
    html = ''
    for log in reversed(trade_log):
        top3_tags = ''.join(f'<span class="tag">{etf_map[s][2] if s in etf_map else s}</span>' for s in log["top3"])
        changes = ''
        if log["changed"]:
            ex = ' '.join(f'<span class="badge badge-red">− {etf_map[s][2] if s in etf_map else s}</span>' for s in log["exiting"])
            en = ' '.join(f'<span class="badge badge-green">+ {etf_map[s][2] if s in etf_map else s}</span>' for s in log["entering"])
            changes = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">{ex} {en}</div>'
        no_change = '<span style="color:var(--muted);font-size:.74rem;margin-left:6px">no change</span>' if not log["changed"] else ''
        changed_attr = '1' if log["changed"] else '0'
        html += f'''<div class="lr" data-changed="{changed_attr}">
  <div class="ld">{log["date"]}</div>
  <div class="lb">
    <div style="display:flex;align-items:center;flex-wrap:wrap">{top3_tags}{no_change}</div>
    {changes}
    <div style="color:var(--muted);font-size:.74rem;margin-top:3px">₹{log["capital"]/1000:.1f}K</div>
  </div>
</div>'''
    return html

def render_html(stats, equity_curve, nifty_curve, dd_curve, trade_log,
                hold_count, instr_trades, last_scores, last_rets, last_matrix,
                available_syms):

    equity_json = json.dumps([{"x": d["date"], "y": d["value"]} for d in equity_curve])
    nifty_json  = json.dumps([{"x": d["date"], "y": d["value"]} for d in nifty_curve])
    dd_json     = json.dumps([{"x": d["date"], "y": d["dd"]}    for d in dd_curve])

    s = stats
    cls = lambda v: "green" if v >= 0 else "red"

    html = HTML_TEMPLATE
    html = html.replace("__UPDATED__",    datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    html = html.replace("__FINAL__",     inr(s["final_val"]))
    html = html.replace("__RETURN__",    pct(s["total_ret"]))
    html = html.replace("__RETCLS__",    cls(s["total_ret"]))
    html = html.replace("__CAGR__",     pct(s["cagr"]))
    html = html.replace("__CAGRCLS__",  cls(s["cagr"]))
    html = html.replace("__SHARPE__",    f"{s['sharpe']:.2f}")
    html = html.replace("__SHARPCLS__", cls(s["sharpe"] - 1))
    html = html.replace("__MAXDD__",    pct(s["max_dd"]))
    html = html.replace("__TRADES__",   str(s["total_trades"]))
    html = html.replace("__NIFTY__",    pct(s["nifty_cagr"]))
    html = html.replace("__ALPHA__",    pct(s["alpha"]))
    html = html.replace("__ALPHACLS__", cls(s["alpha"]))
    html = html.replace("__EQUITY_JSON__", equity_json)
    html = html.replace("__NIFTY_JSON__",  nifty_json)
    html = html.replace("__DD_JSON__",     dd_json)
    html = html.replace("__RANKINGS__",    build_rankings_html(last_scores, last_rets, available_syms))
    html = html.replace("__CONTRIB__",     build_contrib_html(hold_count, instr_trades, len(equity_curve)))
    html = html.replace("__MATRIX__",      build_matrix_html(last_matrix, last_rets, available_syms))
    html = html.replace("__LOG__",         build_log_html(trade_log))
    return html

# ─── MAIN EXECUTION ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices = fetch_prices()

    if prices.empty:
        print("ERROR: No price data fetched. Network/Scraping blocked.", file=sys.stderr)
        sys.exit(1)

    available_syms = prices.columns.tolist()
    print(f"Available Elements: {[s.split('.')[0] for s in available_syms]}\n")

    (equity_curve, nifty_curve, dd_curve, trade_log,
     total_trades, hold_count, instr_trades,
     last_scores, last_rets, last_matrix) = run_backtest(prices)

    if not equity_curve:
        print("ERROR: Backtest pipeline crash.", file=sys.stderr)
        sys.exit(1)

    stats = calc_stats(equity_curve, nifty_curve, total_trades, dd_curve)

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
    report_html = render_html(
        stats, equity_curve, nifty_curve, dd_curve, trade_log,
        hold_count, instr_trades, last_scores, last_rets, last_matrix,
        available_syms,
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_html)

    print(f"✅ Report deployment written → {OUT_PATH}")
