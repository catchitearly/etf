"""
Fixed 45-Day RS Backtester — v3
====================================
Key changes from v2:
  1. FIXED LOOKBACK: 45-day always (no VIX regime switching)
  2. PERIODS:
     - Backtest: 2019-01-01 → 2024-12-31
     - Forward 1: 2025-01-01 → 2025-12-31
     - Forward 2: 2026-01-01 → today
  3. Full trade log with all changes
  4. Current positions display
  5. Comprehensive HTML dashboard
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time, random
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
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

NIFTY_SYM = "NIFTYBEES.NS"
FIXED_LB = 45
TOP_N = 3
COST_PCT = 0.001
INITIAL = 1_000_000

# Period definitions
BT_START = "2019-01-01"
BT_END = "2024-12-31"
FT1_START = "2025-01-01"
FT1_END = "2025-12-31"
FT2_START = "2026-01-01"
FT2_END = date.today().strftime("%Y-%m-%d")

FETCH_START = "2018-01-01"  # Need extra data for 45-day lookback
END = date.today().strftime("%Y-%m-%d")

CACHE_DIR = ".cache/fixed45"
OUT_PATH = "docs/index_fixed45.html"
RS_TABLE_PATH = "docs/rs_table_45d_fixed.html"

# ── FETCH ─────────────────────────────────────────────────────────────────────
def _cache_path(sym):
    safe = sym.replace("^","_")
    return os.path.join(CACHE_DIR, f"{safe}_{FETCH_START}_{END}.csv")

def _fetch_one(sym, retries=4):
    cp = _cache_path(sym)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cp):
        try:
            df = pd.read_csv(cp, index_col=0, parse_dates=True)
            if not df.empty and df.index[0] <= pd.Timestamp(FETCH_START) + timedelta(days=30):
                s = df.iloc[:,0].copy(); s.name = sym; return s
            os.remove(cp)
        except Exception:
            pass

    for attempt in range(1, retries+1):
        try:
            df = yf.download(sym, start=FETCH_START, end=END,
                             progress=False, timeout=30, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("empty — Yahoo Finance may be blocked")
            if isinstance(df.columns, pd.MultiIndex):
                col = "Close"
                s = df[col][sym].copy() if sym in df[col].columns else df[col].iloc[:,0].copy()
            else:
                s = df["Close"].copy()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.name = sym
            s.to_csv(cp)
            return s
        except Exception as e:
            if attempt == retries: print(f"    failed {sym}: {e}")
            time.sleep(3*attempt + random.uniform(0,2))
    return None

def fetch_all():
    print(f"Fetching {len(ETFS)} instruments | {FETCH_START} -> {END}")
    series = {}

    for i,(sym,name,short) in enumerate(ETFS,1):
        print(f"  [{i:>2}/{len(ETFS)}] {name:<16} ({sym})", end=" ... ", flush=True)
        s = _fetch_one(sym)
        if s is not None and s.notna().sum() > 100:
            series[sym] = s
            print(f"ok ({s.notna().sum()} rows)")
        else:
            print("skip")
        time.sleep(random.uniform(0.8, 1.5))

    if not series:
        return pd.DataFrame()

    etf_syms = [sym for sym,_,_ in ETFS if sym in series]
    prices = pd.concat([series[s] for s in etf_syms], axis=1)
    prices.columns = etf_syms
    prices = prices.sort_index().ffill(limit=5)
    valid = [c for c in prices.columns if prices[c].notna().sum() > 200]
    prices = prices[valid].dropna(how="all")

    print(f"\n  ETFs: {len(valid)}")
    print(f"  Price range: {prices.index[0].date()} -> {prices.index[-1].date()}\n")
    return prices

# ── RS ENGINE ─────────────────────────────────────────────────────────────────
def period_return(series, idx_now, lb):
    idx_past = idx_now - lb
    if idx_past < 0: return None
    p0, p1 = series.iloc[idx_past], series.iloc[idx_now]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0: return None
    return float(p1/p0 - 1.0)

def compute_rs(prices, idx, avail, lb=FIXED_LB):
    rets = {s: period_return(prices[s], idx, lb) for s in avail if s in prices.columns}
    valid = [s for s in rets if rets[s] is not None]
    scores = {}
    for sym in avail:
        if rets.get(sym) is None: scores[sym] = None; continue
        peers = [rets[s] for s in valid if s != sym]
        scores[sym] = float(np.mean([rets[sym]-p for p in peers])) if peers else 0.0
    return scores, rets

# ── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def run_backtest(prices, period_start, period_end, label, initial_capital=None):
    """
    Run fixed 45-day RS backtest for a given period.
    Returns full equity curve, log, and stats dict.
    """
    print(f"\nRunning [{label}]: {period_start} → {period_end} ...")
    avail = prices.columns.tolist()

    data_start = prices.index[0]
    start_dt = max(pd.Timestamp(period_start),
                   data_start + pd.Timedelta(days=FIXED_LB + 5))

    fridays = pd.date_range(start_dt, period_end, freq="W-FRI")
    pairs = []
    for f in fridays:
        si = prices.index.searchsorted(f, side="right") - 1
        ei = si + 1
        if 0 <= si < len(prices) and ei < len(prices):
            pairs.append((si, ei))

    if not pairs:
        print(f"  No trading weeks found for {label}")
        return None

    cap = float(initial_capital or INITIAL)
    cash = cap
    holdings = {}
    cur3 = []
    peak = cap
    total_trades = 0
    hold_count = {e[0]: 0 for e in ETFS}
    
    eq = []
    nf = []
    dd_curve = []
    trade_log = []  # Detailed trade log
    weekly_log = []  # Weekly decision log
    
    nifty_px0 = (float(prices[NIFTY_SYM].iloc[pairs[0][1]])
                 if pairs and NIFTY_SYM in prices.columns else None)
    
    last_scores = last_rets = None

    for wi, (si, ei) in enumerate(pairs):
        date_str = str(prices.index[si].date())
        exec_str = str(prices.index[ei].date())

        # ── Compute RS scores ───────────────────────────────────────────────
        scores, rets = compute_rs(prices, si, avail, FIXED_LB)
        ranked = sorted([(s, v) for s, v in scores.items() if v is not None],
                        key=lambda x: -x[1])
        new3 = [s for s, _ in ranked[:TOP_N]]
        last_scores = scores
        last_rets = rets

        if len(new3) < 1:
            continue

        changed = (wi == 0) or (set(new3) != set(cur3))
        exiting = [s for s in cur3 if s not in new3] if not wi == 0 else []
        entering = [s for s in new3 if s not in cur3] if not wi == 0 else []

        # ── Execute trades if changed ──────────────────────────────────────
        trades_executed = []
        if changed:
            # Step 1: sell ALL holdings at execution price
            for sym, shares in list(holdings.items()):
                px = prices[sym].iloc[ei] if sym in prices.columns else float('nan')
                if shares > 0 and np.isfinite(px) and float(px) > 0:
                    proceeds = shares * float(px) * (1.0 - COST_PCT)
                    cash += proceeds
                    trades_executed.append({
                        'date': exec_str,
                        'symbol': sym,
                        'action': 'SELL',
                        'shares': round(shares, 4),
                        'price': round(float(px), 2),
                        'value': round(proceeds, 2)
                    })
                    total_trades += 1
            holdings.clear()
            cash = max(cash, 0.0)

            # Step 2: buy equal slices of new TOP_N
            target = cash / TOP_N
            for sym in new3:
                if sym not in prices.columns: continue
                px = float(prices[sym].iloc[ei])
                if not np.isfinite(px) or px <= 0: continue
                buy_amt = min(target, max(cash, 0.0))
                if buy_amt > 0:
                    shares = buy_amt * (1.0 - COST_PCT) / px
                    holdings[sym] = shares
                    cash -= buy_amt
                    trades_executed.append({
                        'date': exec_str,
                        'symbol': sym,
                        'action': 'BUY',
                        'shares': round(shares, 4),
                        'price': round(float(px), 2),
                        'value': round(buy_amt, 2)
                    })
                    total_trades += 1
            cash = max(cash, 0.0)

        # ── Mark-to-market ──────────────────────────────────────────────────
        holdings_val = 0.0
        for sym, shares in holdings.items():
            px = float(prices[sym].iloc[ei]) if sym in prices.columns else float('nan')
            if np.isfinite(px) and px > 0:
                holdings_val += shares * px
        port = max(cash + holdings_val, 0.0)
        if port > peak: peak = port
        ddown = (port - peak) / peak * 100 if peak > 0 else 0

        nifty_px = (float(prices[NIFTY_SYM].iloc[ei])
                    if NIFTY_SYM in prices.columns else None)
        nifty_v = (nifty_px / nifty_px0 * (initial_capital or INITIAL)
                   if nifty_px and nifty_px0 else (initial_capital or INITIAL))

        for sym in new3:
            hold_count[sym] = hold_count.get(sym, 0) + 1

        eq.append({"x": date_str, "y": round(port, 2)})
        nf.append({"x": date_str, "y": round(nifty_v, 2)})
        dd_curve.append({"x": date_str, "y": round(ddown, 3)})
        
        # Weekly decision log
        weekly_log.append({
            "date": date_str,
            "exec": exec_str,
            "top3": new3,
            "exiting": exiting,
            "entering": entering,
            "changed": changed and wi > 0,
            "capital": round(port, 2),
            "holdings": {s: round(holdings.get(s, 0), 4) for s in new3},
            "scores": {s: round(scores.get(s, 0), 6) if scores.get(s) else None for s in new3},
            "rets": {s: round(rets.get(s, 0) * 100, 2) if rets.get(s) else None for s in new3}
        })
        
        # Add trades to log if any
        if trades_executed:
            trade_log.extend(trades_executed)
            
        cur3 = new3

    return {
        'eq': eq,
        'nf': nf,
        'dd': dd_curve,
        'weekly_log': weekly_log,
        'trade_log': trade_log,
        'trades': total_trades,
        'hold_count': hold_count,
        'last_scores': last_scores,
        'last_rets': last_rets,
        'final_holdings': {s: round(holdings.get(s, 0), 4) for s in holdings},
        'avail': avail,
        'label': label,
        'period_start': period_start,
        'period_end': period_end,
    }

# ── STATS ─────────────────────────────────────────────────────────────────────
def calc_stats(result, initial=None):
    if not result or not result['eq']:
        return None
    
    eq = result["eq"]
    nf = result["nf"]
    dd = result["dd"]
    trades = result["trades"]
    
    init = initial or INITIAL
    fv = eq[-1]["y"]
    nfv = nf[-1]["y"]
    ps = pd.Timestamp(result["period_start"])
    pe = pd.Timestamp(result["period_end"])
    yrs = max((pe - ps).days / 365.25, 0.1)
    tr = (fv - init) / init * 100
    cagr = (pow(fv / init, 1 / yrs) - 1) * 100 if fv > 0 and init > 0 else 0
    nc = (pow(nfv / init, 1 / yrs) - 1) * 100 if nfv > 0 and init > 0 else 0
    
    wr = [(eq[i]["y"] - eq[i-1]["y"]) / eq[i-1]["y"] 
          for i in range(1, len(eq)) if eq[i-1]["y"] > 0]
    shp = (np.mean(wr) / np.std(wr)) * np.sqrt(52) if wr and np.std(wr) > 0 else 0
    
    mdd = min(d["y"] for d in dd) if dd else 0
    
    # Calculate number of rebalances (weeks with changes)
    rebalances = sum(1 for log in result["weekly_log"] if log["changed"])
    
    return {
        'final': round(fv, 2),
        'ret': round(tr, 2),
        'cagr': round(cagr, 2),
        'sharpe': round(shp, 2),
        'mdd': round(mdd, 2),
        'trades': trades,
        'rebalances': rebalances,
        'nifty_cagr': round(nc, 2),
        'alpha': round(cagr - nc, 2),
        'weeks': len(eq)
    }

# ── CURRENT SIGNAL ────────────────────────────────────────────────────────────
def get_current_signal(prices, ft2_result):
    if not ft2_result or not ft2_result["weekly_log"]:
        return None
    
    last = ft2_result["weekly_log"][-1]
    em = {e[0]: e for e in ETFS}
    
    current_prices = {}
    for sym in last["top3"]:
        if sym in prices.columns:
            current_prices[sym] = round(float(prices[sym].dropna().iloc[-1]), 2)
    
    # Get current holdings
    current_holdings = ft2_result["final_holdings"]
    
    return {
        'signal_date': last["date"],
        'exec_date': last["exec"],
        'top3': last["top3"],
        'top3_names': [em[s][1] if s in em else s for s in last["top3"]],
        'top3_short': [em[s][2] if s in em else s for s in last["top3"]],
        'scores': {s: round(last["scores"].get(s, 0) or 0, 6) for s in last["top3"]},
        'rets': {s: round(last["rets"].get(s, 0) or 0, 2) for s in last["top3"]},
        'holdings': current_holdings,
        'current_prices': current_prices,
        'portfolio_val': last["capital"],
        'total_value': last["capital"]
    }

# ── BUILD RS TABLE ──────────────────────────────────────────────────────────
def build_rs_table_html(prices):
    """
    Build standalone HTML page showing full RS table for 45-day lookback.
    """
    avail = [c for c in prices.columns if prices[c].notna().sum() > 50]
    last_idx = len(prices) - 1
    
    scores_45, rets_45 = compute_rs(prices, last_idx, avail, 45)
    
    em = {e[0]: e for e in ETFS}
    table_rows = []
    for sym in avail:
        if sym not in scores_45 or scores_45[sym] is None:
            continue
        ret_pct = (rets_45.get(sym, 0) or 0) * 100
        score = scores_45[sym] or 0
        name = em.get(sym, (sym, sym, sym))[1]
        short = em.get(sym, (sym, sym, sym))[2]
        price = float(prices[sym].iloc[-1]) if sym in prices.columns else None
        table_rows.append({
            'symbol': sym,
            'name': name,
            'short': short,
            'price': price,
            'return': ret_pct,
            'score': score,
            'rank': 0
        })
    
    table_rows.sort(key=lambda x: -x['score'])
    for i, row in enumerate(table_rows, 1):
        row['rank'] = i
        row['top3'] = i <= 3
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>45-Day Relative Strength Table — All Instruments</title>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--border:#2e3250;--accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#8892b0}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:30px}
.container{max-width:1200px;margin:0 auto}
h1{color:var(--accent);font-size:1.5rem;margin-bottom:6px}
.subtitle{color:var(--muted);font-size:.85rem;margin-bottom:24px}
table{width:100%;border-collapse:collapse;font-size:.85rem;background:var(--card);border-radius:10px;overflow:hidden}
th{background:var(--border);color:var(--text);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;padding:12px 14px;text-align:left}
td{padding:10px 14px;border-bottom:1px solid var(--border)}
tr:hover{background:rgba(79,142,247,0.05)}
tr.top3{background:rgba(34,197,94,0.06)}
tr.top3:hover{background:rgba(34,197,94,0.12)}
.rank-badge{display:inline-block;width:28px;height:28px;border-radius:50%;text-align:center;line-height:28px;font-weight:700;font-size:.8rem}
.rank-1{background:rgba(255,215,0,0.2);color:#ffd700}
.rank-2{background:rgba(192,192,192,0.2);color:#c0c0c0}
.rank-3{background:rgba(205,127,50,0.2);color:#cd7f32}
.rank-other{color:var(--muted)}
.green{color:var(--green)}
.red{color:var(--red)}
.accent{color:var(--accent)}
.muted{color:var(--muted)}
.text-right{text-align:right}
.text-center{text-align:center}
.stats-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:24px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 18px}
.stat-label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-value{font-size:1.3rem;font-weight:700;margin-top:3px}
.export-btn{background:var(--accent);color:#fff;border:none;padding:10px 24px;border-radius:8px;font-weight:600;cursor:pointer;font-size:.85rem;margin-bottom:20px}
.export-btn:hover{opacity:.85}
</style>
</head>
<body>
<div class="container">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:8px">
        <div>
            <h1>📊 45-Day Relative Strength Table</h1>
            <div class="subtitle">
                All instruments ranked by RS score · Latest data: """ + str(prices.index[-1].date()) + """
            </div>
        </div>
        <div>
            <button class="export-btn" onclick="downloadCSV()">📥 Download CSV</button>
            <button class="export-btn" onclick="window.print()" style="background:var(--border)">🖨️ Print</button>
        </div>
    </div>

    <div class="stats-bar">"""
    
    total = len(table_rows)
    avg_score = np.mean([r['score'] for r in table_rows]) if table_rows else 0
    max_score = max([r['score'] for r in table_rows]) if table_rows else 0
    min_score = min([r['score'] for r in table_rows]) if table_rows else 0
    
    html += f"""
        <div class="stat-card"><div class="stat-label">Total Instruments</div><div class="stat-value">{total}</div></div>
        <div class="stat-card"><div class="stat-label">Average RS Score</div><div class="stat-value" style="color:var(--accent)">{avg_score:.4f}</div></div>
        <div class="stat-card"><div class="stat-label">Highest RS</div><div class="stat-value" style="color:var(--green)">{max_score:.4f}</div></div>
        <div class="stat-card"><div class="stat-label">Lowest RS</div><div class="stat-value" style="color:var(--red)">{min_score:.4f}</div></div>
    </div>
    
    <div style="overflow-x:auto;border-radius:10px;border:1px solid var(--border)">
    <table>
        <thead>
            <tr>
                <th style="width:60px">Rank</th>
                <th>ETF Name</th>
                <th style="width:80px">Symbol</th>
                <th style="width:100px;text-align:right">Price (Rs)</th>
                <th style="width:120px;text-align:right">45d Return</th>
                <th style="width:120px;text-align:right">RS Score</th>
                <th style="width:80px;text-align:center">Signal</th>
            </tr>
        </thead>
        <tbody>"""
    
    for row in table_rows:
        rank_class = "rank-1" if row['rank'] == 1 else "rank-2" if row['rank'] == 2 else "rank-3" if row['rank'] == 3 else "rank-other"
        top3_class = "top3" if row['top3'] else ""
        ret_color = "green" if row['return'] > 0 else "red"
        signal_text = "LONG" if row['top3'] else "OUT"
        signal_color = "green" if row['top3'] else "muted"
        price_str = f"{row['price']:.2f}" if row['price'] is not None else "—"
        
        html += f"""
            <tr class="{top3_class}">
                <td><span class="rank-badge {rank_class}">{row['rank']}</span></td>
                <td><b>{row['name']}</b></td>
                <td style="color:var(--muted)">{row['short']}</td>
                <td class="text-right">{price_str}</td>
                <td class="text-right {ret_color}">{row['return']:+.2f}%</td>
                <td class="text-right accent">{row['score']:.4f}</td>
                <td class="text-center"><span class="{signal_color}" style="font-weight:700">{signal_text}</span></td>
            </tr>"""
    
    html += """
        </tbody>
    </table>
    </div>
    <div style="margin-top:12px;font-size:.7rem;color:var(--muted);display:flex;gap:20px;flex-wrap:wrap">
        <span>🟢 TOP 3 = Buy signal</span>
        <span>🔴 OUT = Not in portfolio</span>
        <span>📊 RS Score = Average excess return vs peers (45-day lookback)</span>
    </div>
</div>

<script>
function downloadCSV() {
    let csv = "Rank,ETF Name,Symbol,Price (Rs),45d Return (%),RS Score,Signal\\n";
    const rows = document.querySelectorAll("tbody tr");
    rows.forEach(row => {
        const cells = row.querySelectorAll("td");
        const rank = cells[0].textContent.trim();
        const name = cells[1].textContent.trim();
        const symbol = cells[2].textContent.trim();
        const price = cells[3].textContent.trim();
        const ret = cells[4].textContent.trim();
        const score = cells[5].textContent.trim();
        const signal = cells[6].textContent.trim();
        csv += `${rank},${name},${symbol},${price},${ret},${score},${signal}\\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rs_45d_table.csv';
    a.click();
    URL.revokeObjectURL(url);
}
</script>
</body>
</html>"""
    
    return html

# ── BUILD HTML DASHBOARD ────────────────────────────────────────────────────
def build_dashboard(prices, bt_result, ft1_result, ft2_result, bt_stats, ft1_stats, ft2_stats, signal):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [script_dir, os.getcwd(), os.path.expanduser("~")]
    chartjs = adapter = None
    for base in candidates:
        cj = os.path.join(base, "node_modules", "chart.js", "dist", "chart.umd.js")
        ad = os.path.join(base, "node_modules", "chartjs-adapter-date-fns", "dist", "chartjs-adapter-date-fns.bundle.js")
        if os.path.exists(cj) and os.path.exists(ad):
            with open(cj) as f: chartjs = f.read()
            with open(ad) as f: adapter = f.read()
            break

    def p(v, d=1): return f"{'+' if v>0 else ''}{v:.{d}f}%"
    def inr(v): return f"Rs {v/1e5:.2f}L"
    def cls(v): return "green" if v >= 0 else "red"

    em = {e[0]: e for e in ETFS}
    upd = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Stats cards ──────────────────────────────────────────────────────
    def stats_card(stats, label, color, period):
        if not stats:
            return f'<div class="cc" style="border-left:3px solid {color}"><div class="st">{label}</div><p style="color:var(--muted)">No data</p></div>'
        return f"""
<div class="cc" style="border-left:3px solid {color}">
  <div class="st">{label}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px">
    <div><div class="lb">Final Value</div><div class="vl accent">{inr(stats["final"])}</div></div>
    <div><div class="lb">Total Return</div><div class="vl {cls(stats['ret'])}">{p(stats['ret'])}</div></div>
    <div><div class="lb">CAGR</div><div class="vl {cls(stats['cagr'])}">{p(stats['cagr'])}</div></div>
    <div><div class="lb">Sharpe</div><div class="vl {cls(stats['sharpe']-1)}">{stats['sharpe']:.2f}</div></div>
    <div><div class="lb">Max DD</div><div class="vl red">{p(stats['mdd'])}</div></div>
    <div><div class="lb">Rebalances</div><div class="vl">{stats.get('rebalances', 0)}</div></div>
    <div><div class="lb">Total Trades</div><div class="vl">{stats['trades']}</div></div>
    <div><div class="lb">Nifty CAGR</div><div class="vl">{p(stats['nifty_cagr'])}</div></div>
    <div><div class="lb">Alpha</div><div class="vl {cls(stats['alpha'])}">{p(stats['alpha'])}</div></div>
    <div><div class="lb">Trading Weeks</div><div class="vl">{stats['weeks']}</div></div>
  </div>
</div>"""

    bt_stats_html = stats_card(bt_stats, f"BACKTEST 2019–2024", "#4f8ef7", BT_END)
    ft1_stats_html = stats_card(ft1_stats, f"FORWARD TEST 2025", "#22c55e", FT1_END)
    ft2_stats_html = stats_card(ft2_stats, f"FORWARD TEST 2026–today", "#f59e0b", FT2_END)

    # ── Current Signal Card ──────────────────────────────────────────────
    if signal:
        sig_html = f"""
<div class="cc signal-card" style="border-color:#4f8ef7;background:linear-gradient(135deg,#1a1d27 0%,#1e2338 100%)">
  <div class="st" style="font-size:1.2rem">📈 CURRENT POSITIONS — {signal['signal_date']}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="mini-stat"><div class="mini-label">Portfolio Value</div>
      <div class="mini-val" style="color:#4f8ef7">{inr(signal['portfolio_val'])}</div></div>
    <div class="mini-stat"><div class="mini-label">Current Holdings</div>
      <div class="mini-val" style="font-size:1.2rem">{len(signal['holdings'])} ETFs</div></div>
    <div class="mini-stat"><div class="mini-label">Next Signal Date</div>
      <div class="mini-val" style="font-size:1rem">{signal['exec_date']}</div></div>
  </div>
  <div style="margin-bottom:16px">
    <div style="font-size:.75rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">CURRENT LONG POSITIONS</div>
    <div style="display:flex;gap:12px;flex-wrap:wrap">"""
        for sym in signal["top3"]:
            name = em.get(sym, (sym, sym, sym))[1]
            short = em.get(sym, (sym, sym, sym))[2]
            sc = signal["scores"].get(sym, 0) * 100
            ret = signal["rets"].get(sym, 0)
            px = signal["current_prices"].get(sym, "N/A")
            shares = signal["holdings"].get(sym, 0)
            rc = "green" if ret > 0 else "red"
            sig_html += f"""
      <div class="etf-pill">
        <div class="etf-pill-name">{name}</div>
        <div class="etf-pill-short">{short}</div>
        <div class="etf-pill-price">Rs {px}</div>
        <div class="etf-pill-shares">{shares:.0f} shares</div>
        <div class="etf-pill-ret {rc}">{ret:+.2f}% (45d)</div>
        <div class="etf-pill-score">RS: {sc:+.3f}</div>
      </div>"""
        sig_html += f"""
    </div>
  </div>
  <div style="font-size:.78rem;color:var(--muted)">
    Signal from FT2: {signal['signal_date']} · Execute: {signal['exec_date']}
  </div>
</div>"""
    else:
        sig_html = '<div class="cc"><p style="color:var(--muted)">No current signal available.</p></div>'

    # ── Trade Log ────────────────────────────────────────────────────────
    def build_trade_table(trades, label):
        if not trades:
            return f'<p style="color:var(--muted)">No trades in {label}</p>'
        html = '<div style="max-height:400px;overflow-y:auto"><table><thead><tr><th>Date</th><th>Action</th><th>ETF</th><th>Shares</th><th>Price</th><th>Value</th></tr></thead><tbody>'
        for t in trades[-100:]:  # Last 100 trades for performance
            sym = t['symbol']
            name = em.get(sym, (sym, sym, sym))[1]
            short = em.get(sym, (sym, sym, sym))[2]
            action_color = "green" if t['action'] == 'BUY' else "red"
            html += f'<tr><td>{t["date"]}</td><td style="color:var(--{action_color})">{t["action"]}</td><td>{name} ({short})</td><td>{t["shares"]:.0f}</td><td>Rs {t["price"]:.2f}</td><td>Rs {t["value"]:.2f}</td></tr>'
        html += '</tbody></table></div>'
        return html

    bt_trades = build_trade_table(bt_result["trade_log"], "Backtest")
    ft1_trades = build_trade_table(ft1_result["trade_log"], "FT1")
    ft2_trades = build_trade_table(ft2_result["trade_log"], "FT2")

    # ── Chart data ──────────────────────────────────────────────────────
    bt_eq = bt_result["eq"]
    bt_nf = bt_result["nf"]
    bt_dd = bt_result["dd"]
    ft1_eq = ft1_result["eq"] if ft1_result else []
    ft1_nf = ft1_result["nf"] if ft1_result else []
    ft1_dd = ft1_result["dd"] if ft1_result else []
    ft2_eq = ft2_result["eq"] if ft2_result else []
    ft2_nf = ft2_result["nf"] if ft2_result else []
    ft2_dd = ft2_result["dd"] if ft2_result else []

    if chartjs and adapter:
        chartjs_tag = f"<script>{chartjs}</script>"
        adapter_tag = f"<script>{adapter}</script>"
    else:
        chartjs_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
        adapter_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>'

    gc = "#2e3250"

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Fixed 45-Day RS Backtester</title>
<style>
:root{{--bg:#0f1117;--card:#1a1d27;--card2:#22263a;--border:#2e3250;--accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#8892b0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif}}
header{{background:var(--card);border-bottom:1px solid var(--border);padding:16px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}}
header h1{{font-size:1.2rem;font-weight:700;color:var(--accent)}}
.meta{{color:var(--muted);font-size:.82rem}}
.container{{max-width:1440px;margin:0 auto;padding:22px}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:13px;margin-bottom:0}}
.sc{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:15px 17px}}
.sc .lb{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}}
.sc .vl{{font-size:1.35rem;font-weight:700}}
.green{{color:var(--green)}}.red{{color:var(--red)}}.accent{{color:var(--accent)}}
.cc{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:20px}}
.signal-card{{border-color:#4f8ef7;background:linear-gradient(135deg,#1a1d27 0%,#1e2338 100%)}}
.cc h3{{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}}
.chartbox{{position:relative;width:100%}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
@media(max-width:780px){{.g2{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:var(--card2);color:var(--muted);font-weight:600;font-size:.67rem;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}}
td{{padding:7px 10px;border-bottom:1px solid var(--border)}}
tr:last-child td{{border-bottom:none}}
.st{{font-size:.93rem;font-weight:700;margin-bottom:13px;display:flex;align-items:center;gap:8px}}
.st::before{{content:'';display:block;width:4px;height:17px;background:var(--accent);border-radius:2px}}
.mw{{overflow-x:auto}}
.period-label{{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.period-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
.mini-stat{{background:var(--card2);border-radius:8px;padding:12px 16px;text-align:center}}
.mini-label{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
.mini-val{{font-size:1.5rem;font-weight:700}}
.etf-pill{{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;min-width:160px;flex:1}}
.etf-pill-name{{font-weight:700;font-size:.95rem;margin-bottom:2px}}
.etf-pill-short{{color:var(--muted);font-size:.72rem;margin-bottom:4px}}
.etf-pill-price{{font-size:1.1rem;font-weight:700;color:var(--accent)}}
.etf-pill-shares{{font-size:.82rem;color:var(--muted);margin:2px 0}}
.etf-pill-ret{{font-size:.82rem;font-weight:600;margin:2px 0}}
.etf-pill-score{{font-size:.75rem;color:var(--muted)}}
.divider{{border:none;border-top:2px dashed var(--border);margin:24px 0}}
.upd{{text-align:right;color:var(--muted);font-size:.76rem;padding:6px 0}}
.tabs{{display:flex;gap:4px;background:var(--card2);padding:4px;border-radius:8px;width:fit-content;margin-bottom:14px}}
.tab{{padding:5px 15px;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:500;color:var(--muted);border:none;background:none}}
.tab.active{{background:var(--accent);color:#fff}}
.trade-section{{margin-bottom:16px}}
.trade-section h4{{color:var(--muted);font-size:.78rem;margin-bottom:8px}}
</style>
</head><body>
<header>
  <div>
    <h1>📊 Fixed 45-Day Relative Strength Backtester</h1>
    <div class="meta">{len(prices.columns)} ETFs · Fixed 45-day lookback · Top-3 Long · Weekly rebalance</div>
  </div>
  <div class="meta">Updated: {upd}</div>
</header>
<div class="container">

{sig_html}

<hr class="divider">
<div class="period-label">▶ BACKTEST PERIOD — 2019 to 2024 (In-sample)</div>
{bt_stats_html}

<div class="g2">
  <div class="cc">
    <h3>Equity Curve — Backtest 2019–2024</h3>
    <div class="chartbox" style="height:280px"><canvas id="ec-bt"></canvas></div>
  </div>
  <div class="cc">
    <h3>Drawdown — Backtest 2019–2024</h3>
    <div class="chartbox" style="height:280px"><canvas id="dc-bt"></canvas></div>
  </div>
</div>

<div class="cc">
  <h3>Trade Log — Backtest 2019–2024</h3>
  {bt_trades}
</div>

<hr class="divider">
<div class="period-label">▶ FORWARD TEST 1 — 2025 (Out-of-sample)</div>
{ft1_stats_html}

<div class="g2">
  <div class="cc">
    <h3>Equity Curve — FT1 2025</h3>
    <div class="chartbox" style="height:280px"><canvas id="ec-ft1"></canvas></div>
  </div>
  <div class="cc">
    <h3>Drawdown — FT1 2025</h3>
    <div class="chartbox" style="height:280px"><canvas id="dc-ft1"></canvas></div>
  </div>
</div>

<div class="cc">
  <h3>Trade Log — FT1 2025</h3>
  {ft1_trades}
</div>

<hr class="divider">
<div class="period-label">▶ FORWARD TEST 2 — 2026 to today (Out-of-sample)</div>
{ft2_stats_html}

<div class="g2">
  <div class="cc">
    <h3>Equity Curve — FT2 2026–today</h3>
    <div class="chartbox" style="height:280px"><canvas id="ec-ft2"></canvas></div>
  </div>
  <div class="cc">
    <h3>Drawdown — FT2 2026–today</h3>
    <div class="chartbox" style="height:280px"><canvas id="dc-ft2"></canvas></div>
  </div>
</div>

<div class="cc">
  <h3>Trade Log — FT2 2026–today</h3>
  {ft2_trades}
</div>

<hr class="divider">
<div class="upd">Fixed 45-day lookback · Backtest: 2019-2024 · Forward: 2025, 2026-today</div>
</div>

{chartjs_tag}
{adapter_tag}
<script>
var gc='{gc}';
var btEq={json.dumps(bt_eq)};
var btNf={json.dumps(bt_nf)};
var btDd={json.dumps(bt_dd)};
var ft1Eq={json.dumps(ft1_eq)};
var ft1Nf={json.dumps(ft1_nf)};
var ft1Dd={json.dumps(ft1_dd)};
var ft2Eq={json.dumps(ft2_eq)};
var ft2Nf={json.dumps(ft2_nf)};
var ft2Dd={json.dumps(ft2_dd)};

function mkLine(id, datasets, yFmt){{
  new Chart(document.getElementById(id),{{
    type:'line', data:{{datasets:datasets}},
    options:{{responsive:true,maintainAspectRatio:false,parsing:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{labels:{{color:'#8892b0',boxWidth:12}}}},
        tooltip:{{backgroundColor:'#1a1d27',titleColor:'#e2e8f0',bodyColor:'#8892b0',
          callbacks:{{label:function(c){{return c.dataset.label+': '+yFmt(c.raw.y);}}}}}}
      }},
      scales:{{
        x:{{type:'time',time:{{unit:'month'}},ticks:{{color:'#8892b0',maxTicksLimit:18}},grid:{{color:gc}}}},
        y:{{ticks:{{color:'#8892b0',callback:function(v){{return yFmt(v);}}}},grid:{{color:gc}}}}
      }}
    }}
  }});
}}

var rupee = function(v){{return 'Rs '+(v/1000).toFixed(0)+'K';}};
var pct   = function(v){{return v.toFixed(1)+'%';}};

mkLine('ec-bt',[
  {{label:'Fixed 45d RS', data:btEq, parsing:false, borderColor:'#4f8ef7',
   backgroundColor:'rgba(79,142,247,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
  {{label:'NiftyBees', data:btNf, parsing:false, borderColor:'#f59e0b',
   backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, borderDash:[5,4], tension:0.3}}
], rupee);

mkLine('dc-bt',[
  {{label:'Drawdown', data:btDd, parsing:false, borderColor:'#ef4444',
   backgroundColor:'rgba(239,68,68,0.12)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true}}
], pct);

mkLine('ec-ft1',[
  {{label:'Fixed 45d RS', data:ft1Eq, parsing:false, borderColor:'#22c55e',
   backgroundColor:'rgba(34,197,94,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
  {{label:'NiftyBees', data:ft1Nf, parsing:false, borderColor:'#f59e0b',
   backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, borderDash:[5,4], tension:0.3}}
], rupee);

mkLine('dc-ft1',[
  {{label:'Drawdown', data:ft1Dd, parsing:false, borderColor:'#ef4444',
   backgroundColor:'rgba(239,68,68,0.12)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true}}
], pct);

mkLine('ec-ft2',[
  {{label:'Fixed 45d RS', data:ft2Eq, parsing:false, borderColor:'#f59e0b',
   backgroundColor:'rgba(245,158,11,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
  {{label:'NiftyBees', data:ft2Nf, parsing:false, borderColor:'#f59e0b',
   backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, borderDash:[5,4], tension:0.3}}
], rupee);

mkLine('dc-ft2',[
  {{label:'Drawdown', data:ft2Dd, parsing:false, borderColor:'#ef4444',
   backgroundColor:'rgba(239,68,68,0.12)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true}}
], pct);
</script>
</body></html>"""

# ── MOCK DATA ─────────────────────────────────────────────────────────────────
def make_mock_data():
    print("MOCK MODE: generating synthetic data")
    trade_dates = pd.date_range(FETCH_START, END, freq="B")
    np.random.seed(42)
    mock_etfs = [e[0] for e in ETFS[:15]]
    prices_dict = {}
    for sym in mock_etfs:
        drift = np.random.uniform(0.00008, 0.00035)
        vol = np.random.uniform(0.010, 0.020)
        start = np.random.uniform(30, 300)
        lr = np.random.normal(drift, vol, len(trade_dates))
        # Add some correlation
        if 'NIFTY' in sym:
            prices_dict[sym] = start * np.exp(np.cumsum(lr))
        else:
            prices_dict[sym] = start * np.exp(np.cumsum(lr + np.random.normal(0, 0.005, len(trade_dates))))
    prices = pd.DataFrame(prices_dict, index=trade_dates)
    print(f"  Mock: {len(prices)} days, {len(prices.columns)} ETFs")
    return prices

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    USE_MOCK = "--mock" in sys.argv

    if USE_MOCK:
        prices = make_mock_data()
    else:
        prices = fetch_all()

    if prices.empty:
        print("ERROR: No data. Try: python fixed45_backtest.py --mock", file=sys.stderr)
        sys.exit(1)

    # ── Run backtest 2019-2024 ──────────────────────────────────────────────
    bt_result = run_backtest(prices,
                             period_start=BT_START,
                             period_end=BT_END,
                             label="Backtest 2019-2024",
                             initial_capital=INITIAL)
    bt_stats = calc_stats(bt_result, initial=INITIAL) if bt_result else None

    # ── Forward test 2025 ────────────────────────────────────────────────────
    ft1_result = run_backtest(prices,
                              period_start=FT1_START,
                              period_end=FT1_END,
                              label="Forward Test 2025",
                              initial_capital=INITIAL)
    ft1_stats = calc_stats(ft1_result, initial=INITIAL) if ft1_result else None

    # ── Forward test 2026-today ──────────────────────────────────────────────
    ft2_result = run_backtest(prices,
                              period_start=FT2_START,
                              period_end=FT2_END,
                              label="Forward Test 2026-today",
                              initial_capital=INITIAL)
    ft2_stats = calc_stats(ft2_result, initial=INITIAL) if ft2_result else None

    # ── Current signal ───────────────────────────────────────────────────────
    signal = get_current_signal(prices, ft2_result or ft1_result)

    # ── Build RS Table ──────────────────────────────────────────────────────
    rs_table_html = build_rs_table_html(prices)
    os.makedirs("docs", exist_ok=True)
    with open(RS_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(rs_table_html)
    print(f"\n✅ 45-day RS table → {RS_TABLE_PATH}")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BACKTEST 2019–2024")
    print(f"{'='*60}")
    if bt_stats:
        for k, v in [("Final", f"Rs {bt_stats['final']:,.0f}"),
                     ("Return", f"{bt_stats['ret']:+.1f}%"),
                     ("CAGR", f"{bt_stats['cagr']:+.1f}%"),
                     ("Sharpe", f"{bt_stats['sharpe']:.2f}"),
                     ("Max DD", f"{bt_stats['mdd']:.1f}%"),
                     ("Rebalances", f"{bt_stats['rebalances']}"),
                     ("Trades", f"{bt_stats['trades']}"),
                     ("Alpha", f"{bt_stats['alpha']:+.1f}%")]:
            print(f"  {k:<12}: {v}")

    print(f"\n{'='*60}")
    print("FORWARD TEST 2025")
    print(f"{'='*60}")
    if ft1_stats:
        for k, v in [("Final", f"Rs {ft1_stats['final']:,.0f}"),
                     ("Return", f"{ft1_stats['ret']:+.1f}%"),
                     ("CAGR", f"{ft1_stats['cagr']:+.1f}%"),
                     ("Sharpe", f"{ft1_stats['sharpe']:.2f}"),
                     ("Max DD", f"{ft1_stats['mdd']:.1f}%"),
                     ("Rebalances", f"{ft1_stats['rebalances']}"),
                     ("Trades", f"{ft1_stats['trades']}"),
                     ("Alpha", f"{ft1_stats['alpha']:+.1f}%")]:
            print(f"  {k:<12}: {v}")

    print(f"\n{'='*60}")
    print("FORWARD TEST 2026–today")
    print(f"{'='*60}")
    if ft2_stats:
        for k, v in [("Final", f"Rs {ft2_stats['final']:,.0f}"),
                     ("Return", f"{ft2_stats['ret']:+.1f}%"),
                     ("CAGR", f"{ft2_stats['cagr']:+.1f}%"),
                     ("Sharpe", f"{ft2_stats['sharpe']:.2f}"),
                     ("Max DD", f"{ft2_stats['mdd']:.1f}%"),
                     ("Rebalances", f"{ft2_stats['rebalances']}"),
                     ("Trades", f"{ft2_stats['trades']}"),
                     ("Alpha", f"{ft2_stats['alpha']:+.1f}%")]:
            print(f"  {k:<12}: {v}")

    if signal:
        print(f"\nCURRENT POSITIONS ({signal['signal_date']})")
        print(f"  Portfolio Value: Rs {signal['portfolio_val']:,.0f}")
        print(f"  Holdings: {', '.join(signal['top3_names'])}")
        print(f"  Next Signal Date: {signal['exec_date']}")

    # ── Build Dashboard ──────────────────────────────────────────────────────
    html = build_dashboard(prices, bt_result, ft1_result, ft2_result,
                           bt_stats, ft1_stats, ft2_stats, signal)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Dashboard → {OUT_PATH}")
    print(f"✅ RS Table → {RS_TABLE_PATH}")
