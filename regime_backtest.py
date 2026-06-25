import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ── CONFIGURATIONS ────────────────────────────────────────────────────────────
FETCH_START = "2016-01-01"
END = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 1000000.0  # Rs 10 Lakhs
TOP_N = 3
VIX_SYM = "^INDIAVIX"
OUT_PATH = "docs/index.html"

# Lookback parameters
LOOKBACKS = [15, 30, 45, 55, 65]

# Setup ETF Master Data (Ticker, Full Name, Display Name)
ETFS = [
    ("NIFTYBEES.NS", "Nifty 50 ETF", "NiftyBees"),
    ("JUNIORBEES.NS", "Nifty Next 50 ETF", "JuniorBees"),
    ("BANKBEES.NS", "Nifty Bank ETF", "BankBees"),
    ("GOLDBEES.NS", "Gold ETF", "GoldBees"),
    ("ITBEES.NS", "Nifty IT ETF", "ITBees"),
    ("PHARMABEES.NS", "Nifty Pharma ETF", "PharmaBees"),
    ("CPSEETF.NS", "CPSE ETF", "CPSE"),
    ("MON100.NS", "Nasdaq 100 ETF", "Motilal Nasdaq"),
    ("MAFANG.NS", "FANG+ ETF", "Mirae FANG+"),
    ("INFRA-I.NS", "Nifty Infrastructure ETF", "ICICI Infra"),
    ("MID150BEES.NS", "Nifty Midcap 150 ETF", "MidcapBees"),
    ("CONSUMBEES.NS", "Nifty Consumption ETF", "ConsumerBees")
]

# VIX thresholds mapping to lookbacks: (low, high, Label, lookback)
VIX_REGIMES = [
    (0,  13, "Extremely Complacent", 15),
    (13, 17, "Normal / Balanced",   30),
    (17, 22, "Elevated Anxiety",    45),
    (22, 28, "High Stress / Panic", 55),
    (28, 999, "Systemic Crisis",     65)
]

def get_regime_info(vix_val):
    if pd.isna(vix_val):
        return "Normal / Balanced", 30
    for lo, hi, label, lb in VIX_REGIMES:
        if lo <= vix_val < hi:
            return label, lb
    return "Normal / Balanced", 30

# ── CORE SIMULATION ENGINE ────────────────────────────────────────────────────
def run_strategy(prices, vix, fixed_lookback=None):
    """
    Simulates a momentum rotation strategy across the ETF universe.
    - If fixed_lookback is provided, it uses that exact window throughout.
    - If None, it dynamically updates lookbacks on Friday based on India VIX.
    """
    # 1. Resample to standard strategy execution weeks
    weekly_signals = prices.resample('W-FRI').last()
    
    # Pre-calculate asset lookback changes
    returns_cache = {lb: prices.pct_change(lb) for lb in LOOKBACKS}
    
    # Working data structures
    date_index = prices.index
    eq_curve = pd.Series(index=date_index, dtype=float)
    eq_curve.iloc[0] = INITIAL_CAPITAL
    
    current_capital = INITIAL_CAPITAL
    active_shares = {}     # {ticker: shares_held}
    active_tickers = []    # list of active tokens
    
    log_book = []
    avail_universe = [e[0] for e in ETFS]
    
    # Performance helper values
    last_scores_map = {}
    last_rets_map = {}

    for idx, (signal_date, signal_row) in enumerate(weekly_signals.iterrows()):
        # Filter down to operational tickers that have traded historical data
        avail = [col for col in avail_universe if col in prices.columns and not pd.isna(signal_row[col])]
        if not avail:
            continue
            
        # Determine lookback allocation rule
        if fixed_lookback:
            current_lb = fixed_lookback
            regime_label = f"Fixed {fixed_lookback}d"
            cur_vix = vix.loc[:signal_date].iloc[-1] if not vix.empty and signal_date in vix.index else None
        else:
            cur_vix = vix.loc[:signal_date].iloc[-1] if not vix.empty else 16.0
            regime_label, current_lb = get_regime_info(cur_vix)
            
        # Extract respective momentum scores
        lb_returns = returns_cache[current_lb]
        scores = lb_returns.loc[signal_date, avail].dropna()
        
        # Determine top execution slice
        top_targets = scores.nlargest(TOP_N).index.tolist()
        
        # Capture scores and returns to display in current UI rankings panel
        if idx == len(weekly_signals) - 1:
            last_scores_map = scores.to_dict()
            last_rets_map = prices.pct_change(5).loc[signal_date, avail].dropna().to_dict() # 1-week return placeholder
            
        # Calculate execution date window (Next business day closing price / Monday close mapping)
        try:
            pos = date_index.get_loc(signal_date)
            exec_date = date_index[pos + 1] if pos + 1 < len(date_index) else signal_date
        except:
            exec_date = signal_date
            
        # Calculate intermediate valuation updates up to the next rotation window
        # For simplicity, performance is evaluated at execution boundaries:
        if active_shares:
            # Re-value current state matching incoming pricing arrays
            current_capital = sum(active_shares[sym] * prices.loc[exec_date, sym] for sym in active_tickers)
        
        # Calculate execution allocations across our portfolio targets
        entering = [s for s in top_targets if s not in active_tickers]
        exiting  = [s for s in active_tickers if s not in top_targets]
        changed  = len(entering) > 0
        
        # Execute orders at target Close Price parameters
        active_shares = {}
        active_tickers = top_targets.copy()
        allocation_block = current_capital / max(len(top_targets), 1)
        
        for sym in top_targets:
            px = prices.loc[exec_date, sym]
            active_shares[sym] = allocation_block / px if px > 0 else 0
            
        log_book.append({
            "date": signal_date.strftime("%Y-%m-%d"),
            "lb": current_lb,
            "vix": round(float(cur_vix), 2) if cur_vix else None,
            "regime": regime_label,
            "top3": top_targets,
            "entering": entering,
            "exiting": exiting,
            "changed": changed,
            "capital": round(current_capital, 2)
        })
        
        # Sync linear progression blocks across equity tracking arrays
        eq_curve.loc[signal_date:exec_date] = current_capital

    eq_curve = eq_curve.ffill().fillna(INITIAL_CAPITAL)
    return eq_curve, log_book, last_scores_map, last_rets_map

# ── CALIBRATION ENGINE ────────────────────────────────────────────────────────
def calibrate_yearly_with_benchmarks(prices, vix):
    """
    Calculates detailed calendar year performance returns and CAGRs 
    for each fixed lookback period parameter.
    """
    years = sorted(prices.index.year.unique())
    yearly_calib = {}
    
    # Pre-simulate all baseline tracking runs across structural timelines
    static_runs = {lb: run_strategy(prices, vix, fixed_lookback=lb)[0] for lb in LOOKBACKS}
    
    for yr in years:
        vix_slice = vix[vix.index.year == yr]
        avg_vx = round(float(vix_slice.mean()), 1) if not vix_slice.empty else None
        max_vx = round(float(vix_slice.max()), 1) if not vix_slice.empty else None
        
        # Track baseline benchmark mapping label
        repr_vix = avg_vx if avg_vx else 16.0
        label, _ = get_regime_info(repr_vix)
        
        yearly_calib[yr] = {
            "avg_vix": avg_vx,
            "max_vix": max_vx,
            "regime": label,
            "lb_stats": {}
        }
        
        best_sharpe = -np.inf
        best_lb = 30
        
        for lb in LOOKBACKS:
            eq = static_runs[lb]
            eq_yr = eq[eq.index.year == yr]
            if len(eq_yr) < 2: continue
            
            # Absolute Yearly Return computation profile
            y_ret = ((eq_yr.iloc[-1] / eq_yr.iloc[0]) - 1) * 100
            
            # Risk/Reward metric modeling (Sharpe Profiling parameters)
            daily_pct = eq_yr.pct_change().dropna()
            std_dev = daily_pct.std()
            ann_vol = std_dev * np.sqrt(252) if std_dev > 0 else 0
            ann_ret = daily_pct.mean() * 252
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
            
            # Standard structural conversion matrix (Yearly Return = CAGR across 1 Year window bounds)
            yearly_calib[yr]["lb_stats"][lb] = {
                "sharpe": round(sharpe, 2),
                "return": round(y_ret, 2),
                "cagr": round(y_ret, 2)
            }
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lb = lb
                
        yearly_calib[yr]["best_lb"] = best_lb
        
    return yearly_calib

# ── PERFORMANCE ANALYTICS MATRICES ────────────────────────────────────────────
def calc_stats(eq, benchmark_prices, log_book):
    if eq.empty: return {}
    
    # Match benchmark indexes cleanly
    bench = benchmark_prices.reindex(eq.index).ffill().bfill()
    
    total_ret = ((eq.iloc[-1] / eq.iloc[0]) - 1) * 100
    bench_ret = ((bench.iloc[-1] / bench.iloc[0]) - 1) * 100
    
    days = (eq.index[-1] - eq.index[0]).days
    cagr = (((eq.iloc[-1] / eq.iloc[0]) ** (365.0 / max(days, 1))) - 1) * 100
    b_cagr = (((bench.iloc[-1] / bench.iloc[0]) ** (365.0 / max(days, 1))) - 1) * 100
    
    # Drawdown profile loop
    cum_max = eq.cummax()
    dd_series = ((eq - cum_max) / cum_max) * 100
    mdd = dd_series.min()
    
    # Shape ratio evaluation profile
    daily_pct = eq.pct_change().dropna()
    sharpe = (daily_pct.mean() / daily_pct.std() * np.sqrt(252)) if daily_pct.std() > 0 else 0
    
    trades_count = sum(1 for lg in log_book if lg["changed"])
    
    return {
        "final": eq.iloc[-1],
        "ret": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades": trades_count,
        "nifty_cagr": b_cagr,
        "alpha": cagr - b_cagr,
        "dd_series": dd_series
    }

# ── DATA MOCK IMPLEMENTATION ENGINE ───────────────────────────────────────────
def make_mock_data():
    print("MOCK MODE: Generating full tracking metrics vectors safely...")
    trade_dates = pd.date_range(FETCH_START, END, freq="B")
    np.random.seed(42)
    
    prices_dict = {}
    for sym in [e[0] for e in ETFS]:
        drift = np.random.uniform(0.0001, 0.0003)
        vol = np.random.uniform(0.012, 0.018)
        start_px = np.random.uniform(50, 250)
        log_rets = np.random.normal(drift, vol, len(trade_dates))
        prices_dict[sym] = start_px * np.exp(np.cumsum(log_rets))
        
    prices_df = pd.DataFrame(prices_dict, index=trade_dates)
    vix_vals = np.random.normal(16, 2.5, len(trade_dates))
    
    # Model cyclical black swan events
    for yr, bump in [(2016, 8), (2020, 28), (2022, 12)]:
        mask = trade_dates.year == yr
        vix_vals[mask] += np.random.uniform(2, bump, mask.sum())
        
    vix_series = pd.Series(np.clip(vix_vals, 9.5, 62.0), index=trade_dates, name=VIX_SYM)
    return prices_df, vix_series

# ── HTML BUILD ENGINE ─────────────────────────────────────────────────────────
def build_html(prices, vix, yearly, adaptive_eq, stats, log_book, last_scores, last_rets):
    avail = [e[0] for e in ETFS if e[0] in prices.columns]
    em = {e[0]: e for e in ETFS}
    upd = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    def p(v, d=1): return f"{'+' if v>0 else ''}{v:.{d}f}%"
    def inr(v): return f"Rs {v/1e5:.2f}L"
    def cls(v): return "green" if v>=0 else "red"

    # Assemble Last Week's Edge Execution Context (Signal Block Card)
    latest_log = log_book[-1] if log_book else None
    if latest_log:
        vix_val = latest_log["vix"] or 15.0
        vix_color = "#22c55e" if vix_val < 14 else ("#f59e0b" if vix_val < 22 else "#ef4444")
        
        sig_html = f"""
<div class="cc signal-card">
  <div class="lb">ACTIVE ROTATION ALLOCATION SIGNAL</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(120px, 1fr));gap:16px;margin-bottom:20px">
    <div class="mini-stat">
      <div class="mini-label">India VIX</div>
      <div class="mini-val" style="color:{vix_color}">{vix_val}</div>
    </div>
    <div class="mini-stat">
      <div class="mini-label">Regime Zone</div>
      <div class="mini-val" style="font-size:1rem; white-space:nowrap;">{latest_log['regime']}</div>
    </div>
    <div class="mini-stat">
      <div class="mini-label">Active Lookback</div>
      <div class="mini-val accent">{latest_log['lb']}d</div>
    </div>
  </div>
  <div>
    <div style="font-size:.75rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">STRATEGY ALLOCATION TARGETS (TOP 3)</div>
    <div style="display:flex;gap:12px;flex-wrap:wrap">"""
        for sym in latest_log["top3"]:
            name = em[sym][1] if sym in em else sym
            lbl = em[sym][2] if sym in em else sym
            px = round(float(prices[sym].iloc[-1]), 2)
            sc = last_scores.get(sym, 0) * 100
            ret = last_rets.get(sym, 0) * 100
            
            sig_html += f"""
      <div class="etf-pill">
        <div class="etf-pill-name">{name}</div>
        <div class="etf-pill-short">{lbl}</div>
        <div class="etf-pill-price">Rs {px}</div>
        <div class="etf-pill-ret {cls(ret)}">{p(ret, 2)} (Wk)</div>
        <div class="etf-pill-score">RS Score: {sc:+.3f}</div>
      </div>"""
        sig_html += f"""
    </div>
  </div>
</div>"""
    else:
        sig_html = ""

    # Assemble VIX Regime Selection Rule Set Mapping Table
    regime_html = """<table><thead><tr><th>VIX Thresholds</th><th>Regime Label</th><th>Assigned Lookback</th><th>Rationale</th></tr></thead><tbody>"""
    for lo, hi, label, lb in VIX_REGIMES:
        hi_str = str(hi) if hi < 999 else "+"
        is_cur = latest_log and lo <= (latest_log["vix"] or 15) < hi
        bg = "background:rgba(79,142,247,0.08);" if is_cur else ""
        tag = ' <span class="bd bg">ACTIVE</span>' if is_cur else ""
        regime_html += f"""<tr style="{bg}"><td><b>{lo} – {hi_str}</b></td><td>{label}{tag}</td><td style="color:var(--accent);font-weight:700">{lb}d</td><td style="color:var(--muted)">Rotation alpha matching asset pricing noise bounds.</td></tr>"""
    regime_html += "</tbody></table>"

    # Assemble Calibration View containing accurate backtest profiles
    yr_html = """<table><thead><tr>
      <th>Year</th><th>Avg VIX</th><th>Max VIX</th><th>Regime</th><th>Best LB</th>
      <th>15d</th><th>30d</th><th>45d</th><th>55d</th><th>65d</th>
    </tr></thead><tbody>"""
    for yr, obj in sorted(yearly.items()):
        best = obj["best_lb"]
        row = f"""<tr><td><b>{yr}</b></td><td>{obj['avg_vix'] or '—'}</td><td>{obj['max_vix'] or '—'}</td><td>{obj['regime']}</td><td style="color:var(--accent);font-weight:700">{best}d</td>"""
        for lb in LOOKBACKS:
            st = obj["lb_stats"].get(lb)
            if st:
                val = st["return"]
                row += f"""<td class="{cls(val)}" style="{'font-weight:700;' if lb==best else ''}">{val:+.1f}%</td>"""
            else:
                row += "<td>—</td>"
        row += "</tr>"
        yr_html += row
    yr_html += "</tbody></table>"

    # Assemble Log Row Streamers
    log_html = ""
    for lg in reversed(log_book):
        p_tags = "".join(f'<span class="tg">{em[s][2] if s in em else s}</span>' for s in lg["top3"])
        chg_block = ""
        if lg["changed"]:
            ex = " ".join(f'<span class="bd br">- {em[s][2] if s in em else s}</span>' for s in lg["exiting"])
            en = " ".join(f'<span class="bd bg">+ {em[s][2] if s in em else s}</span>' for s in lg["entering"])
            chg_block = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">{ex} {en}</div>'
        
        nc_lbl = "" if lg["changed"] else '<span style="color:var(--muted);font-size:.7rem;margin-left:6px">rolled positioning</span>'
        log_html += f"""
        <div class="lr" data-c="{"1" if lg['changed'] else "0"}">
          <div class="ld">{lg['date']}<span style="display:block;font-size:.65rem;color:var(--muted)">{lg['lb']}d · VIX {lg['vix'] or '—'}</span></div>
          <div class="lb2">
            <div style="display:flex;align-items:center;flex-wrap:wrap">{p_tags}{nc_lbl}</div>
            {chg_block}
            <div style="color:var(--muted);font-size:.7rem;margin-top:2px">Portfolio Value: Rs {lg['capital']/1e3:.1f}K · {lg['regime']}</div>
          </div>
        </div>"""

    # Format line chart objects cleanly for browser render pass
    eq_json = json.dumps([{"x": str(d.date()), "y": round(float(v), 2)} for d, v in adaptive_eq.items()])
    nf_json = json.dumps([{"x": str(d.date()), "y": round(float(prices["NIFTYBEES.NS"].loc[d]), 2) * (INITIAL_CAPITAL/prices["NIFTYBEES.NS"].iloc[0])} for d in adaptive_eq.index])
    dd_json = json.dumps([{"x": str(d.date()), "y": round(float(v), 2)} for d, v in stats["dd_series"].items()])
    vx_json = json.dumps([{"x": str(d.date()), "y": round(float(v), 2)} for d, v in vix.reindex(adaptive_eq.index).ffill().items()])

    chartjs_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
    adapter_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>India ETF RS — Regime Adaptive Backtest</title>
<style>
:root{{--bg:#0f1117;--card:#1a1d27;--card2:#22263a;--border:#2e3250;--accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#8892b0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:0;margin:0}}
header{{background:var(--card);border-bottom:1px solid var(--border);padding:16px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}}
header h1{{font-size:1.2rem;font-weight:700;color:var(--accent)}}
.meta{{color:var(--muted);font-size:.82rem}}
.container{{max-width:1440px;margin:0 auto;padding:22px}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:13px;margin-bottom:22px}}
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
.bd{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:600}}
.bg{{background:rgba(34,197,94,.15);color:var(--green)}}
.br{{background:rgba(239,68,68,.15);color:var(--red)}}
.bm{{background:rgba(136,146,176,.1);color:var(--muted)}}
.st{{font-size:.93rem;font-weight:700;margin-bottom:13px;display:flex;align-items:center;gap:8px}}
.st::before{{content:'';display:block;width:4px;height:17px;background:var(--accent);border-radius:2px}}
.mw{{overflow-x:auto}}
.tg{{display:inline-flex;background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.3);color:var(--accent);border-radius:5px;padding:2px 8px;font-size:.77rem;font-weight:600;margin:2px}}
.tabs{{display:flex;gap:4px;background:var(--card2);padding:4px;border-radius:8px;width:fit-content;margin-bottom:14px}}
.tab{{padding:5px 15px;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:500;color:var(--muted);border:none;background:none}}
.tab.active{{background:var(--accent);color:#fff}}
#log{{max-height:360px;overflow-y:auto}}
.lr{{display:flex;gap:11px;padding:7px 0;border-bottom:1px solid var(--border);font-size:.79rem}}
.ld{{color:var(--muted);min-width:88px;flex-shrink:0}}
.lb2{{flex:1}}
.mini-stat{{background:var(--card2);border-radius:8px;padding:12px 16px;text-align:center}}
.mini-label{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
.mini-val{{font-size:1.5rem;font-weight:700}}
.etf-pill{{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;min-width:160px;flex:1}}
.etf-pill-name{{font-weight:700;font-size:.95rem;margin-bottom:2px}}
.etf-pill-short{{color:var(--muted);font-size:.72rem;margin-bottom:8px}}
.etf-pill-price{{font-size:1.1rem;font-weight:700;color:var(--accent);margin-bottom:4px}}
.etf-pill-ret{{font-size:.82rem;font-weight:600;margin-bottom:2px}}
.etf-pill-score{{font-size:.75rem;color:var(--muted)}}
.upd{{text-align:right;color:var(--muted);font-size:.76rem;padding:6px 0}}
</style>
</head><body>
<header>
  <div>
    <h1>India ETF — Regime-Adaptive RS Backtest</h1>
    <div class="meta">{len(avail)} ETFs &middot; VIX Selection Matrix &middot; Top-3 Long &middot; Weekly Rotations</div>
  </div>
  <div class="meta">Updated: {upd}</div>
</header>
<div class="container">

{sig_html}

<div class="sg">
  <div class="sc"><div class="lb">Final Value</div><div class="vl accent">{inr(stats["final"])}</div></div>
  <div class="sc"><div class="lb">Total Return</div><div class="vl {cls(stats["ret"])}">{p(stats["ret"])}</div></div>
  <div class="sc"><div class="lb">CAGR</div><div class="vl {cls(stats["cagr"])}">{p(stats["cagr"])}</div></div>
  <div class="sc"><div class="lb">Sharpe Ratio</div><div class="vl {cls(stats["sharpe"]-1)}">{stats["sharpe"]:.2f}</div></div>
  <div class="sc"><div class="lb">Max Drawdown</div><div class="vl red">{p(stats["mdd"])}</div></div>
  <div class="sc"><div class="lb">Total Trades</div><div class="vl">{stats["trades"]}</div></div>
  <div class="sc"><div class="lb">Nifty CAGR</div><div class="vl">{p(stats["nifty_cagr"])}</div></div>
  <div class="sc"><div class="lb">Alpha vs Nifty</div><div class="vl {cls(stats["alpha"])}">{p(stats["alpha"])}</div></div>
</div>

<div class="cc">
  <h3>Equity Curve — Regime-Adaptive Performance</h3>
  <div class="chartbox" style="height:300px"><canvas id="ec"></canvas></div>
</div>

<div class="cc">
  <h3>India VIX History</h3>
  <div class="chartbox" style="height:160px"><canvas id="vx"></canvas></div>
</div>

<div class="cc">
  <h3>Drawdown Waveform</h3>
  <div class="chartbox" style="height:140px"><canvas id="dc"></canvas></div>
</div>

<div class="cc">
  <div class="st">VIX Regime Map — Lookback Allocation Rules</div>
  {regime_html}
</div>

<div class="cc">
  <div class="st">Per-Year Performance Comparisons — Fixed Lookbacks (Absolute Return %)</div>
  <div class="mw">{yr_html}</div>
</div>

<div class="g2">
  <div class="cc">
    <div class="st">Weekly Strategy Action Log</div>
    <div class="tabs">
      <button class="tab active" onclick="sw('all',this)">All</button>
      <button class="tab" onclick="sw('chg',this)">Changes</button>
    </div>
    <div id="log">{log_html}</div>
  </div>
</div>

</div>

{chartjs_tag}
{adapter_tag}
<script>
var eqD = {eq_json}; var nfD = {nf_json}; var ddD = {dd_json}; var vxD = {vx_json};

new Chart(document.getElementById('ec'), {{
  type:'line', data:{{datasets:[
    {{label:'Adaptive RS Portfolio', data:eqD, parsing:false, borderColor:'#4f8ef7', backgroundColor:'rgba(79,142,247,0.06)', borderWidth:2, pointRadius:0, tension:0.1, fill:true}},
    {{label:'NiftyBees (Indexed)', data:nfD, parsing:false, borderColor:'#f59e0b', backgroundColor:'transparent', borderWidth:1.2, pointRadius:0, borderDash:[4,4], tension:0.1}}
  ]}},
  options:{{responsive:true, maintainAspectRatio:false, interaction:{{mode:'index',intersect:false}},
    scales:{{x:{{type:'time',time:{{unit:'year'}},ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}},y:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}}}}
  }}
}});

new Chart(document.getElementById('vx'), {{
  type:'line', data:{{datasets:[{{label:'India VIX', data:vxD, parsing:false, borderColor:'#a78bfa', backgroundColor:'rgba(167,139,250,0.05)', borderWidth:1.2, pointRadius:0, fill:true}}]}},
  options:{{responsive:true, maintainAspectRatio:false, scales:{{x:{{type:'time',time:{{unit:'year'}},ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}},y:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}}}} }}
}});

new Chart(document.getElementById('dc'), {{
  type:'line', data:{{datasets:[{{label:'Drawdown %', data:ddD, parsing:false, borderColor:'#ef4444', backgroundColor:'rgba(239,68,68,0.08)', borderWidth:1.2, pointRadius:0, fill:true}}]}},
  options:{{responsive:true, maintainAspectRatio:false, scales:{{x:{{type:'time',time:{{unit:'year'}},ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}},y:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}}}} }}
}});

function sw(t,el){{
  document.querySelectorAll('.tab').forEach(function(x){{x.classList.remove('active');}}); el.classList.add('active');
  document.querySelectorAll('.lr').forEach(function(r){{ r.style.display=(t==='all'||r.dataset.c==='1')?'flex':'none'; }});
}}
</script>
</body></html>"""

# ── PIPELINE ENTRY POINT ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate mock vectors directly to run standalone execution safely
    prices, vix = make_mock_data()
    
    # 1. Run per-year calibration benchmark evaluations
    yearly_calib = calibrate_yearly_with_benchmarks(prices, vix)
    
    # 2. Execute active tactical model tracking run
    adaptive_eq, log_book, last_scores, last_rets = run_strategy(prices, vix, fixed_lookback=None)
    
    # 3. Calculate processing analytics structures
    stats = calc_stats(adaptive_eq, prices["NIFTYBEES.NS"], log_book)
    
    # 4. Render output page
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    html_content = build_html(prices, vix, yearly_calib, adaptive_eq, stats, log_book, last_scores, last_rets)
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\nPipeline successfully completed! Output view saved here -> {OUT_PATH}")
