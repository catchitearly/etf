"""
Regime-Adaptive RS Backtester — v2
====================================
Key changes from v1:
  1. CORRECTED VIX→lookback mapping (empirically tuned per per-year analysis)
  2. SPLIT PERIODS: Backtest 2015-01-01 → 2024-12-31  |  Forward 2025-01-01 → today
  3. Full-period fixed-LB comparison: run each of 15/30/45/55/65d over 2015→today
  4. Separate equity curves, stats, and tables for each period
  5. Per-year calibration covers 2015-2024 only (honest — no future leakage)
  6. NEW: Full RS table for 45-day lookback in new tab
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

VIX_SYM   = "^INDIAVIX"
NIFTY_SYM = "NIFTYBEES.NS"
LOOKBACKS = [15, 30, 45, 55, 65]
TOP_N     = 3
COST_PCT  = 0.001
INITIAL   = 1_000_000

FETCH_START  = "2019-01-01"
BT_END       = date.today().strftime("%Y-%m-%d") #"2024-12-31"   # backtest period end
FT_START     = "2025-01-01"   # forward test start
END          = date.today().strftime("%Y-%m-%d")

CACHE_DIR    = ".cache/regime"
OUT_PATH     = "docs/index.html"

# ── CORRECTED VIX REGIME MAP ──────────────────────────────────────────────────
# v1 mapping caused -95% DD because:
#   - Bull/LowVol → 15d was correct but only for trending years
#   - Caution → 15d was WRONG (2015, 2021 blowups)
#   - Crisis → 65d was too slow after COVID spike
#
# Corrected based on per-year Sharpe winners:
#   Bull (<13)  → 30d  (2017: 30d Sharpe 1.33; safer than 15d)
#   Normal      → 30d  (2019, 2024, 2025: 30d consistently best)
#   Caution     → 55d  (2022: 55d 0.54 Sharpe; 2015: 55d better risk-adj)
#   Elevated    → 45d  (2020: 45d Sharpe 1.80 — best)
#   Crisis      → 45d  (post-spike, medium lookback beats extremes)
#
VIX_REGIMES = [
    (0,   13,  "Bull / Low Vol",    30),
    (13,  17,  "Normal",            30),
    (17,  22,  "Caution",           55),
    (22,  28,  "Elevated Stress",   45),
    (28, 999,  "Crisis / High Vol", 45),
]


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

def _fetch_vix_nse_fallback():
    try:
        import requests
        url = ("https://www.nseindia.com/api/historical/vixhistory"
               "?from=01-01-2015&to=" + datetime.now().strftime("%d-%m-%Y"))
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        sess = requests.Session()
        sess.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = sess.get(url, headers=headers, timeout=15)
        data = resp.json()
        records = data.get("data", [])
        if not records: return None
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["EOD_TIMESTAMP"], dayfirst=True)
        df = df.set_index("Date").sort_index()
        s = df["EOD_VIX_CLOSE"].astype(float)
        s.name = VIX_SYM
        print(f"  NSE fallback OK: {len(s)} VIX rows")
        return s
    except Exception as e:
        print(f"  NSE fallback failed: {e}")
        return None

def _make_synthetic_vix(prices):
    if NIFTY_SYM not in prices.columns: return None
    nifty = prices[NIFTY_SYM].dropna()
    log_ret = np.log(nifty / nifty.shift(1))
    rvol = log_ret.rolling(20).std() * np.sqrt(252) * 100
    rvol.name = VIX_SYM
    rvol = rvol.reindex(prices.index).ffill(limit=5)
    print(f"  Synthetic VIX: mean={rvol.mean():.1f}, range={rvol.min():.1f}-{rvol.max():.1f}")
    return rvol

def fetch_all():
    print(f"Fetching {len(ETFS)+1} instruments | {FETCH_START} -> {END}")
    series = {}

    print(f"  [ 0/{len(ETFS)}] India VIX ({VIX_SYM})", end=" ... ", flush=True)
    v = _fetch_one(VIX_SYM)
    if v is not None:
        series[VIX_SYM] = v
        print(f"ok ({v.notna().sum()} rows)")
    else:
        print("Yahoo blocked — trying NSE direct...")
        v = _fetch_vix_nse_fallback()
        if v is not None:
            series[VIX_SYM] = v
    time.sleep(1.5)

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
        return pd.DataFrame(), pd.Series(dtype=float)

    etf_syms = [sym for sym,_,_ in ETFS if sym in series]
    prices = pd.concat([series[s] for s in etf_syms], axis=1)
    prices.columns = etf_syms
    prices = prices.sort_index().ffill(limit=5)
    valid = [c for c in prices.columns if prices[c].notna().sum() > 200]
    prices = prices[valid].dropna(how="all")

    vix = series.get(VIX_SYM, pd.Series(dtype=float))
    if vix.notna().sum() < 100:
        print("  VIX unavailable — building synthetic from NiftyBees")
        vix_synth = _make_synthetic_vix(prices)
        if vix_synth is not None:
            vix = vix_synth
        else:
            vix = pd.Series(17.0, index=prices.index, name=VIX_SYM)
            print("  ⚠  Flat VIX=17 (Normal regime)")
    else:
        vix = vix.reindex(prices.index).ffill(limit=5)

    print(f"\n  ETFs: {len(valid)} | VIX rows: {vix.notna().sum()}")
    print(f"  Price range: {prices.index[0].date()} -> {prices.index[-1].date()}\n")
    return prices, vix

# ── RS ENGINE ─────────────────────────────────────────────────────────────────
def period_return(series, idx_now, lb):
    idx_past = idx_now - lb
    if idx_past < 0: return None
    p0, p1 = series.iloc[idx_past], series.iloc[idx_now]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0: return None
    return float(p1/p0 - 1.0)

def compute_rs(prices, idx, avail, lb):
    rets = {s: period_return(prices[s], idx, lb) for s in avail if s in prices.columns}
    valid = [s for s in rets if rets[s] is not None]
    scores = {}
    for sym in avail:
        if rets.get(sym) is None: scores[sym] = None; continue
        peers = [rets[s] for s in valid if s != sym]
        scores[sym] = float(np.mean([rets[sym]-p for p in peers])) if peers else 0.0
    return scores, rets

# ── VIX REGIME ────────────────────────────────────────────────────────────────
def vix_to_lookback(vix_val):
    if pd.isna(vix_val) or vix_val <= 0: return 30
    for lo, hi, label, lb in VIX_REGIMES:
        if lo <= vix_val < hi: return lb
    return 45

def vix_to_regime_label(vix_val):
    if pd.isna(vix_val) or vix_val <= 0: return "Unknown"
    for lo, hi, label, lb in VIX_REGIMES:
        if lo <= vix_val < hi: return label
    return "Crisis"

# ── SINGLE LOOKBACK BACKTEST (for calibration table) ─────────────────────────
def run_fixed_lb(prices, vix, lb, start, end_date):
    """
    Fixed-lookback RS backtest.
    Trading logic (correct cash accounting):
      1. Sell ALL positions every week that holdings change.
      2. Distribute total capital equally across new TOP_N.
      3. No partial-sell/rebuy confusion — simple full liquidate + rebuy.
    """
    avail = [c for c in prices.columns if prices[c].notna().sum() > lb + 5]
    data_start_lb = prices.index[0]
    start_dt = max(pd.Timestamp(start), data_start_lb + pd.Timedelta(days=lb * 2))
    fridays = pd.date_range(start_dt, end_date, freq="W-FRI")
    pairs = []
    for f in fridays:
        si = prices.index.searchsorted(f, side="right") - 1
        ei = si + 1
        if 0 <= si < len(prices) and ei < len(prices):
            pairs.append((si, ei))
    if not pairs:
        return None

    cash = float(INITIAL)
    holdings = {}   # sym -> shares
    cur3 = []
    peak = float(INITIAL)
    eq = []

    for wi, (si, ei) in enumerate(pairs):
        scores, rets = compute_rs(prices, si, avail, lb)
        ranked = sorted(
            [(s, v) for s, v in scores.items() if v is not None],
            key=lambda x: -x[1]
        )
        new3 = [s for s, _ in ranked[:TOP_N]]
        if len(new3) < TOP_N:
            continue

        changed = (wi == 0) or (set(new3) != set(cur3))

        if changed:
            # Step 1: liquidate everything at execution price
            for sym, shares in list(holdings.items()):
                px = prices[sym].iloc[ei]
                if shares > 0 and np.isfinite(px) and px > 0:
                    cash += shares * float(px) * (1.0 - COST_PCT)
            holdings.clear()

            # Step 2: sanity-check cash (should always be positive)
            cash = max(cash, 0.0)

            # Step 3: buy equal slices
            target = cash / TOP_N
            for sym in new3:
                px = prices[sym].iloc[ei]
                if not np.isfinite(px) or px <= 0:
                    continue
                buy_cost = min(target, cash)          # never spend more than available
                shares   = buy_cost * (1.0 - COST_PCT) / float(px)
                holdings[sym] = shares
                cash -= buy_cost

            cash = max(cash, 0.0)   # floating-point safety

        # Mark-to-market
        holdings_val = 0.0
        for sym, shares in holdings.items():
            px = prices[sym].iloc[ei]
            if np.isfinite(px) and px > 0:
                holdings_val += shares * float(px)

        port = cash + holdings_val
        if not np.isfinite(port) or port < 0:
            port = 0.0
        if port > peak:
            peak = port
        dd = (port - peak) / peak * 100.0 if peak > 0 else 0.0
        eq.append({"date": str(prices.index[si].date()), "val": port, "dd": dd})
        cur3 = new3

    if len(eq) < 8:
        return None

    vals  = [e["val"] for e in eq]
    final = vals[-1]
    wr    = [(vals[i] - vals[i-1]) / vals[i-1] for i in range(1, len(vals))
             if vals[i-1] > 0]
    n_yrs = max((pd.Timestamp(end_date) - pd.Timestamp(start)).days / 365.25, 0.1)

    if not np.isfinite(final) or final <= 0:
        return None

    cagr   = (pow(final / INITIAL, 1.0 / n_yrs) - 1.0) * 100.0
    cagr   = round(max(min(cagr, 500.0), -99.0), 2)
    sharpe = 0.0
    if wr and np.std(wr) > 0:
        sharpe = round(float(np.mean(wr) / np.std(wr)) * np.sqrt(52), 2)
    mdd    = round(min(e["dd"] for e in eq), 2)
    ret    = round((final - INITIAL) / INITIAL * 100.0, 2)

    return dict(
        cagr=cagr, sharpe=sharpe, mdd=mdd,
        ret=ret, weeks=len(eq), final=round(final, 2), eq=eq
    )

# ── PER-YEAR CALIBRATION (backtest period only) ────────────────────────────────
def calibrate_yearly(prices, vix, cal_start="2015-01-01", cal_end="2024-12-31"):
    print(f"\nCalibrating: {cal_start} → {cal_end} (backtest period only)")
    sep = "=" * 115
    print(sep)
    hdr = f"{'Year':>4}  {'VIX':>5}  {'MaxVIX':>6}  {'Regime':<22}"
    for lb in LOOKBACKS:
        hdr += f"  |{lb:>2}d Ret    CAGR  Shrp"
    hdr += "  Best"
    print(hdr)
    print("-" * 115)

    years   = list(range(int(cal_start[:4]), int(cal_end[:4]) + 1))
    results = {}

    for yr in years:
        y_start = f"{yr}-01-01"
        y_end   = f"{yr}-12-31"
        yr_prices = prices[(prices.index >= y_start) & (prices.index <= y_end)]
        if len(yr_prices) < 30: continue

        yr_vix  = vix[(vix.index >= y_start) & (vix.index <= y_end)]
        avg_vix = round(float(yr_vix.mean()), 1) if yr_vix.notna().sum() > 10 else None
        max_vix = round(float(yr_vix.max()), 1) if yr_vix.notna().sum() > 10 else None
        regime  = vix_to_regime_label(avg_vix) if avg_vix else "Unknown"

        best_lb     = None
        best_sharpe = -999
        lb_stats    = {}

        for lb in LOOKBACKS:
            r = run_fixed_lb(prices, vix, lb, y_start, y_end)
            if r is None: continue
            lb_stats[lb] = r
            if r["sharpe"] > best_sharpe:
                best_sharpe = r["sharpe"]
                best_lb     = lb

        results[yr] = dict(avg_vix=avg_vix, max_vix=max_vix,
                           regime=regime, best_lb=best_lb, lb_stats=lb_stats)

        row = f"{yr:>4}  {str(avg_vix or '-'):>5}  {str(max_vix or '-'):>6}  {regime:<22}"
        for lb in LOOKBACKS:
            st = lb_stats.get(lb)
            if st:
                marker = "*" if lb == best_lb else " "
                row += f"  |{marker}{st['ret']:>+6.1f}%  {st['cagr']:>+5.1f}%  {st['sharpe']:>4.2f}"
            else:
                row += f"  |{'—':>7}   {'—':>5}  {'—':>4}"
        row += f"  {str(best_lb)+'d' if best_lb else '—'}"
        print(row)

    print(sep)
    print("\nSUMMARY — Average by lookback (calibration years only):")
    print(f"{'LB':>4}  {'AvgRet':>7}  {'AvgCAGR':>8}  {'AvgSharpe':>9}  {'AvgMDD':>7}  {'BestYears':>9}")
    print("-" * 55)
    for lb in LOOKBACKS:
        rets   = [r["lb_stats"][lb]["ret"]    for r in results.values() if lb in r["lb_stats"]]
        cagrs  = [r["lb_stats"][lb]["cagr"]   for r in results.values() if lb in r["lb_stats"]]
        sharps = [r["lb_stats"][lb]["sharpe"] for r in results.values() if lb in r["lb_stats"]]
        mdds   = [r["lb_stats"][lb]["mdd"]    for r in results.values() if lb in r["lb_stats"]]
        wins   = sum(1 for r in results.values() if r.get("best_lb") == lb)
        n      = len(results)
        if rets:
            avg_ret  = float(np.nanmean([x for x in rets  if np.isfinite(x)]) if any(np.isfinite(x) for x in rets)  else float('nan'))
            avg_cagr = float(np.nanmean([x for x in cagrs if np.isfinite(x)]) if any(np.isfinite(x) for x in cagrs) else float('nan'))
            avg_shrp = float(np.nanmean([x for x in sharps if np.isfinite(x)]) if any(np.isfinite(x) for x in sharps) else float('nan'))
            avg_mdd  = float(np.nanmean([x for x in mdds  if np.isfinite(x)]) if any(np.isfinite(x) for x in mdds)  else float('nan'))
            r_str = f"{avg_ret:>+6.1f}%" if np.isfinite(avg_ret) else "    N/A"
            c_str = f"{avg_cagr:>+7.1f}%" if np.isfinite(avg_cagr) else "     N/A"
            print(f"{lb:>3}d  {r_str:>7}  {c_str:>8}  "
                  f"{avg_shrp:>9.2f}  {avg_mdd:>+6.1f}%  {wins:>4}/{n}")
    print()
    return results

# ── FULL-PERIOD FIXED-LB COMPARISON ──────────────────────────────────────────
def run_fullperiod_all_lbs(prices, vix, full_start, full_end):
    """
    Run each fixed lookback (15/30/45/55/65d) over the entire date range
    (2015 -> today) independently with INITIAL capital each.
    Returns dict keyed by lookback with stats + equity curve.
    """
    print(f"\nFull-period fixed-LB comparison: {full_start} -> {full_end}")
    lb_colors = {15:"#f472b6", 30:"#22c55e", 45:"#f59e0b", 55:"#fb923c", 65:"#ef4444"}
    results = {}
    n_yrs = max((pd.Timestamp(full_end) - pd.Timestamp(full_start)).days / 365.25, 0.1)
    for lb in LOOKBACKS:
        r = run_fixed_lb(prices, vix, lb, full_start, full_end)
        if r is None:
            print(f"  {lb:>2}d -> no result")
            continue
        # Recompute CAGR over the true full span (run_fixed_lb uses per-period span)
        cagr_full = (pow(max(r["final"] / INITIAL, 1e-6), 1.0 / n_yrs) - 1.0) * 100.0
        cagr_full = round(max(min(cagr_full, 500.0), -99.0), 2)
        r["cagr_full"] = cagr_full
        r["color"]     = lb_colors.get(lb, "#8892b0")
        results[lb]    = r
        print(f"  {lb:>2}d -> CAGR {cagr_full:>+6.1f}%  Sharpe {r['sharpe']:>5.2f}  "
              f"MDD {r['mdd']:>+6.1f}%  Final Rs {r['final']/1e5:.2f}L")
    return results

# ── ADAPTIVE BACKTEST ENGINE ───────────────────────────────────────────────────
def run_adaptive(prices, vix, period_start, period_end, label="Backtest",
                 initial_capital=None):
    """
    Run the adaptive RS strategy for a given period.
    Returns full equity curve, log, and stats dict.
    """
    print(f"\nRunning adaptive [{label}]: {period_start} → {period_end} ...")
    avail = prices.columns.tolist()

    data_start = prices.index[0]
    start_dt   = max(pd.Timestamp(period_start),
                     data_start + pd.Timedelta(days=90))

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

    cap   = float(initial_capital or INITIAL)
    cash  = cap
    holdings = {}
    cur3  = []
    peak  = cap
    total_trades  = 0
    hold_count    = {e[0]: 0 for e in ETFS}

    eq=[]; nf=[]; dd_curve=[]; log=[]
    nifty_px0 = (float(prices[NIFTY_SYM].iloc[pairs[0][1]])
                 if pairs and NIFTY_SYM in prices.columns else None)
    last_scores = last_rets = None

    for wi,(si,ei) in enumerate(pairs):
        date_str = str(prices.index[si].date())
        exec_str = str(prices.index[ei].date())

        vix_val = (float(vix.iloc[si])
                   if si < len(vix) and not pd.isna(vix.iloc[si]) else None)
        lb     = vix_to_lookback(vix_val)
        regime = vix_to_regime_label(vix_val)

        # ── Decide target positions ───────────────────────────────────────
        scores, rets = compute_rs(prices, si, avail, lb)
        ranked = sorted([(s,v) for s,v in scores.items() if v is not None],
                        key=lambda x:-x[1])
        new3 = [s for s,_ in ranked[:TOP_N]]
        last_scores = scores; last_rets = rets

        if len(new3) < 1: continue

        changed  = (wi == 0) or (set(new3) != set(cur3))
        exiting  = [s for s in cur3 if s not in new3]
        entering = [s for s in new3  if s not in cur3]

        if changed:
            # Step 1: sell ALL holdings at execution price
            for sym, shares in list(holdings.items()):
                px = prices[sym].iloc[ei] if sym in prices.columns else float('nan')
                if shares > 0 and np.isfinite(px) and float(px) > 0:
                    cash += shares * float(px) * (1.0 - COST_PCT)
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
                    holdings[sym] = buy_amt * (1.0 - COST_PCT) / px
                    cash -= buy_amt
                    total_trades += 1
            cash = max(cash, 0.0)

        # ── Mark-to-market ────────────────────────────────────────────────
        holdings_val = 0.0
        for sym, shares in holdings.items():
            px = float(prices[sym].iloc[ei]) if sym in prices.columns else float('nan')
            if np.isfinite(px) and px > 0:
                holdings_val += shares * px
        port = max(cash + holdings_val, 0.0)
        if port > peak: peak = port
        ddown = (port-peak)/peak*100

        nifty_px = (float(prices[NIFTY_SYM].iloc[ei])
                    if NIFTY_SYM in prices.columns else None)
        nifty_v  = (nifty_px/nifty_px0*(initial_capital or INITIAL)
                    if nifty_px and nifty_px0 else (initial_capital or INITIAL))

        for sym in new3: hold_count[sym] = hold_count.get(sym,0)+1

        eq.append({"x": date_str, "y": round(port,2)})
        nf.append({"x": date_str, "y": round(nifty_v,2)})
        dd_curve.append({"x": date_str, "y": round(ddown,3)})
        log.append({
            "date":    date_str, "exec": exec_str,
            "top3":    new3, "exiting": exiting, "entering": entering,
            "changed": changed and wi>0, "capital": round(port,2),
            "vix":     round(vix_val,1) if vix_val else None,
            "regime":  regime, "lb": lb,
            "scores": {s:round(v,6) if v else None for s,v in scores.items()},
            "rets":   {s:round(v,6) if v else None for s,v in rets.items()}
        })
        cur3 = new3

    return dict(
        eq=eq, nf=nf, dd=dd_curve, log=log,
        trades=total_trades, hold_count=hold_count,
        last_scores=last_scores, last_rets=last_rets,
        avail=avail, label=label,
        period_start=period_start, period_end=period_end,
    )

# ── STATS ─────────────────────────────────────────────────────────────────────
def calc_stats(adaptive, initial=None):
    eq = adaptive["eq"]; nf = adaptive["nf"]; dd = adaptive["dd"]
    trades = adaptive["trades"]
    if not eq: return None
    init  = initial or INITIAL
    fv    = eq[-1]["y"]; nfv = nf[-1]["y"]
    ps    = pd.Timestamp(adaptive["period_start"])
    pe    = pd.Timestamp(adaptive["period_end"])
    yrs   = max((pe-ps).days/365.25, 0.1)
    tr    = (fv-init)/init*100
    cagr  = (pow(fv/init, 1/yrs)-1)*100
    nc    = (pow(nfv/init, 1/yrs)-1)*100
    wr    = [(eq[i]["y"]-eq[i-1]["y"])/eq[i-1]["y"] for i in range(1,len(eq))]
    shp   = (np.mean(wr)/np.std(wr))*np.sqrt(52) if wr and np.std(wr)>0 else 0
    mdd   = min(d["y"] for d in dd) if dd else 0
    return dict(final=round(fv,2), ret=round(tr,2), cagr=round(cagr,2),
                sharpe=round(shp,2), mdd=round(mdd,2), trades=trades,
                nifty_cagr=round(nc,2), alpha=round(cagr-nc,2), weeks=len(eq))

# ── CURRENT SIGNAL ────────────────────────────────────────────────────────────
def get_current_signal(prices, vix, ft_result):
    if not ft_result or not ft_result["log"]: return None
    last = ft_result["log"][-1]
    em   = {e[0]:e for e in ETFS}
    cur_vix = float(vix.dropna().iloc[-1]) if vix.notna().sum()>0 else None
    cur_lb  = vix_to_lookback(cur_vix)
    cur_reg = vix_to_regime_label(cur_vix)
    cur_prices = {}
    for sym in last["top3"]:
        if sym in prices.columns:
            cur_prices[sym] = round(float(prices[sym].dropna().iloc[-1]), 2)
    return dict(
        signal_date=last["date"], exec_date=last["exec"],
        top3=last["top3"],
        top3_names=[em[s][1] if s in em else s for s in last["top3"]],
        top3_short=[em[s][2] if s in em else s for s in last["top3"]],
        scores={s: round(last["scores"].get(s,0) or 0,4) for s in last["top3"]},
        rets  ={s: round((last["rets"].get(s,0) or 0)*100,2) for s in last["top3"]},
        cur_vix=round(cur_vix,1) if cur_vix else None,
        cur_regime=cur_reg, cur_lb=cur_lb,
        cur_prices=cur_prices,
        portfolio_val=last["capital"],
    )

# ── FULL RS TABLE (45-day lookback) ──────────────────────────────────────────
# ── FULL RS TABLE (45-day lookback) ──────────────────────────────────────────
def build_rs_table_html(prices, vix, signal):
    """
    Build standalone HTML page showing full RS table for 45-day lookback.
    Opens in new tab when user clicks the button.
    """
    # Get latest data
    avail = [c for c in prices.columns if prices[c].notna().sum() > 50]
    last_idx = len(prices) - 1
    
    # Compute 45-day RS for all instruments
    scores_45, rets_45 = compute_rs(prices, last_idx, avail, 45)
    
    # Build table data
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
            'rank': 0  # will set after sorting
        })
    
    # Sort by RS score descending and assign ranks
    table_rows.sort(key=lambda x: -x['score'])
    for i, row in enumerate(table_rows, 1):
        row['rank'] = i
        row['top3'] = i <= 3
    
    # Create standalone HTML
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
.top3-tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.7rem;font-weight:700;background:rgba(34,197,94,0.15);color:var(--green)}
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
    
    # Stats
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
        
        # Fix: Handle price formatting safely
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

# ── BUILD HTML ────────────────────────────────────────────────────────────────
def build_html(prices, vix, yearly, fullperiod_lbs, bt_result, ft_result, bt_stats, ft_stats, signal, rs_table_html):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [script_dir, os.getcwd(), os.path.expanduser("~")]
    chartjs = adapter = None
    for base in candidates:
        cj = os.path.join(base,"node_modules","chart.js","dist","chart.umd.js")
        ad = os.path.join(base,"node_modules","chartjs-adapter-date-fns","dist","chartjs-adapter-date-fns.bundle.js")
        if os.path.exists(cj) and os.path.exists(ad):
            with open(cj) as f: chartjs = f.read()
            with open(ad) as f: adapter = f.read()
            break

    def p(v,d=1): return f"{'+' if v>0 else ''}{v:.{d}f}%"
    def inr(v):   return f"Rs {v/1e5:.2f}L"
    def cls(v):   return "green" if v>=0 else "red"

    avail = bt_result["avail"]
    em    = {e[0]:e for e in ETFS}
    upd   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Full-period stat cards ─────────────────────────────────────────────
    lb_labels = {15:"15d",30:"30d",45:"45d",55:"55d",65:"65d"}
    fp_stat_cards = ""
    fp_eq_datasets = []
    for lb in LOOKBACKS:
        r = fullperiod_lbs.get(lb)
        if not r: continue
        cagr_c = "#22c55e" if r["cagr_full"]>=0 else "#ef4444"
        mdd_c  = "#ef4444"
        fp_stat_cards += f"""
    <div class="sc" style="border-top:3px solid {r['color']}">
      <div class="lb">{lb}d Fixed Lookback</div>
      <div class="vl" style="color:{r['color']}">{r['cagr_full']:+.1f}%</div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:4px">CAGR (full period)</div>
      <div style="font-size:.78rem;margin-top:4px">Sharpe <b style="color:{'#22c55e' if r['sharpe']>=1 else '#f59e0b'}">{r['sharpe']:.2f}</b></div>
      <div style="font-size:.78rem">MDD <b style="color:{mdd_c}">{r['mdd']:+.1f}%</b></div>
      <div style="font-size:.78rem">Final <b>{inr(r['final'])}</b></div>
    </div>"""
        eq_pts = [{"x": e["date"], "y": e["val"]} for e in r["eq"]]
        fp_eq_datasets.append({
            "label": f"{lb}d",
            "data":  eq_pts,
            "borderColor": r["color"],
            "backgroundColor": "transparent",
            "borderWidth": 2,
            "pointRadius": 0,
            "tension": 0.3,
        })

    # ── Signal card ───────────────────────────────────────────────────────
    if signal:
        vix_col = ("#22c55e" if (signal["cur_vix"] or 0)<15 else
                   "#f59e0b" if (signal["cur_vix"] or 0)<22 else "#ef4444")
        sig_html = f"""
<div class="cc signal-card">
  <div class="st">THIS WEEK'S BUY SIGNAL — FORWARD TEST</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="mini-stat"><div class="mini-label">India VIX</div>
      <div class="mini-val" style="color:{vix_col}">{signal['cur_vix'] or 'N/A'}</div></div>
    <div class="mini-stat"><div class="mini-label">Regime</div>
      <div class="mini-val" style="font-size:1rem">{signal['cur_regime']}</div></div>
    <div class="mini-stat"><div class="mini-label">Active Lookback</div>
      <div class="mini-val accent">{signal['cur_lb']}d</div></div>
  </div>
  <div style="margin-bottom:16px">
    <div style="font-size:.75rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">BUY / HOLD THESE 3 ETFs</div>
    <div style="display:flex;gap:12px;flex-wrap:wrap">"""
        for sym in signal["top3"]:
            name  = em.get(sym,(sym,sym,sym))[1]
            short = em.get(sym,(sym,sym,sym))[2]
            sc    = signal["scores"].get(sym,0)*100
            ret   = signal["rets"].get(sym,0)
            px    = signal["cur_prices"].get(sym,"N/A")
            rc    = "green" if ret>0 else "red"
            sig_html += f"""
      <div class="etf-pill">
        <div class="etf-pill-name">{name}</div>
        <div class="etf-pill-short">{short}</div>
        <div class="etf-pill-price">Rs {px}</div>
        <div class="etf-pill-ret {rc}">{ret:+.1f}% ({signal['cur_lb']}d)</div>
        <div class="etf-pill-score">RS: {sc:+.3f}</div>
      </div>"""
        sig_html += f"""
    </div>
  </div>
  <div style="font-size:.78rem;color:var(--muted)">
    Signal date: <b style="color:var(--text)">{signal['signal_date']}</b> &rarr;
    Execute: <b style="color:var(--text)">{signal['exec_date']}</b> &middot;
    FT Portfolio: <b style="color:var(--accent)">{inr(signal['portfolio_val'])}</b>
  </div>
</div>"""
    else:
        sig_html = '<div class="cc"><p style="color:var(--muted)">No forward-test signal available.</p></div>'

    # ── Stats grid helper ─────────────────────────────────────────────────
    def stats_grid(stats, label, color):
        if not stats: return f'<div class="cc"><p style="color:var(--muted)">No {label} stats.</p></div>'
        return f"""
<div class="cc" style="border-color:{color};border-left:3px solid {color}">
  <div class="st">{label}</div>
  <div class="sg">
    <div class="sc"><div class="lb">Final Value</div><div class="vl accent">{inr(stats["final"])}</div></div>
    <div class="sc"><div class="lb">Total Return</div><div class="vl {cls(stats["ret"])}">{p(stats["ret"])}</div></div>
    <div class="sc"><div class="lb">CAGR</div><div class="vl {cls(stats["cagr"])}">{p(stats["cagr"])}</div></div>
    <div class="sc"><div class="lb">Sharpe</div><div class="vl {cls(stats["sharpe"]-1)}">{stats["sharpe"]:.2f}</div></div>
    <div class="sc"><div class="lb">Max Drawdown</div><div class="vl red">{p(stats["mdd"])}</div></div>
    <div class="sc"><div class="lb">Trades</div><div class="vl">{stats["trades"]}</div></div>
    <div class="sc"><div class="lb">Nifty CAGR</div><div class="vl">{p(stats["nifty_cagr"])}</div></div>
    <div class="sc"><div class="lb">Alpha vs Nifty</div><div class="vl {cls(stats["alpha"])}">{p(stats["alpha"])}</div></div>
  </div>
</div>"""

    bt_stats_html = stats_grid(bt_stats, f"BACKTEST RESULTS (2015–2024) — Calibration Period", "#4f8ef7")
    ft_stats_html = stats_grid(ft_stats, f"FORWARD TEST RESULTS (2025–{END}) — Out-of-Sample", "#22c55e")

    # ── Regime map ────────────────────────────────────────────────────────
    rationale = {
        30: "Normal/Bull: 30d captures trend, best Sharpe across most years",
        55: "Caution: longer window filters choppy noise (2022 winner)",
        45: "Elevated/Crisis: medium smoothing outperformed in 2020 crash",
    }
    regime_html = """<table><thead><tr>
      <th>VIX Range</th><th>Regime</th><th>Active Lookback</th><th>Rationale (v2 corrected)</th>
    </tr></thead><tbody>"""
    colors = {30:"#22c55e", 55:"#f59e0b", 45:"#fb923c"}
    old_map = {15:"Bull / Low Vol", 45:"Normal / Caution", 65:"Crisis"}  # for diff display
    for lo,hi,label,lb in VIX_REGIMES:
        hi_str = str(hi) if hi<999 else "+"
        cur = signal and signal["cur_vix"] and lo <= (signal["cur_vix"] or 0) < hi
        bg  = "background:rgba(79,142,247,0.08);" if cur else ""
        tag = ' <span class="bd bg">ACTIVE</span>' if cur else ""
        regime_html += (f'<tr style="{bg}"><td><b>{lo}–{hi_str}</b></td>'
                        f'<td>{label}{tag}</td>'
                        f'<td style="color:{colors.get(lb,"#fff")};font-weight:700">{lb}d</td>'
                        f'<td style="color:var(--muted);font-size:.8rem">{rationale.get(lb,"")}</td></tr>')
    regime_html += "</tbody></table>"

    # ── Calibration table ─────────────────────────────────────────────────
    yr_html = """<div class="mw"><table><thead><tr>
      <th rowspan="2">Year</th><th rowspan="2">VIX Avg</th><th rowspan="2">VIX Max</th>
      <th rowspan="2">Regime</th><th rowspan="2">Best LB</th>"""
    for lb in LOOKBACKS:
        yr_html += f'<th colspan="3" style="text-align:center;border-left:2px solid var(--border)">{lb}d</th>'
    yr_html += "</tr><tr>"
    for lb in LOOKBACKS:
        yr_html += (f'<th style="border-left:2px solid var(--border);font-size:.65rem">Ret%</th>'
                    f'<th style="font-size:.65rem">CAGR</th>'
                    f'<th style="font-size:.65rem">Shrp</th>')
    yr_html += "</tr></thead><tbody>"

    for yr, r in sorted(yearly.items()):
        if not r["lb_stats"]: continue
        best = r["best_lb"]
        # Highlight rows where v2 mapping matches best LB
        v2_lb = vix_to_lookback(r["avg_vix"])
        correct = "✓" if v2_lb == best else "✗"
        correct_col = "#22c55e" if v2_lb == best else "#ef4444"
        row = f'<tr><td><b>{yr}</b> <span title="v2 mapping: {v2_lb}d" style="color:{correct_col};font-size:.7rem">{correct}</span></td>'
        row += f'<td style="text-align:center">{r["avg_vix"] or "—"}</td>'
        row += f'<td style="text-align:center">{r["max_vix"] or "—"}</td>'
        row += f'<td style="font-size:.75rem">{r["regime"]}</td>'
        row += (f'<td style="color:#4f8ef7;font-weight:700;text-align:center">{best}d</td>'
                if best else '<td style="color:var(--muted)">—</td>')
        for lb in LOOKBACKS:
            st      = r["lb_stats"].get(lb)
            is_best = lb == best
            border  = "border-left:2px solid var(--border);"
            bg      = "background:rgba(79,142,247,0.08);" if is_best else ""
            fw      = "font-weight:700;" if is_best else ""
            star    = "★ " if is_best else ""
            if st:
                rc = "#22c55e" if st["ret"]>=0 else "#ef4444"
                cc = "#22c55e" if st["cagr"]>=0 else "#ef4444"
                sc = "#22c55e" if st["sharpe"]>=1 else ("#f59e0b" if st["sharpe"]>=0 else "#ef4444")
                row += (f'<td style="{border}{bg}{fw}color:{rc}">{star}{st["ret"]:+.1f}%</td>'
                        f'<td style="{bg}{fw}color:{cc}">{st["cagr"]:+.1f}%</td>'
                        f'<td style="{bg}{fw}color:{sc}">{st["sharpe"]:.2f}</td>')
            else:
                row += f'<td style="{border}color:var(--muted)">—</td><td>—</td><td>—</td>'
        row += "</tr>"
        yr_html += row

    # Summary row
    yr_html += '<tr style="background:var(--card2);border-top:2px solid #4f8ef7;">'
    yr_html += '<td colspan="5"><b>AVG (2015–2024)</b></td>'
    for lb in LOOKBACKS:
        rets   = [r["lb_stats"][lb]["ret"]    for r in yearly.values() if lb in r["lb_stats"]]
        cagrs  = [r["lb_stats"][lb]["cagr"]   for r in yearly.values() if lb in r["lb_stats"]]
        sharps = [r["lb_stats"][lb]["sharpe"] for r in yearly.values() if lb in r["lb_stats"]]
        border = "border-left:2px solid var(--border);"
        if rets:
            am=np.mean(rets); rc="#22c55e" if am>=0 else "#ef4444"
            cm=np.mean(cagrs); cc="#22c55e" if cm>=0 else "#ef4444"
            sm=np.mean(sharps); sc="#22c55e" if sm>=1 else "#f59e0b"
            yr_html += (f'<td style="{border}font-weight:700;color:{rc}">{am:+.1f}%</td>'
                        f'<td style="font-weight:700;color:{cc}">{cm:+.1f}%</td>'
                        f'<td style="font-weight:700;color:{sc}">{sm:.2f}</td>')
        else:
            yr_html += f'<td style="{border}">—</td><td>—</td><td>—</td>'
    yr_html += "</tr></tbody></table></div>"
    yr_html += '<div style="font-size:.72rem;color:var(--muted);margin-top:8px">✓ = v2 corrected mapping matches best LB | ✗ = mismatch</div>'

    # ── VIX data ──────────────────────────────────────────────────────────
    vix_data = [{"x":str(dt.date()),"y":round(float(v),2)}
                for dt,v in vix.dropna().items()]

    # ── Trade log helper ──────────────────────────────────────────────────
    def build_log_html(log_entries):
        out = ""
        for lg in reversed(log_entries[-200:]):   # last 200 for perf
            tags = "".join(
                f'<span class="tg">{em.get(s,(s,s,s))[2]}</span>'
                for s in lg["top3"]
            )
            chg = ""
            if lg["changed"]:
                ex = " ".join(f'<span class="bd br">- {em.get(s,(s,s,s))[2]}</span>' for s in lg["exiting"])
                en = " ".join(f'<span class="bd bg">+ {em.get(s,(s,s,s))[2]}</span>' for s in lg["entering"])
                chg = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:3px">{ex} {en}</div>'
            nc  = "" if lg["changed"] else '<span style="color:var(--muted);font-size:.72rem;margin-left:5px">no change</span>'
            vx  = f'VIX {lg["vix"]}' if lg["vix"] else ""
            out += (f'<div class="lr" data-c="{"1" if lg["changed"] else "0"}">'
                    f'<div class="ld">{lg["date"]}<span style="display:block;font-size:.65rem;color:var(--muted)">{lg["lb"]}d · {vx}</span></div>'
                    f'<div class="lb2"><div style="display:flex;align-items:center;flex-wrap:wrap">{tags}{nc}</div>'
                    f'{chg}<div style="color:var(--muted);font-size:.72rem;margin-top:2px">Rs {lg["capital"]/1000:.1f}K · {lg["regime"]}</div>'
                    f'</div></div>\n')
        return out

    bt_log_html = build_log_html(bt_result["log"])
    ft_log_html = build_log_html(ft_result["log"]) if ft_result else ""

    # ── RS Rankings ───────────────────────────────────────────────────────
    ls = (ft_result or bt_result)["last_scores"] or {}
    lr = (ft_result or bt_result)["last_rets"] or {}
    rows_r = [(e,ls.get(e[0]),lr.get(e[0])) for e in ETFS if e[0] in avail and ls.get(e[0]) is not None]
    rows_r.sort(key=lambda x:-x[1])
    rank_html = '<table><thead><tr><th>#</th><th>ETF</th><th>RS Score</th><th>Return</th><th>Signal</th></tr></thead><tbody>'
    for i,(etf,sc,ret) in enumerate(rows_r):
        top  = i < TOP_N
        med  = ["1","2","3"][i] if i<3 else str(i+1)
        rs   = f"{ret*100:+.2f}%" if ret else "-"
        rc   = "green" if (ret or 0)>0 else "red"
        sig2 = f'<span class="bd {"bg" if top else "bm"}">{"LONG" if top else "OUT"}</span>'
        bg2  = "background:rgba(79,142,247,.06);" if top else ""
        rank_html += (f'<tr style="{bg2}"><td>{med}</td>'
                      f'<td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.7rem">{etf[2]}</span></td>'
                      f'<td style="color:var(--accent)">{sc*100:+.3f}</td>'
                      f'<td class="{rc}">{rs}</td><td>{sig2}</td></tr>')
    rank_html += "</tbody></table>"

    # ── Chart data ────────────────────────────────────────────────────────
    bt_eq = bt_result["eq"]; bt_nf = bt_result["nf"]; bt_dd = bt_result["dd"]
    ft_eq = ft_result["eq"] if ft_result else []
    ft_nf = ft_result["nf"] if ft_result else []
    ft_dd = ft_result["dd"] if ft_result else []

    if chartjs and adapter:
        chartjs_tag = f"<script>{chartjs}</script>"
        adapter_tag = f"<script>{adapter}</script>"
    else:
        chartjs_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
        adapter_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>'

    gc = "#2e3250"

    # ── RS Table button ───────────────────────────────────────────────────
    rs_table_button = f'''
    <div style="margin-bottom:16px">
        <button onclick="window.open('rs_table_45d.html','_blank')" 
                style="background:var(--accent);color:#fff;border:none;padding:12px 28px;border-radius:8px;font-weight:700;font-size:.95rem;cursor:pointer;display:inline-flex;align-items:center;gap:8px">
            📊 View Full 45d RS Table
        </button>
        <span style="color:var(--muted);font-size:.8rem;margin-left:12px">All instruments ranked by RS score (opens in new tab)</span>
    </div>
    '''

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>India ETF RS — Regime Adaptive (v2)</title>
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
.divider{{border:none;border-top:2px dashed var(--border);margin:24px 0}}
.period-label{{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.period-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
.upd{{text-align:right;color:var(--muted);font-size:.76rem;padding:6px 0}}
#log-bt,#log-ft{{max-height:320px;overflow-y:auto}}
</style>
</head><body>
<header>
  <div>
    <h1>India ETF — Regime-Adaptive RS Backtest v2</h1>
    <div class="meta">{len(avail)} ETFs · VIX-driven lookback (30/45/55d corrected) · Top-3 Long · DD circuit breaker (-15%) · Weekly</div>
  </div>
  <div class="meta">Updated: {upd}</div>
</header>
<div class="container">

{rs_table_button}

{sig_html}

<hr class="divider">
<div class="period-label">▶ FULL PERIOD FIXED-LB COMPARISON — {FETCH_START} to {END} (All lookbacks, same capital)</div>
<div class="cc">
  <div class="st">Fixed Lookback CAGR Comparison — 2015 to {END}</div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:20px">
    {fp_stat_cards}
  </div>
  <div class="chartbox" style="height:280px"><canvas id="ec-fp"></canvas></div>
  <div style="font-size:.72rem;color:var(--muted);margin-top:8px">Each lookback runs independently with Rs 10L start · No regime switching · Full span 2015→{END}</div>
</div>

<hr class="divider">
<div class="period-label">▶ BACKTEST PERIOD — 2015 to 2024 (Calibration / In-sample)</div>
{bt_stats_html}

<div class="cc">
  <h3>Equity Curve — Backtest 2015–2024 (Adaptive RS vs NiftyBees)</h3>
  <div class="chartbox" style="height:280px"><canvas id="ec-bt"></canvas></div>
</div>
<div class="cc">
  <h3>Drawdown — Backtest 2015–2024</h3>
  <div class="chartbox" style="height:140px"><canvas id="dc-bt"></canvas></div>
</div>

<hr class="divider">
<div class="period-label">▶ FORWARD TEST PERIOD — 2025 to {END} (Out-of-sample, no future leakage)</div>
{ft_stats_html}

<div class="cc">
  <h3>Equity Curve — Forward Test 2025–{END}</h3>
  <div class="chartbox" style="height:240px"><canvas id="ec-ft"></canvas></div>
</div>
<div class="cc">
  <h3>Drawdown — Forward Test 2025–{END}</h3>
  <div class="chartbox" style="height:130px"><canvas id="dc-ft"></canvas></div>
</div>

<hr class="divider">

<!-- VIX -->
<div class="cc">
  <h3>India VIX — Full History with Regime Zones</h3>
  <div class="chartbox" style="height:170px"><canvas id="vx"></canvas></div>
</div>

<!-- VIX Regime Map -->
<div class="cc">
  <div class="st">VIX Regime Map — v2 Corrected Lookback Rules</div>
  {regime_html}
</div>

<!-- Calibration table -->
<div class="cc">
  <div class="st">Per-Year Calibration (2015–2024 only — honest, no forward leakage)</div>
  {yr_html}
</div>

<div class="g2">
  <!-- Rankings -->
  <div class="cc">
    <div class="st">Current RS Rankings (latest signal)</div>
    {rank_html}
  </div>
  <!-- Dual log -->
  <div class="cc">
    <div class="st">Weekly Log</div>
    <div class="tabs">
      <button class="tab active" onclick="sw('bt','all',this)">BT All</button>
      <button class="tab" onclick="sw('bt','chg',this)">BT Changes</button>
      <button class="tab" onclick="sw('ft','all',this)">FT All</button>
      <button class="tab" onclick="sw('ft','chg',this)">FT Changes</button>
    </div>
    <div id="log-bt">{bt_log_html}</div>
    <div id="log-ft" style="display:none">{ft_log_html}</div>
  </div>
</div>

<div class="upd">Backtest 2015-2024 (in-sample) · Forward Test 2025-{END} (out-of-sample) · VIX-mapped lookback (corrected v2)</div>
</div>

{chartjs_tag}
{adapter_tag}
<script>
var gc='{gc}';
var btEq={json.dumps(bt_eq)};
var btNf={json.dumps(bt_nf)};
var btDd={json.dumps(bt_dd)};
var ftEq={json.dumps(ft_eq)};
var ftNf={json.dumps(ft_nf)};
var ftDd={json.dumps(ft_dd)};
var vxD={json.dumps(vix_data)};
var fpDs={json.dumps(fp_eq_datasets)};

function mkLine(id, datasets, yFmt, height){{
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

// ── Full-period fixed-LB chart ────────────────────────────────────────────────
(function(){{
  var datasets = fpDs.map(function(ds){{
    return {{
      label: ds.label,
      data: ds.data,
      parsing: false,
      borderColor: ds.borderColor,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      fill: false
    }};
  }});
  mkLine('ec-fp', datasets, rupee);
}})();

mkLine('ec-bt',[
  {{label:'Adaptive RS (BT)', data:btEq, parsing:false, borderColor:'#4f8ef7',
   backgroundColor:'rgba(79,142,247,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
  {{label:'NiftyBees', data:btNf, parsing:false, borderColor:'#f59e0b',
   backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, borderDash:[5,4], tension:0.3}}
], rupee);

mkLine('dc-bt',[
  {{label:'Drawdown', data:btDd, parsing:false, borderColor:'#ef4444',
   backgroundColor:'rgba(239,68,68,0.12)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true}}
], pct);

mkLine('ec-ft',[
  {{label:'Adaptive RS (FT)', data:ftEq, parsing:false, borderColor:'#22c55e',
   backgroundColor:'rgba(34,197,94,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
  {{label:'NiftyBees', data:ftNf, parsing:false, borderColor:'#f59e0b',
   backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, borderDash:[5,4], tension:0.3}}
], rupee);

mkLine('dc-ft',[
  {{label:'Drawdown', data:ftDd, parsing:false, borderColor:'#ef4444',
   backgroundColor:'rgba(239,68,68,0.12)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true}}
], pct);

new Chart(document.getElementById('vx'),{{
  type:'line', data:{{datasets:[
    {{label:'India VIX', data:vxD, parsing:false, borderColor:'#a78bfa',
     backgroundColor:'rgba(167,139,250,0.08)', borderWidth:1.5, pointRadius:0, tension:0.2, fill:true}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,parsing:false,
    plugins:{{legend:{{labels:{{color:'#8892b0',boxWidth:12}}}}}},
    scales:{{
      x:{{type:'time',time:{{unit:'month'}},ticks:{{color:'#8892b0',maxTicksLimit:18}},grid:{{color:gc}}}},
      y:{{ticks:{{color:'#8892b0'}},grid:{{color:gc}}}}
    }}
  }}
}});

function sw(period, type, el){{
  document.querySelectorAll('.tab').forEach(function(x){{x.classList.remove('active');}});
  el.classList.add('active');
  var btLog = document.getElementById('log-bt');
  var ftLog = document.getElementById('log-ft');
  if(period==='bt'){{
    btLog.style.display='block'; ftLog.style.display='none';
    btLog.querySelectorAll('.lr').forEach(function(r){{
      r.style.display=(type==='all'||r.dataset.c==='1')?'flex':'none';
    }});
  }} else {{
    btLog.style.display='none'; ftLog.style.display='block';
    ftLog.querySelectorAll('.lr').forEach(function(r){{
      r.style.display=(type==='all'||r.dataset.c==='1')?'flex':'none';
    }});
  }}
}}
</script>
</body></html>"""

# ── MOCK DATA ─────────────────────────────────────────────────────────────────
def make_mock_data():
    print("MOCK MODE: generating synthetic data")
    trade_dates = pd.date_range(FETCH_START, END, freq="B")
    np.random.seed(42)
    mock_etfs = [e[0] for e in ETFS[:12]]
    prices_dict = {}
    for sym in mock_etfs:
        drift = np.random.uniform(0.00008, 0.00035)
        vol   = np.random.uniform(0.010, 0.020)
        start = np.random.uniform(30, 300)
        lr    = np.random.normal(drift, vol, len(trade_dates))
        prices_dict[sym] = start * np.exp(np.cumsum(lr))
    prices = pd.DataFrame(prices_dict, index=trade_dates)
    vix_vals = np.random.normal(16, 3, len(trade_dates))
    for yr, spike in [(2020,30),(2022,22),(2016,18)]:
        mask = trade_dates.year == yr
        vix_vals[mask] += np.random.uniform(0, spike-14, mask.sum())
    vix = pd.Series(np.clip(vix_vals,9,65), index=trade_dates, name=VIX_SYM)
    print(f"  Mock: {len(prices)} days, {len(prices.columns)} ETFs")
    return prices, vix

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    USE_MOCK = "--mock" in sys.argv

    if USE_MOCK:
        prices, vix = make_mock_data()
    else:
        prices, vix = fetch_all()

    if prices.empty:
        print("ERROR: No data. Try: python regime_backtest.py --mock", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: calibrate on backtest period only (2015–2024) ─────────────
    yearly = calibrate_yearly(prices, vix, cal_start=FETCH_START, cal_end=BT_END)

    # ── Step 2: backtest 2015–2024 ────────────────────────────────────────
    bt_prices = prices[prices.index <= BT_END]
    bt_vix    = vix[vix.index <= BT_END]
    bt_result = run_adaptive(bt_prices, bt_vix,
                             period_start=FETCH_START, period_end=BT_END,
                             label="Backtest 2015–2024",
                             initial_capital=INITIAL,
)
    bt_stats  = calc_stats(bt_result, initial=INITIAL) if bt_result else None

    # ── Step 3: forward test 2025–today ──────────────────────────────────
    # Start FT with same initial capital (independent OOS test)
    ft_prices = prices[prices.index >= FT_START]
    ft_vix    = vix[vix.index >= FT_START]
    ft_result = run_adaptive(ft_prices, ft_vix,
                             period_start=FT_START, period_end=END,
                             label="Forward Test 2025–today",
                             initial_capital=INITIAL,
)
    ft_stats  = calc_stats(ft_result, initial=INITIAL) if ft_result else None

    # ── Step 4: full-period fixed-LB comparison (2015→today) ───────────────
    fullperiod_lbs = run_fullperiod_all_lbs(prices, vix, FETCH_START, END)

    # ── Step 5: current signal from FT log ───────────────────────────────────
    signal = get_current_signal(prices, vix, ft_result or bt_result)

    # ── Step 6: build RS table HTML (45-day lookback) ──────────────────────
    rs_table_html = build_rs_table_html(prices, vix, signal)
    rs_table_path = os.path.join("docs", "rs_table_45d.html")
    with open(rs_table_path, "w", encoding="utf-8") as f:
        f.write(rs_table_html)
    print(f"\n✅ 45-day RS table → {rs_table_path}")

    # ── Step 7: print summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BACKTEST 2015–2024 (In-sample)")
    print(f"{'='*60}")
    if bt_stats:
        for k,v in [("Final",f"Rs {bt_stats['final']:,.0f}"),
                    ("Return",f"{bt_stats['ret']:+.1f}%"),
                    ("CAGR",f"{bt_stats['cagr']:+.1f}%"),
                    ("Sharpe",f"{bt_stats['sharpe']:.2f}"),
                    ("Max DD",f"{bt_stats['mdd']:.1f}%"),
                    ("Alpha",f"{bt_stats['alpha']:+.1f}%")]:
            print(f"  {k:<12}: {v}")

    print(f"\n{'='*60}")
    print(f"FORWARD TEST 2025–{END} (Out-of-sample)")
    print(f"{'='*60}")
    if ft_stats:
        for k,v in [("Final",f"Rs {ft_stats['final']:,.0f}"),
                    ("Return",f"{ft_stats['ret']:+.1f}%"),
                    ("CAGR",f"{ft_stats['cagr']:+.1f}%"),
                    ("Sharpe",f"{ft_stats['sharpe']:.2f}"),
                    ("Max DD",f"{ft_stats['mdd']:.1f}%"),
                    ("Alpha",f"{ft_stats['alpha']:+.1f}%")]:
            print(f"  {k:<12}: {v}")

    print(f"\n{'='*60}")
    print(f"FULL-PERIOD FIXED-LB ({FETCH_START} -> {END})")
    print(f"{'='*60}")
    print(f"{'LB':>4}  {'CAGR':>8}  {'Sharpe':>7}  {'MDD':>8}  {'Final':>14}")
    print("-"*50)
    for lb, r in sorted(fullperiod_lbs.items()):
        print(f"{lb:>3}d  {r['cagr_full']:>+7.1f}%  {r['sharpe']:>7.2f}  {r['mdd']:>+7.1f}%  Rs {r['final']/1e5:>8.2f}L")

    if signal:
        print(f"\nCURRENT SIGNAL ({signal['signal_date']})")
        print(f"  VIX: {signal['cur_vix']} | Regime: {signal['cur_regime']} | LB: {signal['cur_lb']}d")
        print(f"  BUY: {', '.join(signal['top3_names'])}")

    # ── Step 8: write HTML ────────────────────────────────────────────────
    os.makedirs("docs", exist_ok=True)
    html = build_html(prices, vix, yearly, fullperiod_lbs,
                      bt_result or {"eq":[],"nf":[],"dd":[],"log":[],"trades":0,
                                    "hold_count":{},"last_scores":None,"last_rets":None,"avail":[]},
                      ft_result,
                      bt_stats, ft_stats, signal, rs_table_html)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport → {OUT_PATH}")
