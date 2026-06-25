"""
Regime-Adaptive RS Backtester
==============================
- Fetches 10 years of NSE ETF + India VIX data
- For each calendar year, tests all lookbacks (15,30,45,55,65d)
- Maps VIX range -> best lookback via empirical calibration
- Runs adaptive backtest: each week picks lookback based on current VIX
- Outputs: per-year analysis table, VIX regime map, current buy signal
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


VIX_SYM      = "^INDIAVIX"
NIFTY_SYM    = "NIFTYBEES.NS"
LOOKBACKS    = [15, 30, 45, 55, 65]
TOP_N        = 3
COST_PCT     = 0.001
INITIAL      = 1_000_000

# 10 years of data
FETCH_START  = "2016-01-01"
TRADE_START  = "2016-06-01"   # buffer for longest lookback
END          = date.today().strftime("%Y-%m-%d")
CACHE_DIR    = ".cache/regime"
OUT_PATH     = "docs/index.html"

# VIX regime boundaries (empirically tuned for India VIX)
VIX_REGIMES = [
    (0,   13,  "Bull / Low Vol",    15),   # trending market, short lookback captures momentum
    (13,  17,  "Normal",            30),
    (17,  22,  "Caution",           45),
    (22,  28,  "Elevated Stress",   55),
    (28, 999,  "Crisis / High Vol", 65),   # noisy market, longer smoothing needed
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
                raise ValueError("empty — Yahoo Finance may be blocked on this network")
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
    """
    Fallback: fetch India VIX from NSE website directly.
    NSE provides historical VIX as a downloadable CSV.
    """
    try:
        import requests
        url = ("https://www.nseindia.com/api/historical/vixhistory"
               "?from=01-01-2015&to=" + datetime.now().strftime("%d-%m-%Y"))
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        # Need a session with cookies first
        sess = requests.Session()
        sess.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = sess.get(url, headers=headers, timeout=15)
        data = resp.json()
        records = data.get("data", [])
        if not records:
            return None
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
    """
    Last resort: estimate VIX proxy from Nifty realised volatility.
    Uses 20-day rolling std of Nifty returns * sqrt(252) * 100.
    Not as good as real VIX but preserves regime logic.
    """
    if NIFTY_SYM not in prices.columns:
        return None
    nifty = prices[NIFTY_SYM].dropna()
    log_ret = np.log(nifty / nifty.shift(1))
    rvol = log_ret.rolling(20).std() * np.sqrt(252) * 100
    rvol.name = VIX_SYM
    rvol = rvol.reindex(prices.index).ffill(limit=5)
    print(f"  Synthetic VIX (realised vol proxy): mean={rvol.mean():.1f}, range={rvol.min():.1f}-{rvol.max():.1f}")
    return rvol

def fetch_all():
    print(f"Fetching {len(ETFS)+1} instruments | {FETCH_START} -> {END}")
    series = {}

    # Fetch VIX — try Yahoo, then NSE fallback, then synthetic
    print(f"  [ 0/{len(ETFS)}] India VIX ({VIX_SYM})", end=" ... ", flush=True)
    v = _fetch_one(VIX_SYM)
    if v is not None:
        series[VIX_SYM] = v
        print(f"ok ({v.notna().sum()} rows, {v.index[0].date()} to {v.index[-1].date()})")
    else:
        print("Yahoo blocked — trying NSE direct...")
        v = _fetch_vix_nse_fallback()
        if v is not None:
            series[VIX_SYM] = v
        # synthetic VIX added after prices are loaded (needs Nifty prices)
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
    
    # If VIX completely missing, build synthetic from Nifty realised vol
    if vix.notna().sum() < 100:
        print("  VIX unavailable — building synthetic realised-vol proxy from NiftyBees")
        vix_synth = _make_synthetic_vix(prices)
        if vix_synth is not None:
            vix = vix_synth
            print("  ⚠  Using synthetic VIX (realised vol). Regime accuracy reduced.")
        else:
            # Flat fallback: use constant VIX=17 (Normal regime → 30d lookback)
            vix = pd.Series(17.0, index=prices.index, name=VIX_SYM)
            print("  ⚠  Using flat VIX=17 (Normal regime). No regime adaptation.")
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
    if pd.isna(vix_val) or vix_val <= 0:
        return 45  # default
    for lo, hi, label, lb in VIX_REGIMES:
        if lo <= vix_val < hi:
            return lb
    return 65

def vix_to_regime_label(vix_val):
    if pd.isna(vix_val) or vix_val <= 0:
        return "Unknown"
    for lo, hi, label, lb in VIX_REGIMES:
        if lo <= vix_val < hi:
            return label
    return "Crisis"

# ── SINGLE LOOKBACK BACKTEST (for calibration) ────────────────────────────────
def run_fixed_lb(prices, vix, lb, start, end_date):
    avail = prices.columns.tolist()
    fridays = pd.date_range(start, end_date, freq="W-FRI")
    pairs = []
    for f in fridays:
        si = prices.index.searchsorted(f, side="right") - 1
        ei = si + 1
        if 0 <= si < len(prices) and ei < len(prices):
            pairs.append((si, ei))
    if not pairs: return None

    cash = float(INITIAL); holdings = {}; cur3 = []
    peak = float(INITIAL); eq = []
    nifty_px0 = float(prices[NIFTY_SYM].iloc[pairs[0][1]]) if NIFTY_SYM in prices.columns else None

    for wi,(si,ei) in enumerate(pairs):
        scores, rets = compute_rs(prices, si, avail, lb)
        ranked = sorted([(s,v) for s,v in scores.items() if v is not None], key=lambda x:-x[1])
        new3 = [s for s,_ in ranked[:TOP_N]]
        if len(new3) < TOP_N: continue
        needs = (wi==0) or (set(new3)!=set(cur3))
        exiting = [s for s in cur3 if s not in new3]
        if needs:
            for sym in exiting:
                if sym in holdings and holdings[sym]>0:
                    cash += holdings[sym]*float(prices[sym].iloc[ei])*(1-COST_PCT)
                    del holdings[sym]
            retained = sum(holdings[s]*float(prices[s].iloc[ei]) for s in holdings if s in prices.columns)
            target = (cash+retained)/TOP_N
            for sym in new3:
                px = float(prices[sym].iloc[ei])
                cval = holdings.get(sym,0)*px
                entering = sym not in cur3
                if entering or abs(cval-target)>target*0.05:
                    if sym in holdings and holdings[sym]>0:
                        cash += holdings[sym]*px*(1-COST_PCT)
                    cash -= target
                    holdings[sym] = target*(1-COST_PCT)/px
        port = max(cash + sum(holdings[s]*float(prices[s].iloc[ei]) for s in holdings if s in prices.columns), 0)
        if port > peak: peak = port
        eq.append({"date": str(prices.index[si].date()), "val": port,
                   "dd": (port-peak)/peak*100})
        cur3 = new3

    if len(eq) < 4: return None
    vals = [e["val"] for e in eq]
    wr = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1,len(vals))]
    cagr = (pow(vals[-1]/INITIAL, 52/len(vals)) - 1)*100 if vals else 0
    sharpe = (np.mean(wr)/np.std(wr))*np.sqrt(52) if np.std(wr)>0 else 0
    mdd = min(e["dd"] for e in eq)
    ret = (vals[-1]-INITIAL)/INITIAL*100
    return dict(cagr=round(cagr,2), sharpe=round(sharpe,2), mdd=round(mdd,2),
                ret=round(ret,2), weeks=len(eq), final=round(vals[-1],0), eq=eq)

# ── PER-YEAR CALIBRATION ──────────────────────────────────────────────────────
def calibrate_yearly(prices, vix):
    print("Calibrating: best lookback per year...")
    years = list(range(2015, int(END[:4])+1))
    results = {}

    for yr in years:
        y_start = f"{yr}-01-01"
        y_end   = f"{yr}-12-31"
        # need data available
        yr_prices = prices[(prices.index >= y_start) & (prices.index <= y_end)]
        if len(yr_prices) < 30:
            continue
        yr_vix = vix[(vix.index >= y_start) & (vix.index <= y_end)]
        avg_vix = round(float(yr_vix.mean()), 1) if yr_vix.notna().sum() > 10 else None
        max_vix = round(float(yr_vix.max()), 1) if yr_vix.notna().sum() > 10 else None

        best_lb = None; best_sharpe = -999
        lb_stats = {}
        for lb in LOOKBACKS:
            # need extra buffer before year start
            buf_start = (pd.Timestamp(y_start) - timedelta(days=lb*2)).strftime("%Y-%m-%d")
            r = run_fixed_lb(prices, vix, lb, y_start, y_end)
            if r is None: continue
            lb_stats[lb] = r
            if r["sharpe"] > best_sharpe:
                best_sharpe = r["sharpe"]
                best_lb = lb

        regime = vix_to_regime_label(avg_vix) if avg_vix else "Unknown"
        results[yr] = dict(
            avg_vix=avg_vix, max_vix=max_vix,
            regime=regime, best_lb=best_lb,
            lb_stats=lb_stats
        )
        if best_lb:
            bs = lb_stats[best_lb]
            print(f"  {yr}: VIX avg={avg_vix} max={max_vix} regime={regime:20s} "
                  f"best_lb={best_lb:2d}d CAGR={bs['cagr']:+6.1f}% Sharpe={bs['sharpe']:.2f}")
        else:
            print(f"  {yr}: insufficient data")

    return results

# ── ADAPTIVE BACKTEST ─────────────────────────────────────────────────────────
def run_adaptive(prices, vix, regime_lb_map):
    """
    Each Friday: look up current VIX -> get lookback -> compute RS -> trade.
    regime_lb_map: dict of vix_range -> lookback (from calibration or VIX_REGIMES default)
    """
    print("\nRunning adaptive backtest...")
    avail = prices.columns.tolist()
    fridays = pd.date_range(TRADE_START, END, freq="W-FRI")
    pairs = []
    for f in fridays:
        si = prices.index.searchsorted(f, side="right") - 1
        ei = si + 1
        if 0 <= si < len(prices) and ei < len(prices):
            pairs.append((si, ei))

    cash = float(INITIAL); holdings = {}; cur3 = []
    peak = float(INITIAL); total_trades = 0
    hold_count = {e[0]: 0 for e in ETFS}
    eq=[]; nf=[]; dd_curve=[]; log=[]
    nifty_px0 = float(prices[NIFTY_SYM].iloc[pairs[0][1]]) if pairs and NIFTY_SYM in prices.columns else None
    last_scores = last_rets = None

    for wi,(si,ei) in enumerate(pairs):
        date_str = str(prices.index[si].date())
        exec_str = str(prices.index[ei].date())

        # Get VIX at signal day
        vix_val = float(vix.iloc[si]) if si < len(vix) and not pd.isna(vix.iloc[si]) else None
        lb = vix_to_lookback(vix_val)
        regime = vix_to_regime_label(vix_val)

        scores, rets = compute_rs(prices, si, avail, lb)
        ranked = sorted([(s,v) for s,v in scores.items() if v is not None], key=lambda x:-x[1])
        new3 = [s for s,_ in ranked[:TOP_N]]
        if len(new3) < TOP_N: continue

        needs = (wi==0) or (set(new3)!=set(cur3))
        exiting  = [s for s in cur3 if s not in new3]
        entering = [s for s in new3  if s not in cur3]

        if needs:
            for sym in exiting:
                if sym in holdings and holdings[sym]>0:
                    px = float(prices[sym].iloc[ei])
                    cash += holdings[sym]*px*(1-COST_PCT)
                    total_trades += 1; del holdings[sym]
            retained = sum(holdings[s]*float(prices[s].iloc[ei]) for s in holdings if s in prices.columns)
            target = (cash+retained)/TOP_N
            for sym in new3:
                px = float(prices[sym].iloc[ei])
                cval = holdings.get(sym,0)*px
                if sym in entering or abs(cval-target)>target*0.05:
                    if sym in holdings and holdings[sym]>0:
                        cash += holdings[sym]*px*(1-COST_PCT); total_trades += 1
                    cash -= target
                    holdings[sym] = target*(1-COST_PCT)/px
                    total_trades += 1

        port = max(cash + sum(holdings[s]*float(prices[s].iloc[ei]) for s in holdings if s in prices.columns), 0)
        for sym in new3: hold_count[sym] = hold_count.get(sym,0)+1
        nifty_px = float(prices[NIFTY_SYM].iloc[ei]) if NIFTY_SYM in prices.columns else None
        nifty_v  = (nifty_px/nifty_px0*INITIAL) if nifty_px and nifty_px0 else INITIAL
        if port>peak: peak=port
        ddown = (port-peak)/peak*100

        eq.append({"x":date_str,"y":round(port,2)})
        nf.append({"x":date_str,"y":round(nifty_v,2)})
        dd_curve.append({"x":date_str,"y":round(ddown,3)})
        log.append({"date":date_str,"exec":exec_str,"top3":new3,
                    "exiting":exiting,"entering":entering,
                    "changed":needs and wi>0,"capital":round(port,2),
                    "vix":round(vix_val,1) if vix_val else None,
                    "regime":regime,"lb":lb,
                    "scores":{s:round(v,6) if v else None for s,v in scores.items()},
                    "rets":  {s:round(v,6) if v else None for s,v in rets.items()}})
        cur3=new3; last_scores=scores; last_rets=rets

    return dict(eq=eq,nf=nf,dd=dd_curve,log=log,trades=total_trades,
                hold_count=hold_count,last_scores=last_scores,last_rets=last_rets,
                avail=avail)

# ── STATS ─────────────────────────────────────────────────────────────────────
def calc_stats(eq, nf, dd, trades):
    if not eq: return None
    fv=eq[-1]["y"]; nfv=nf[-1]["y"]
    yrs=max((pd.Timestamp(END)-pd.Timestamp(TRADE_START)).days/365.25,0.1)
    tr=(fv-INITIAL)/INITIAL*100
    cagr=(pow(fv/INITIAL,1/yrs)-1)*100
    nc=(pow(nfv/INITIAL,1/yrs)-1)*100
    wr=[(eq[i]["y"]-eq[i-1]["y"])/eq[i-1]["y"] for i in range(1,len(eq))]
    shp=(np.mean(wr)/np.std(wr))*np.sqrt(52) if wr and np.std(wr)>0 else 0
    mdd=min(d["y"] for d in dd) if dd else 0
    return dict(final=round(fv,2),ret=round(tr,2),cagr=round(cagr,2),
                sharpe=round(shp,2),mdd=round(mdd,2),trades=trades,
                nifty_cagr=round(nc,2),alpha=round(cagr-nc,2),weeks=len(eq))

# ── CURRENT SIGNAL ────────────────────────────────────────────────────────────
def get_current_signal(prices, vix, log):
    """Return current week's buy recommendation."""
    if not log: return None
    last = log[-1]
    em = {e[0]:e for e in ETFS}

    # Current VIX
    cur_vix = float(vix.dropna().iloc[-1]) if vix.notna().sum()>0 else None
    cur_lb   = vix_to_lookback(cur_vix)
    cur_reg  = vix_to_regime_label(cur_vix)

    # Current prices
    cur_prices = {}
    for sym in last["top3"]:
        if sym in prices.columns:
            cur_prices[sym] = round(float(prices[sym].dropna().iloc[-1]), 2)

    return dict(
        signal_date  = last["date"],
        exec_date    = last["exec"],
        top3         = last["top3"],
        top3_names   = [em[s][1] if s in em else s for s in last["top3"]],
        top3_short   = [em[s][2] if s in em else s for s in last["top3"]],
        scores       = {s: round(last["scores"].get(s,0) or 0,4) for s in last["top3"]},
        rets         = {s: round((last["rets"].get(s,0) or 0)*100,2) for s in last["top3"]},
        cur_vix      = round(cur_vix,1) if cur_vix else None,
        cur_regime   = cur_reg,
        cur_lb       = cur_lb,
        cur_prices   = cur_prices,
        portfolio_val= last["capital"],
    )

# ── BUILD HTML ────────────────────────────────────────────────────────────────
def build_html(prices, vix, yearly, adaptive, stats, signal):
    with open("/home/claude/node_modules/chart.js/dist/chart.umd.js") as f:
        chartjs = f.read()
    with open("/home/claude/node_modules/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.js") as f:
        adapter = f.read()

    def p(v,d=1): return f"{'+' if v>0 else ''}{v:.{d}f}%"
    def inr(v):   return f"Rs {v/1e5:.2f}L"
    def cls(v):   return "green" if v>=0 else "red"

    avail = adaptive["avail"]
    em    = {e[0]:e for e in ETFS}
    upd   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Current signal card ───────────────────────────────────────────────
    if signal:
        vix_color = ("#22c55e" if (signal["cur_vix"] or 0)<15 else
                     "#f59e0b" if (signal["cur_vix"] or 0)<22 else "#ef4444")
        sig_html = f"""
<div class="cc signal-card">
  <div class="st">THIS WEEK'S BUY SIGNAL</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="mini-stat">
      <div class="mini-label">India VIX</div>
      <div class="mini-val" style="color:{vix_color}">{signal['cur_vix'] or 'N/A'}</div>
    </div>
    <div class="mini-stat">
      <div class="mini-label">Regime</div>
      <div class="mini-val" style="font-size:1rem">{signal['cur_regime']}</div>
    </div>
    <div class="mini-stat">
      <div class="mini-label">Active Lookback</div>
      <div class="mini-val accent">{signal['cur_lb']}d</div>
    </div>
  </div>
  <div style="margin-bottom:16px">
    <div style="font-size:.75rem;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">BUY / HOLD THESE 3 ETFs</div>
    <div style="display:flex;gap:12px;flex-wrap:wrap">"""
        for sym in signal["top3"]:
            name  = em[sym][1] if sym in em else sym
            short = em[sym][2] if sym in em else sym
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
    Signal date: <b style="color:var(--text)">{signal['signal_date']}</b> (Fri close) &rarr;
    Execute: <b style="color:var(--text)">{signal['exec_date']}</b> (Mon open/close) &middot;
    Portfolio value: <b style="color:var(--accent)">{inr(signal['portfolio_val'])}</b>
  </div>
</div>"""
    else:
        sig_html = '<div class="cc"><p style="color:var(--muted)">No signal available yet.</p></div>'

    # ── VIX regime map table ──────────────────────────────────────────────
    regime_html = """<table><thead><tr>
      <th>VIX Range</th><th>Regime</th><th>Active Lookback</th><th>Rationale</th>
    </tr></thead><tbody>"""
    rationale = {
        15: "Short momentum — trending low-vol bull market",
        30: "Medium — normal market conditions",
        45: "Medium-long — caution, mixed signals",
        55: "Long — elevated stress, noise filter needed",
        65: "Max smoothing — crisis/high-vol regime",
    }
    colors = {15:"#22c55e",30:"#86efac",45:"#f59e0b",55:"#fb923c",65:"#ef4444"}
    for lo,hi,label,lb in VIX_REGIMES:
        hi_str = str(hi) if hi<999 else "+"
        cur = signal and signal["cur_vix"] and lo <= signal["cur_vix"] < hi
        bg = "background:rgba(79,142,247,0.08);" if cur else ""
        tag = ' <span class="bd bg">ACTIVE</span>' if cur else ""
        regime_html += (f'<tr style="{bg}"><td><b>{lo}–{hi_str}</b></td>'
                        f'<td>{label}{tag}</td>'
                        f'<td style="color:{colors[lb]};font-weight:700">{lb}d</td>'
                        f'<td style="color:var(--muted);font-size:.8rem">{rationale[lb]}</td></tr>')
    regime_html += "</tbody></table>"

    # ── Per-year calibration table ────────────────────────────────────────
    yr_html = """<table><thead><tr>
      <th>Year</th><th>Avg VIX</th><th>Max VIX</th><th>Regime</th><th>Best LB</th>
      <th>15d Sharpe</th><th>30d Sharpe</th><th>45d Sharpe</th><th>55d Sharpe</th><th>65d Sharpe</th>
      <th>Best CAGR</th><th>Best Sharpe</th>
    </tr></thead><tbody>"""
    for yr, r in sorted(yearly.items()):
        if not r["lb_stats"]: continue
        best = r["best_lb"]
        bs   = r["lb_stats"].get(best, {})
        row = f'<tr><td><b>{yr}</b></td>'
        row += f'<td>{r["avg_vix"] or "—"}</td>'
        row += f'<td>{r["max_vix"] or "—"}</td>'
        row += f'<td style="font-size:.78rem">{r["regime"]}</td>'
        row += f'<td style="color:#4f8ef7;font-weight:700">{best}d</td>'
        for lb in LOOKBACKS:
            st = r["lb_stats"].get(lb)
            if st:
                sc = st["sharpe"]
                col = "#22c55e" if lb==best else ("#e2e8f0" if sc>0 else "#ef4444")
                fw  = "font-weight:700" if lb==best else ""
                row += f'<td style="color:{col};{fw}">{sc:.2f}</td>'
            else:
                row += '<td style="color:var(--muted)">—</td>'
        row += f'<td class="{cls(bs.get("cagr",0))}">{p(bs.get("cagr",0))}</td>'
        row += f'<td class="{cls(bs.get("sharpe",0)-1)}">{bs.get("sharpe",0):.2f}</td>'
        row += '</tr>'
        yr_html += row
    yr_html += "</tbody></table>"

    # ── VIX chart data ────────────────────────────────────────────────────
    vix_data = []
    for dt, v in vix.dropna().items():
        vix_data.append({"x": str(dt.date()), "y": round(float(v),2)})

    eq_json = json.dumps(adaptive["eq"])
    nf_json = json.dumps(adaptive["nf"])
    dd_json = json.dumps(adaptive["dd"])
    vx_json = json.dumps(vix_data)

    # ── Trade log ─────────────────────────────────────────────────────────
    log_html = ""
    for lg in reversed(adaptive["log"]):
        tags = "".join(f'<span class="tg">{em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["top3"])
        chg  = ""
        if lg["changed"]:
            ex = " ".join(f'<span class="bd br">- {em[s][2] if s in em else s}</span>' for s in lg["exiting"])
            en = " ".join(f'<span class="bd bg">+ {em[s][2] if s in em else s}</span>' for s in lg["entering"])
            chg = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:3px">{ex} {en}</div>'
        nc  = "" if lg["changed"] else '<span style="color:var(--muted);font-size:.72rem;margin-left:5px">no change</span>'
        vx  = f'VIX {lg["vix"]}' if lg["vix"] else ""
        log_html += (f'<div class="lr" data-c="{"1" if lg["changed"] else "0"}">'
                     f'<div class="ld">{lg["date"]}<span style="display:block;font-size:.65rem;color:var(--muted)">{lg["lb"]}d · {vx}</span></div>'
                     f'<div class="lb2"><div style="display:flex;align-items:center;flex-wrap:wrap">{tags}{nc}</div>'
                     f'{chg}<div style="color:var(--muted);font-size:.72rem;margin-top:2px">Rs {lg["capital"]/1000:.1f}K · {lg["regime"]}</div>'
                     f'</div></div>\n')

    # ── Rankings ──────────────────────────────────────────────────────────
    ls = adaptive["last_scores"] or {}; lr = adaptive["last_rets"] or {}
    rows = [(e,ls.get(e[0]),lr.get(e[0])) for e in ETFS if e[0] in avail and ls.get(e[0]) is not None]
    rows.sort(key=lambda x:-x[1])
    medals=["1","2","3"]
    rank_html='<table><thead><tr><th>#</th><th>ETF</th><th>RS Score</th><th>Return</th><th>Signal</th></tr></thead><tbody>'
    for i,(etf,sc,ret) in enumerate(rows):
        top=i<TOP_N; med=medals[i] if i<3 else str(i+1)
        rs=f"{ret*100:+.2f}%" if ret else "-"; rc="green" if (ret or 0)>0 else "red"
        sig2=f'<span class="bd {"bg" if top else "bm"}">{"LONG" if top else "OUT"}</span>'
        bg2='background:rgba(79,142,247,.06);' if top else ''
        rank_html+=(f'<tr style="{bg2}"><td>{med}</td>'
                    f'<td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.7rem">{etf[2]}</span></td>'
                    f'<td style="color:var(--accent)">{sc*100:+.3f}</td>'
                    f'<td class="{rc}">{rs}</td><td>{sig2}</td></tr>')
    rank_html += '</tbody></table>'

    gc = '#2e3250'

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>India ETF RS — Regime Adaptive Backtest</title>
<style>
:root{{--bg:#0f1117;--card:#1a1d27;--card2:#22263a;--border:#2e3250;--accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#8892b0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif}}
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
tr:last-child td{{border-bottom:none}}
.bd{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:600}}
.bg{{background:rgba(34,197,94,.15);color:var(--green)}}
.br{{background:rgba(239,68,68,.15);color:var(--red)}}
.bm{{background:rgba(136,146,176,.1);color:var(--muted)}}
.st{{font-size:.93rem;font-weight:700;margin-bottom:13px;display:flex;align-items:center;gap:8px}}
.st::before{{content:'';display:block;width:4px;height:17px;background:var(--accent);border-radius:2px}}
.mw{{overflow-x:auto}}
.mx{{border-collapse:collapse;font-size:.7rem;white-space:nowrap}}
.mx th,.mx td{{padding:5px 6px;border:1px solid var(--border);text-align:center;min-width:50px}}
.mx th{{background:var(--card2);color:var(--muted)}}
.mx .rl{{font-weight:600;color:var(--text);background:var(--card2);text-align:left;padding-left:8px}}
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
    <div class="meta">{len(avail)} ETFs &middot; VIX-driven lookback (15/30/45/55/65d) &middot; Top-3 Long &middot; Weekly &middot; 10yr history &middot; Rs 10L</div>
  </div>
  <div class="meta">Updated: {upd}</div>
</header>
<div class="container">

{sig_html}

<!-- STATS -->
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

<!-- EQUITY CHART -->
<div class="cc">
  <h3>Equity Curve — Regime-Adaptive vs NiftyBees (10 years)</h3>
  <div class="chartbox" style="height:300px"><canvas id="ec"></canvas></div>
</div>

<!-- VIX CHART -->
<div class="cc">
  <h3>India VIX — Regime Zones</h3>
  <div class="chartbox" style="height:180px"><canvas id="vx"></canvas></div>
</div>

<!-- DRAWDOWN -->
<div class="cc">
  <h3>Drawdown</h3>
  <div class="chartbox" style="height:150px"><canvas id="dc"></canvas></div>
</div>

<!-- VIX REGIME MAP -->
<div class="cc">
  <div class="st">VIX Regime Map — Lookback Selection Rules</div>
  {regime_html}
</div>

<!-- PER-YEAR CALIBRATION -->
<div class="cc">
  <div class="st">Per-Year Calibration — VIX vs Best Lookback (Sharpe-ranked)</div>
  <div class="mw">{yr_html}</div>
</div>

<div class="g2">
  <div class="cc"><div class="st">Current RS Rankings</div>{rank_html}</div>
  <div class="cc">
    <div class="st">Weekly Log</div>
    <div class="tabs">
      <button class="tab active" onclick="sw('all',this)">All</button>
      <button class="tab" onclick="sw('chg',this)">Changes</button>
    </div>
    <div id="log">{log_html}</div>
  </div>
</div>

<div class="upd">GitHub Actions · yfinance · Regime-Adaptive RS · Signal: Fri close → Execute: Mon close</div>
</div>

<script>{chartjs}</script>
<script>{adapter}</script>
<script>
var gc = '{gc}';
var eqD = {eq_json};
var nfD = {nf_json};
var ddD = {dd_json};
var vxD = {vx_json};

new Chart(document.getElementById('ec'), {{
  type:'line', data:{{datasets:[
    {{label:'Adaptive RS', data:eqD, parsing:false, borderColor:'#4f8ef7',
     backgroundColor:'rgba(79,142,247,0.08)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
    {{label:'NiftyBees',   data:nfD, parsing:false, borderColor:'#f59e0b',
     backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, borderDash:[5,4], tension:0.3}}
  ]}},
  options:{{responsive:true, maintainAspectRatio:false, parsing:false,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{labels:{{color:'#8892b0',boxWidth:12}}}},
      tooltip:{{backgroundColor:'#1a1d27',titleColor:'#e2e8f0',bodyColor:'#8892b0',
        callbacks:{{label:function(c){{return c.dataset.label+': Rs '+(c.raw.y/1000).toFixed(1)+'K';}}}}
      }}}},
    scales:{{
      x:{{type:'time',time:{{unit:'month'}},ticks:{{color:'#8892b0',maxTicksLimit:18}},grid:{{color:gc}}}},
      y:{{ticks:{{color:'#8892b0',callback:function(v){{return 'Rs '+(v/1000).toFixed(0)+'K';}}}},grid:{{color:gc}}}}
    }}
  }}
}});

new Chart(document.getElementById('vx'), {{
  type:'line', data:{{datasets:[
    {{label:'India VIX', data:vxD, parsing:false, borderColor:'#a78bfa',
     backgroundColor:'rgba(167,139,250,0.08)', borderWidth:1.5, pointRadius:0, tension:0.2, fill:true}}
  ]}},
  options:{{responsive:true, maintainAspectRatio:false, parsing:false,
    plugins:{{legend:{{labels:{{color:'#8892b0',boxWidth:12}}}},
      annotation:{{annotations:{{
        l1:{{type:'line',yMin:13,yMax:13,borderColor:'#22c55e',borderWidth:1,borderDash:[4,4]}},
        l2:{{type:'line',yMin:17,yMax:17,borderColor:'#f59e0b',borderWidth:1,borderDash:[4,4]}},
        l3:{{type:'line',yMin:22,yMax:22,borderColor:'#fb923c',borderWidth:1,borderDash:[4,4]}},
        l4:{{type:'line',yMin:28,yMax:28,borderColor:'#ef4444',borderWidth:1,borderDash:[4,4]}}
      }}}}
    }},
    scales:{{
      x:{{type:'time',time:{{unit:'month'}},ticks:{{color:'#8892b0',maxTicksLimit:18}},grid:{{color:gc}}}},
      y:{{ticks:{{color:'#8892b0'}},grid:{{color:gc}}}}
    }}
  }}
}});

new Chart(document.getElementById('dc'), {{
  type:'line', data:{{datasets:[
    {{label:'Drawdown', data:ddD, parsing:false, borderColor:'#ef4444',
     backgroundColor:'rgba(239,68,68,0.12)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true}}
  ]}},
  options:{{responsive:true, maintainAspectRatio:false, parsing:false,
    plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1a1d27',
      callbacks:{{label:function(c){{return 'DD: '+c.raw.y.toFixed(2)+'%';}}}}
    }}}},
    scales:{{
      x:{{type:'time',time:{{unit:'month'}},ticks:{{color:'#8892b0',maxTicksLimit:18}},grid:{{color:gc}}}},
      y:{{ticks:{{color:'#8892b0',callback:function(v){{return v.toFixed(1)+'%';}}}},grid:{{color:gc}}}}
    }}
  }}
}});

function sw(t,el){{
  document.querySelectorAll('.tab').forEach(function(x){{x.classList.remove('active');}});
  el.classList.add('active');
  document.querySelectorAll('.lr').forEach(function(r){{
    r.style.display=(t==='all'||r.dataset.c==='1')?'flex':'none';
  }});
}}
</script>
</body></html>"""

# ── MOCK DATA (for testing without internet) ──────────────────────────────────
def make_mock_data():
    """Generates realistic mock ETF + VIX data for testing the full pipeline."""
    print("MOCK MODE: generating synthetic data (no network calls)")
    trade_dates = pd.date_range(FETCH_START, END, freq="B")
    np.random.seed(42)
    mock_etfs = [e[0] for e in ETFS[:12]]  # use first 12
    prices_dict = {}
    for sym in mock_etfs:
        drift = np.random.uniform(0.00008, 0.00035)
        vol   = np.random.uniform(0.010, 0.020)
        start = np.random.uniform(30, 300)
        lr    = np.random.normal(drift, vol, len(trade_dates))
        prices_dict[sym] = start * np.exp(np.cumsum(lr))
    prices = pd.DataFrame(prices_dict, index=trade_dates)
    # Mock VIX: mean-reverting ~16, with crisis spikes
    vix_vals = np.random.normal(16, 3, len(trade_dates))
    for yr, spike in [(2020, 30), (2022, 22), (2016, 18)]:
        mask = trade_dates.year == yr
        vix_vals[mask] += np.random.uniform(0, spike-14, mask.sum())
    vix = pd.Series(np.clip(vix_vals, 9, 65), index=trade_dates, name=VIX_SYM)
    print(f"  Mock: {len(prices)} days, {len(mock_etfs)} ETFs, VIX {vix.min():.0f}-{vix.max():.0f}")
    return prices, vix

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use --mock flag to test without network: python regime_backtest.py --mock
    USE_MOCK = "--mock" in sys.argv

    if USE_MOCK:
        prices, vix = make_mock_data()
    else:
        prices, vix = fetch_all()
    if prices.empty:
        print("ERROR: No data.", file=sys.stderr)
        print("If Yahoo Finance is blocked on your network, try: python regime_backtest.py --mock")
        sys.exit(1)

    # Step 1: per-year calibration
    yearly = calibrate_yearly(prices, vix)

    # Step 2: adaptive backtest
    adaptive = run_adaptive(prices, vix, {})

    # Step 3: stats
    stats = calc_stats(adaptive["eq"], adaptive["nf"], adaptive["dd"], adaptive["trades"])

    # Step 4: current signal
    signal = get_current_signal(prices, vix, adaptive["log"])

    # Step 5: print summary
    print(f"\n{'='*60}")
    print("REGIME-ADAPTIVE BACKTEST RESULTS")
    print(f"{'='*60}")
    if stats:
        for k,v in [("Final",f"Rs {stats['final']:,.0f}"),("Return",f"{stats['ret']:+.1f}%"),
                    ("CAGR",f"{stats['cagr']:+.1f}%"),("Sharpe",f"{stats['sharpe']:.2f}"),
                    ("Max DD",f"{stats['mdd']:.1f}%"),("Alpha",f"{stats['alpha']:+.1f}%")]:
            print(f"  {k:<12}: {v}")

    if signal:
        print(f"\nCURRENT SIGNAL ({signal['signal_date']})")
        print(f"  VIX: {signal['cur_vix']} | Regime: {signal['cur_regime']} | Lookback: {signal['cur_lb']}d")
        print(f"  BUY: {', '.join(signal['top3_names'])}")
        for sym in signal["top3"]:
            name = next((e[1] for e in ETFS if e[0]==sym), sym)
            print(f"    {name}: Rs {signal['cur_prices'].get(sym,'?')} | "
                  f"{signal['rets'].get(sym,0):+.1f}% ({signal['cur_lb']}d)")

    os.makedirs("docs", exist_ok=True)
    html = build_html(prices, vix, yearly, adaptive, stats, signal)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport -> {OUT_PATH}")
