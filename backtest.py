"""
India ETF Pairwise RS Matrix Backtester
========================================
Universe  : 24 NSE ETFs
Signal    : 63-day pairwise RS — each ETF vs every other ETF
Portfolio : Long top-3 by RS score, equally weighted
Rebalance : Weekly (Friday)
Capital   : ₹10,00,000 | Costs: 0.1% per trade
Period    : Jan 2023 → present

ROOT CAUSE OF "trades starting 2026-06-05":
  The cache directory (.cache/yfinance_data/) contains stale CSV files from a
  previous run where FETCH_START was set to a recent date. The cache key
  includes FETCH_START in the filename, so if the old run used
  FETCH_START="2025-06-01", the cache file is:
      GOLDBEES.NS_2025-06-01_2026-06-18.csv   ← only has ~1yr of data
  When the fixed code tries FETCH_START="2022-06-01", the new filename is:
      GOLDBEES.NS_2022-06-01_2026-06-18.csv   ← doesn't exist → re-downloads ✓
  BUT if the user is still running the old unfixed code (FETCH_START="2023-01-01")
  AND TRADE_START="2025-06-18", only 1-2 weeks of trades are generated.

  PERMANENT FIX APPLIED:
  1. FETCH_START = "2022-06-01"  (well before TRADE_START to cover 63-day lookback)
  2. TRADE_START = "2023-01-01"  (actual backtest start)
  3. Cache validation: if cached file's first date > FETCH_START, treat as stale
  4. period_return returns None on bad data → no fake-zero RS scores
  5. Cash accounting fixed: no double-subtract on retained positions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, sys, os, time, random
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ETFS = [
    # Commodities
    ("GOLDBEES.NS",    "GoldBees",       "GOLD"),
    ("SILVERBEES.NS",  "SilverBees",     "SILV"),
    # Broad Market
    ("NIFTYBEES.NS",   "NiftyBees",      "NFTY"),
    ("JUNIORBEES.NS",  "JuniorBees",     "JNBR"),
    ("MID150BEES.NS",  "Midcap150Bees",  "MIDM"),
    ("NIF100BEES.NS",  "Nifty100Bees",   "NF10"),
    # Sectoral
    ("BANKBEES.NS",    "BankBees",       "BANK"),
    ("ITBEES.NS",      "ITBees",         "ITMC"),
    ("PHARMABEES.NS",  "PharmaBees",     "PHRM"),
    ("AUTOBEES.NS",    "AutoBees",       "AUTO"),
    ("INFRABEES.NS",   "InfraBees",      "INFR"),
    ("CONSUMBEES.NS",  "ConsumeBees",    "CNSM"),
    # PSU / Theme
    ("PSUBNKBEES.NS",  "PSUBankBees",    "PSUB"),
    ("CPSEETF.NS",     "CPSE ETF",       "CETF"),
    # Debt / Fixed Income
    ("LTGILTBEES.NS",  "LT Gilt Bees",   "GSCP"),
    ("GILT5YBEES.NS",  "GSec 5Y Bees",   "GS5Y"),
    ("LIQUIDBEES.NS",  "LiquidBees",     "LIQD"),
    # Smart Beta / Factor / International
    ("MOM100.NS",      "Momentum100",    "MOM" ),
    ("MOMENTUM30.NS",  "Momentum30",     "MOM3"),
    ("NV20BEES.NS",    "Value20Bees",    "NV20"),
    ("DIVOPPBEES.NS",  "DivOppBees",     "DIVO"),
    ("HNGSNGBEES.NS",  "HangSengBees",   "HNGS"),
    ("MAFANG.NS",      "FANGPlus ETF",   "FANG"),
    ("MON100.NS",      "Nasdaq100 ETF",  "NSDQ"),
]

LOOKBACK    = 63            # trading days (~3 calendar months)
TOP_N       = 3
COST_PCT    = 0.001         # 0.1% per trade
INITIAL     = 1_000_000     # ₹10 lakh

# ── DATE CONFIG ───────────────────────────────────────────────────────────────
# FETCH_START must be at least LOOKBACK trading days (~100 calendar days)
# BEFORE TRADE_START so the very first rebalance has a full 63-day return window.
# Using "2022-06-01" gives ~7 months of buffer before "2023-01-01" → plenty.
TRADE_START = "2023-01-01"   # ← backtest begins here
FETCH_START = "2022-06-01"   # ← data fetch begins here (lookback buffer)
END         = "2025-12-31"    #date.today().strftime("%Y-%m-%d")

NIFTY_SYM   = "NIFTYBEES.NS"
OUT_PATH    = "docs/index.html"
CACHE_DIR   = ".cache/yfinance_data"

# ─── DATA FETCH WITH STALE-CACHE PROTECTION ──────────────────────────────────
def _cache_path(sym: str) -> str:
    """Cache filename includes FETCH_START so old caches are never reused."""
    return os.path.join(CACHE_DIR, f"{sym}_{FETCH_START}_{END}.csv")

def _load_cache(sym: str):
    """
    Load cached CSV. Returns None if:
      - file doesn't exist
      - file is empty or unreadable
      - first date in file is later than FETCH_START (stale cache from old run)
    """
    path = _cache_path(sym)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        s = df.iloc[:, 0].copy()
        s.name = sym
        # Stale-cache check: first date must be within 10 days of FETCH_START
        first_date  = s.index[0]
        expected    = pd.Timestamp(FETCH_START)
        if first_date > expected + timedelta(days=10):
            print(f"    ⚠  Stale cache (starts {first_date.date()} vs expected {FETCH_START}) → re-downloading")
            os.remove(path)
            return None
        return s
    except Exception:
        return None

def _fetch_one(sym: str, retries: int = 4, delay: float = 4.0):
    """Download single ticker with retries. Returns tz-naive pd.Series or None."""
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

            # Handle MultiIndex (yfinance >= 0.2.x batch download format)
            if isinstance(df.columns, pd.MultiIndex):
                s = df["Close"][sym].copy()
            else:
                s = df["Close"].copy()

            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.name = sym

            if s.notna().sum() < 50:
                raise ValueError(f"only {s.notna().sum()} non-NaN rows")

            # Save to cache
            os.makedirs(CACHE_DIR, exist_ok=True)
            s.to_csv(_cache_path(sym))
            return s

        except Exception as e:
            if attempt == retries:
                print(f"    ❌ all {retries} attempts failed: {e}")
            time.sleep(delay * attempt + random.uniform(0.5, 2.0))
    return None

def fetch_prices() -> pd.DataFrame:
    print(f"Fetching {len(ETFS)} ETFs  |  data: {FETCH_START} → {END}  |  trades: {TRADE_START} → {END}")
    print(f"Lookback buffer: {(pd.Timestamp(TRADE_START) - pd.Timestamp(FETCH_START)).days} calendar days before TRADE_START\n")

    series_list = []
    for i, (sym, name, short) in enumerate(ETFS, 1):
        print(f"  [{i:>2}/{len(ETFS)}] {name:<18} ({sym})", end=" … ", flush=True)

        s = _load_cache(sym)
        if s is not None:
            print(f"✓ cache  ({s.notna().sum()} rows, from {s.index[0].date()})")
        else:
            s = _fetch_one(sym)
            if s is not None and s.notna().sum() > 50:
                print(f"✓ downloaded  ({s.notna().sum()} rows, from {s.index[0].date()})")
            else:
                print("✗ skipped")
                s = None
            time.sleep(random.uniform(1.0, 2.0))

        if s is not None:
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.ffill(limit=5)   # fill weekends/holidays (max 5 days)

    valid   = [c for c in prices.columns if prices[c].notna().sum() > 100]
    dropped = [e[0] for e in ETFS if e[0] not in valid]
    if dropped:
        print(f"\n  ⚠  Dropped (insufficient history): {[s.replace('.NS','') for s in dropped]}")

    prices = prices[valid].dropna(how="all")

    # Sanity check: verify data starts before TRADE_START
    data_start = prices.index[0]
    trade_ts   = pd.Timestamp(TRADE_START)
    min_needed = trade_ts - timedelta(days=int(LOOKBACK * 1.5))
    if data_start > min_needed:
        print(f"\n  ⚠  WARNING: Data starts {data_start.date()} but need data from ~{min_needed.date()}")
        print(f"     First {LOOKBACK} rebalances may have incomplete RS scores")
    else:
        print(f"\n  ✓  Data covers {data_start.date()} → {prices.index[-1].date()}")

    print(f"  ✓  {len(prices)} trading days · {len(valid)} instruments ready\n")
    return prices

# ─── RS ENGINE ───────────────────────────────────────────────────────────────
def period_return(series: pd.Series, idx_now: int, lookback_days: int):
    """
    Simple return over exactly `lookback_days` trading-day bars.
    Returns None if either price is NaN/zero or if there's not enough history.
    None callers MUST handle — never substitute 0.0 (corrupts RS ranking).
    """
    idx_past = idx_now - lookback_days
    if idx_past < 0:
        return None           # not enough history before this point
    p_now  = series.iloc[idx_now]
    p_past = series.iloc[idx_past]
    if pd.isna(p_now) or pd.isna(p_past) or p_past == 0:
        return None
    return float((p_now / p_past) - 1.0)

def compute_rs(prices: pd.DataFrame, idx: int, available_syms: list):
    """
    Pairwise RS score:
        score(A) = mean over all valid B≠A of [ret_A(63d) − ret_B(63d)]

    Positive = A outperformed the average peer.
    Instruments with None return are excluded from ranking AND peer comparison.
    """
    rets: dict = {}
    for sym in available_syms:
        if sym in prices.columns:
            rets[sym] = period_return(prices[sym], idx, LOOKBACK)

    valid_syms = [s for s in rets if rets[s] is not None]

    scores: dict = {}
    for sym in available_syms:
        if rets.get(sym) is None:
            scores[sym] = None
            continue
        peers = [rets[s] for s in valid_syms if s != sym]
        scores[sym] = float(np.mean([rets[sym] - p for p in peers])) if peers else 0.0

    return scores, rets

def build_matrix(rets: dict, available_syms: list):
    """N×N matrix: matrix[i][j] = ret_i − ret_j in % (row outperforms column)."""
    matrix = []
    for si, _, _ in ETFS:
        row = []
        for sj, _, _ in ETFS:
            if si == sj:
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

    # ── Signal/Execution separation (NO look-ahead bias) ─────────────────────
    # Signal  : Friday close  → compute RS, decide new top-3
    # Execution: Next trading day close (Monday) → trade at that price
    # This mirrors real trading: you can't buy at the same close you screened on.
    all_fridays = pd.date_range(TRADE_START, END, freq="W-FRI")
    signal_exec_pairs = []   # list of (signal_idx, exec_idx)
    for f in all_fridays:
        sig_pos = prices.index.searchsorted(f, side="right") - 1
        if sig_pos < 0 or sig_pos >= len(prices):
            continue
        # Next trading day after Friday = execution day
        exec_pos = sig_pos + 1
        if exec_pos >= len(prices):
            continue       # no next day yet (last week of data)
        signal_exec_pairs.append((sig_pos, exec_pos))

    if not signal_exec_pairs:
        print(f"ERROR: No valid signal/execution pairs found.")
        return [], [], [], [], 0, {}, {}, None, None, None

    print(f"Backtest: {len(signal_exec_pairs)} weeks  |  "
          f"signals {prices.index[signal_exec_pairs[0][0]].date()} → "
          f"{prices.index[signal_exec_pairs[-1][0]].date()}  |  "
          f"execution next trading day (Mon close)\n")

    cash         = float(INITIAL)
    holdings     = {}           # sym → shares
    cur_top3     = []
    peak         = float(INITIAL)
    total_trades = 0
    hold_count   = {e[0]: 0 for e in ETFS}
    instr_trades = {e[0]: 0 for e in ETFS}

    equity_curve, nifty_curve, dd_curve, trade_log = [], [], [], []

    # Nifty baseline at first execution day (Monday after first signal Friday)
    first_exec_pos  = signal_exec_pairs[0][1]
    nifty_start_px  = float(prices[NIFTY_SYM].iloc[first_exec_pos]) if NIFTY_SYM in prices.columns else None

    last_scores = last_rets = last_matrix = None

    for wi, (sig_idx, exec_idx) in enumerate(signal_exec_pairs):
        # sig_idx  = Friday close → used for RS computation (signal)
        # exec_idx = Monday close → used for trade execution (price)
        date_str = str(prices.index[sig_idx].date())   # label by signal date
        exec_date = str(prices.index[exec_idx].date()) # actual trade date
        scores, rets = compute_rs(prices, sig_idx, available_syms)

        # Rank: descending by score, skip None
        ranked   = sorted([(s, v) for s, v in scores.items() if v is not None], key=lambda x: -x[1])
        new_top3 = [s for s, _ in ranked[:TOP_N]]

        if len(new_top3) < TOP_N:
            print(f"  ⚠  {date_str}: only {len(new_top3)} scoreable instruments — skip")
            continue

        needs_rebal = (wi == 0) or (set(new_top3) != set(cur_top3))
        exiting     = [s for s in cur_top3 if s not in new_top3]
        entering    = [s for s in new_top3  if s not in cur_top3]

        if needs_rebal:
            # Step 1: liquidate exiting positions
            for sym in exiting:
                if sym in holdings and holdings[sym] > 0:
                    px    = float(prices[sym].iloc[exec_idx])   # Monday execution price
                    cash += holdings[sym] * px * (1 - COST_PCT)
                    instr_trades[sym] += 1
                    total_trades      += 1
                    del holdings[sym]

            # Step 2: measure retained position values
            retained_val = sum(
                holdings[s] * float(prices[s].iloc[exec_idx])   # Monday prices
                for s in holdings if s in prices.columns
            )

            # Step 3: total capital & target per slot
            total_capital  = cash + retained_val
            target_per_pos = total_capital / TOP_N

            # Step 4: rebalance each top-3 slot
            for sym in new_top3:
                px          = float(prices[sym].iloc[exec_idx])   # Monday execution price
                current_val = holdings.get(sym, 0) * px
                drift       = abs(current_val - target_per_pos)

                # Only trade if: new entry OR drifted >5% from target
                if sym in entering or drift > target_per_pos * 0.05:
                    # Sell existing lot back to cash (if any)
                    if sym in holdings and holdings[sym] > 0:
                        cash += holdings[sym] * px * (1 - COST_PCT)
                        instr_trades[sym] += 1
                        total_trades      += 1

                    # Buy fresh lot at Monday execution price
                    cash          -= target_per_pos
                    holdings[sym]  = (target_per_pos * (1 - COST_PCT)) / px  # px already exec_idx
                    instr_trades[sym] += 1
                    total_trades      += 1

        # Mark-to-market at execution price (Monday close)
        port_val = max(
            cash + sum(holdings[s] * float(prices[s].iloc[exec_idx])
                       for s in holdings if s in prices.columns),
            0
        )

        for sym in new_top3:
            hold_count[sym] = hold_count.get(sym, 0) + 1

        nifty_px  = float(prices[NIFTY_SYM].iloc[exec_idx]) if NIFTY_SYM in prices.columns else None
        nifty_val = (nifty_px / nifty_start_px * INITIAL) if nifty_px and nifty_start_px else INITIAL

        if port_val > peak:
            peak = port_val
        dd = (port_val - peak) / peak * 100

        equity_curve.append({"date": date_str, "value": round(port_val, 2)})
        nifty_curve.append( {"date": date_str, "value": round(nifty_val, 2)})
        dd_curve.append(    {"date": date_str, "dd":    round(dd, 3)})
        trade_log.append({
            "date":      date_str,    # Friday signal date
            "exec_date": exec_date,   # Monday execution date
            "top3":      new_top3,
            "exiting":   exiting,
            "entering":  entering,
            "changed":   needs_rebal and wi > 0,
            "capital":   round(port_val, 2),
            "scores":    {s: round(v, 6) if v is not None else None for s, v in scores.items()},
            "rets":      {s: round(v, 6) if v is not None else None for s, v in rets.items()},
        })

        cur_top3    = new_top3
        last_scores = scores
        last_rets   = rets
        last_matrix = build_matrix(rets, available_syms)

        if wi % 10 == 0:
            top3_s = [s.replace(".NS","") for s in new_top3]
            print(f"  {date_str}  ₹{port_val:>10,.0f}  top3={top3_s}")

    return (equity_curve, nifty_curve, dd_curve, trade_log,
            total_trades, hold_count, instr_trades,
            last_scores, last_rets, last_matrix)

# ─── STATS ───────────────────────────────────────────────────────────────────
def calc_stats(equity_curve, nifty_curve, dd_curve, total_trades):
    final_val   = equity_curve[-1]["value"]
    nifty_final = nifty_curve[-1]["value"]
    years = max((pd.Timestamp(END) - pd.Timestamp(TRADE_START)).days / 365.25, 0.05)

    total_ret  = (final_val - INITIAL) / INITIAL * 100
    cagr       = (pow(final_val / INITIAL,   1/years) - 1) * 100
    nifty_cagr = (pow(nifty_final / INITIAL, 1/years) - 1) * 100

    wr     = [(equity_curve[i]["value"] - equity_curve[i-1]["value"]) / equity_curve[i-1]["value"]
              for i in range(1, len(equity_curve))]
    sharpe = (np.mean(wr) / np.std(wr)) * np.sqrt(52) if wr and np.std(wr) > 0 else 0
    max_dd = min(d["dd"] for d in dd_curve) if dd_curve else 0

    return dict(
        final_val=round(final_val,2), total_ret=round(total_ret,2),
        cagr=round(cagr,2), sharpe=round(sharpe,2), max_dd=round(max_dd,2),
        total_trades=total_trades, nifty_cagr=round(nifty_cagr,2),
        alpha=round(cagr-nifty_cagr,2), weeks=len(equity_curve),
    )

# ─── HTML ────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>India ETF RS Backtest</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--card2:#22263a;--border:#2e3250;--accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#8892b0}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--border);padding:16px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
header h1{font-size:1.2rem;font-weight:700;color:var(--accent)}
.meta{color:var(--muted);font-size:.82rem}
.container{max-width:1400px;margin:0 auto;padding:22px}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:13px;margin-bottom:22px}
.sc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:15px 17px}
.sc .lb{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.sc .vl{font-size:1.4rem;font-weight:700}
.green{color:var(--green)}.red{color:var(--red)}.accent{color:var(--accent)}
.cc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:20px}
.cc h3{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:13px}
.cw{position:relative}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
@media(max-width:780px){.g2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{background:var(--card2);color:var(--muted);font-weight:600;text-transform:uppercase;font-size:.68rem;letter-spacing:.05em;padding:8px 11px;text-align:left;border-bottom:1px solid var(--border)}
td{padding:7px 11px;border-bottom:1px solid var(--border)}
tr:last-child td{border-bottom:none}
.bd{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.71rem;font-weight:600}
.bg{background:rgba(34,197,94,.15);color:var(--green)}
.br{background:rgba(239,68,68,.15);color:var(--red)}
.bm{background:rgba(136,146,176,.1);color:var(--muted)}
.st{font-size:.93rem;font-weight:700;margin-bottom:13px;display:flex;align-items:center;gap:8px}
.st::before{content:'';display:block;width:4px;height:17px;background:var(--accent);border-radius:2px}
.mw{overflow-x:auto}
.mx{border-collapse:collapse;font-size:.7rem;white-space:nowrap}
.mx th,.mx td{padding:5px 6px;border:1px solid var(--border);text-align:center;min-width:52px}
.mx th{background:var(--card2);color:var(--muted);font-weight:600}
.mx .rl{font-weight:600;color:var(--text);background:var(--card2);text-align:left;padding-left:9px;min-width:65px}
.tg{display:inline-flex;align-items:center;background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.3);color:var(--accent);border-radius:5px;padding:2px 8px;font-size:.77rem;font-weight:600;margin:2px}
.tabs{display:flex;gap:4px;background:var(--card2);padding:4px;border-radius:8px;width:fit-content;margin-bottom:14px}
.tab{padding:5px 15px;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:500;color:var(--muted);transition:all .15s}
.tab.active{background:var(--accent);color:#fff}
#log{max-height:400px;overflow-y:auto}
.lr{display:flex;gap:11px;padding:7px 0;border-bottom:1px solid var(--border);font-size:.8rem;align-items:flex-start}
.ld{color:var(--muted);min-width:90px;padding-top:2px}
.lb2{flex:1}
.upd{text-align:right;color:var(--muted);font-size:.76rem;padding:8px 0 0}
</style></head><body>
<header>
  <div>
    <h1>🇮🇳 India ETF — Pairwise RS Matrix Backtest</h1>
    <div class="meta">__N__ ETFs · 63-day pairwise RS · Weekly rebalance · Top-3 Long · ₹10L · Jan 2023 → present</div>
  </div>
  <div class="meta">Updated: __UPD__</div>
</header>
<div class="container">
<div class="sg">
  <div class="sc"><div class="lb">Final Portfolio</div><div class="vl accent">__FINAL__</div></div>
  <div class="sc"><div class="lb">Total Return</div><div class="vl __RC__">__RET__</div></div>
  <div class="sc"><div class="lb">CAGR</div><div class="vl __CC__">__CAGR__</div></div>
  <div class="sc"><div class="lb">Sharpe Ratio</div><div class="vl __SC__">__SHP__</div></div>
  <div class="sc"><div class="lb">Max Drawdown</div><div class="vl red">__MDD__</div></div>
  <div class="sc"><div class="lb">Total Trades</div><div class="vl">__TRD__</div></div>
  <div class="sc"><div class="lb">Nifty CAGR</div><div class="vl">__NFC__</div></div>
  <div class="sc"><div class="lb">Alpha vs Nifty</div><div class="vl __AC__">__ALF__</div></div>
</div>
<div class="cc"><h3>📈 Equity Curve — Strategy vs NiftyBees</h3><div class="cw" style="height:300px"><canvas id="ec"></canvas></div></div>
<div class="cc"><h3>📉 Drawdown</h3><div class="cw" style="height:155px"><canvas id="dc"></canvas></div></div>
<div class="g2">
  <div class="cc"><div class="st">Current RS Rankings</div>__RANK__</div>
  <div class="cc"><div class="st">Time in Portfolio</div>__CONTRIB__</div>
</div>
<div class="cc"><div class="st">Pairwise RS Matrix — last rebalance (row outperforms column, 63-day %)</div><div class="mw">__MATRIX__</div></div>
<div class="cc">
  <div class="st">Weekly Rebalance Log</div>
  <div class="tabs"><div class="tab active" onclick="sw('all',this)">All Weeks</div><div class="tab" onclick="sw('chg',this)">Changes Only</div></div>
  <div id="log">__LOG__</div>
</div>
<div class="upd">GitHub Actions · yfinance · Pairwise RS Strategy · data from __FSTART__ · trades from __TSTART__</div>
</div>
<script>
const toTs = rows => rows.map(r=>({x:new Date(r.x).getTime(),y:r.y}));

const ed = __EQ__.map(r => ({ x: new Date(r.date).getTime(), y: r.value }));
const nd = __NF__.map(r => ({ x: new Date(r.date).getTime(), y: r.value }));
const dd = __DD__.map(r => ({ x: new Date(r.date).getTime(), y: r.dd }));
const gc='#2e3250', tt={backgroundColor:'#1a1d27',titleColor:'#e2e8f0',bodyColor:'#8892b0'};
new Chart(document.getElementById('ec'),{type:'line',data:{datasets:[
  {label:'RS Strategy',data:ed,borderColor:'#4f8ef7',backgroundColor:'rgba(79,142,247,.08)',borderWidth:2,pointRadius:0,tension:.3,fill:true},
  {label:'NiftyBees', data:nd,borderColor:'#f59e0b',backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,tension:.3,borderDash:[5,4]}
]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#8892b0'}},tooltip:{...tt,callbacks:{label:c=>`${c.dataset.label}: ₹${(c.raw.y/1000).toFixed(1)}K`}}},
  scales:{x:{type:'time',time:{unit:'month'},ticks:{color:'#8892b0',maxTicksLimit:20},grid:{color:gc}},
          y:{ticks:{color:'#8892b0',callback:v=>'₹'+(v/1000).toFixed(0)+'K'},grid:{color:gc}}}}});
new Chart(document.getElementById('dc'),{type:'line',data:{datasets:[
  {label:'DD',data:dd,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,.12)',borderWidth:1.5,pointRadius:0,tension:.3,fill:true}
]},options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{...tt,callbacks:{label:c=>`DD: ${c.raw.y.toFixed(2)}%`}}},
  scales:{x:{type:'time',time:{unit:'month'},ticks:{color:'#8892b0',maxTicksLimit:20},grid:{color:gc}},
          y:{ticks:{color:'#8892b0',callback:v=>v.toFixed(1)+'%'},grid:{color:gc}}}}});
function sw(t,el){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');
  document.querySelectorAll('.lr').forEach(r=>{r.style.display=(t==='all'||r.dataset.c==='1')?'flex':'none';});}
</script></body></html>"""

def _p(v,d=1): return f"{'+' if v>0 else ''}{v:.{d}f}%"
def _inr(v):   return f"₹{v/1e5:.2f}L"
def _cls(v):   return "green" if v>=0 else "red"

def rankings_html(scores, rets, avail):
    rows = [(e, scores.get(e[0]), rets.get(e[0])) for e in ETFS
            if e[0] in avail and scores.get(e[0]) is not None]
    rows.sort(key=lambda x: -x[1])
    medals = ["🥇","🥈","🥉"]
    h = '<table><thead><tr><th>Rank</th><th>ETF</th><th>RS Score</th><th>63d Ret</th><th>Signal</th></tr></thead><tbody>'
    for i,(etf,sc,ret) in enumerate(rows):
        top = i<TOP_N
        r_s = f"{ret*100:+.2f}%" if ret is not None else "—"
        h += (f'<tr style="{"background:rgba(79,142,247,.06);" if top else ""}">'
              f'<td>{"medals[i]" if i<3 else str(i+1)}</td>'
              f'<td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.71rem">{etf[2]}</span></td>'
              f'<td style="color:var(--accent)">{sc*100:+.3f}</td>'
              f'<td class="{"green" if (ret or 0)>0 else "red"}">{r_s}</td>'
              f'<td><span class="bd {"bg" if top else "bm"}">{"▲ LONG" if top else "— OUT"}</span></td></tr>')
    # fix medals eval
    h = h  # medals are embedded via f-string below properly
    return h + '</tbody></table>'

def rankings_html(scores, rets, avail):
    rows = [(e, scores.get(e[0]), rets.get(e[0])) for e in ETFS
            if e[0] in avail and scores.get(e[0]) is not None]
    rows.sort(key=lambda x: -x[1])
    medals = ["🥇","🥈","🥉"]
    h = '<table><thead><tr><th>Rank</th><th>ETF</th><th>RS Score</th><th>63d Ret</th><th>Signal</th></tr></thead><tbody>'
    for i,(etf,sc,ret) in enumerate(rows):
        top = i < TOP_N
        med = medals[i] if i < 3 else str(i+1)
        r_s = f"{ret*100:+.2f}%" if ret is not None else "—"
        rc  = "green" if (ret or 0) > 0 else "red"
        sig = f'<span class="bd {"bg" if top else "bm"}">{"▲ LONG" if top else "— OUT"}</span>'
        bg  = 'background:rgba(79,142,247,.06);' if top else ''
        h  += f'<tr style="{bg}"><td>{med}</td><td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.71rem">{etf[2]}</span></td><td style="color:var(--accent)">{sc*100:+.3f}</td><td class="{rc}">{r_s}</td><td>{sig}</td></tr>'
    return h + '</tbody></table>'

def contrib_html(hold_count, instr_trades, total_weeks):
    rows = sorted(ETFS, key=lambda e: -hold_count.get(e[0],0))
    h = '<table><thead><tr><th>ETF</th><th>Weeks</th><th>% Time</th><th>Trades</th></tr></thead><tbody>'
    for etf in rows:
        wk = hold_count.get(etf[0],0)
        pt = round(wk/total_weeks*100) if total_weeks else 0
        bar= f'<div style="height:5px;width:{pt}px;max-width:80px;background:var(--accent);border-radius:3px;min-width:2px;display:inline-block"></div>'
        h += f'<tr><td><b>{etf[1]}</b></td><td>{wk}</td><td>{bar} {pt}%</td><td style="color:var(--muted)">{instr_trades.get(etf[0],0)}</td></tr>'
    return h + '</tbody></table>'

def matrix_html(matrix, rets, avail):
    vis = [e for e in ETFS if e[0] in avail]
    h = '<table class="mx"><thead><tr><th>↓ vs →</th>'
    for e in vis: h += f'<th>{e[2]}</th>'
    h += '<th style="background:rgba(79,142,247,.1);color:var(--accent)">Wins</th></tr></thead><tbody>'
    for i,ei in enumerate(ETFS):
        if ei[0] not in avail: continue
        row  = matrix[i]
        wins = sum(1 for v in row if v is not None and v>0)
        tot  = sum(1 for v in row if v is not None and v!=0)
        h   += f'<tr><td class="rl">{ei[2]}</td>'
        for j,ej in enumerate(ETFS):
            if ej[0] not in avail: continue
            v = row[j]
            if i==j:   h += '<td style="background:var(--card2);color:var(--muted)">—</td>'
            elif v is None: h += '<td style="color:var(--muted)">N/A</td>'
            else:
                inten = min(abs(v)/8, 0.8)
                bg  = f'rgba(34,197,94,{inten:.2f})' if v>0 else f'rgba(239,68,68,{inten:.2f})'
                clr = '#22c55e' if v>0 else '#ef4444'
                h  += f'<td style="background:{bg};color:{clr}">{v:+.1f}%</td>'
        h += f'<td style="background:rgba(79,142,247,.1);color:var(--accent);font-weight:700">{wins}/{tot}</td></tr>'
    return h + '</tbody></table>'

def log_html(trade_log):
    em = {e[0]:e for e in ETFS}
    h  = ''
    for lg in reversed(trade_log):
        tags = ''.join(f'<span class="tg">{em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["top3"])
        chg  = ''
        if lg["changed"]:
            ex = ' '.join(f'<span class="bd br">− {em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["exiting"])
            en = ' '.join(f'<span class="bd bg">+ {em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["entering"])
            chg = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:3px">{ex} {en}</div>'
        nc  = '' if lg["changed"] else '<span style="color:var(--muted);font-size:.72rem;margin-left:5px">no change</span>'
        c   = '1' if lg["changed"] else '0'
        exec_d = lg.get("exec_date", "")
        exec_label = f'<span style="color:var(--muted);font-size:.68rem;display:block">exec {exec_d}</span>' if exec_d else ""
        h  += (f'<div class="lr" data-c="{c}"><div class="ld">{lg["date"]}{exec_label}</div>'
               f'<div class="lb2"><div style="display:flex;align-items:center;flex-wrap:wrap">{tags}{nc}</div>'
               f'{chg}<div style="color:var(--muted);font-size:.72rem;margin-top:2px">₹{lg["capital"]/1000:.1f}K</div>'
               f'</div></div>\n')
    return h

def render(stats, eq, nf, dd, log, hc, it, ls, lr, lm, avail):
    
    ej = json.dumps([{"date": d["date"], "value": d["value"]} for d in eq])
    nj = json.dumps([{"date": d["date"], "value": d["value"]} for d in nf])
    dj = json.dumps([{"date": d["date"], "dd": d["dd"]} for d in dd])
    nj = json.dumps([{"x":d["date"],"y":d["value"]} for d in nf])
    dj = json.dumps([{"x":d["date"],"y":d["dd"]}    for d in dd])
    s  = stats
    r  = {
        "__N__":    str(len(avail)),
        "__UPD__":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "__FSTART__": FETCH_START,
        "__TSTART__": TRADE_START,
        "__FINAL__":_inr(s["final_val"]),
        "__RET__":  _p(s["total_ret"]),  "__RC__": _cls(s["total_ret"]),
        "__CAGR__": _p(s["cagr"]),       "__CC__": _cls(s["cagr"]),
        "__SHP__":  f"{s['sharpe']:.2f}","__SC__": _cls(s["sharpe"]-1),
        "__MDD__":  _p(s["max_dd"]),
        "__TRD__":  str(s["total_trades"]),
        "__NFC__":  _p(s["nifty_cagr"]),
        "__ALF__":  _p(s["alpha"]),      "__AC__": _cls(s["alpha"]),
        "__EQ__":   ej, "__NF__": nj, "__DD__": dj,
        "__RANK__":    rankings_html(ls, lr, avail),
        "__CONTRIB__": contrib_html(hc, it, s["weeks"]),
        "__MATRIX__":  matrix_html(lm, lr, avail),
        "__LOG__":     log_html(log),
    }
    h = HTML
    for k,v in r.items(): h = h.replace(k,v)
    return h

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices = fetch_prices()
    if prices.empty:
        print("ERROR: No price data fetched.", file=sys.stderr); sys.exit(1)

    avail = prices.columns.tolist()
    print(f"Instruments: {[s.replace('.NS','') for s in avail]}\n")

    (eq,nf,dd,log,trades,hc,it,ls,lr,lm) = run_backtest(prices)

    if not eq:
        print("ERROR: Backtest produced no results.", file=sys.stderr); sys.exit(1)

    stats = calc_stats(eq, nf, dd, trades)
    print(f"\n{'='*52}")
    for k,v in [("Final Portfolio",f"₹{stats['final_val']:,.0f}"),
                ("Total Return",   f"{stats['total_ret']:+.1f}%"),
                ("CAGR",           f"{stats['cagr']:+.1f}%"),
                ("Sharpe",         f"{stats['sharpe']:.2f}"),
                ("Max Drawdown",   f"{stats['max_dd']:.1f}%"),
                ("Total Trades",   stats['total_trades']),
                ("Nifty CAGR",     f"{stats['nifty_cagr']:+.1f}%"),
                ("Alpha",          f"{stats['alpha']:+.1f}%"),
                ("Weeks",          stats['weeks'])]:
        print(f"  {k:<18}: {v}")
    print(f"{'='*52}\n")

    os.makedirs("docs", exist_ok=True)
    with open(OUT_PATH,"w",encoding="utf-8") as f:
        f.write(render(stats,eq,nf,dd,log,hc,it,ls,lr,lm,avail))
    print(f"✅  Report → {OUT_PATH}")
