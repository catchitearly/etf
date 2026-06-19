"""
India ETF Pairwise RS Matrix Backtester — Multi-Lookback Edition
=================================================================
Universe  : 24 NSE ETFs
Signal    : Pairwise RS across lookback periods: 15, 30, 45, 55, 65 days
Portfolio : Long top-3 by RS score, equally weighted
Rebalance : Weekly — Signal on Friday close, Execute on Monday close
Capital   : Rs 10,00,000 | Costs: 0.1% per trade
Period    : Jan 2023 to present
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, sys, os, time, random
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

LOOKBACKS   = [15, 30, 45, 55, 65]   # all periods to test
TOP_N       = 3
COST_PCT    = 0.001
INITIAL     = 1_000_000
TRADE_START = "2017-01-01"
FETCH_START = "2026-06-01"           # buffer for longest lookback
END         = "2026-06-19" # date.today().strftime("%Y-%m-%d")
NIFTY_SYM   = "NIFTYBEES.NS"
OUT_PATH    = "docs/index.html"
CACHE_DIR   = ".cache/yf"

# ── FETCH ─────────────────────────────────────────────────────────────────────
def _cache_path(sym):
    return os.path.join(CACHE_DIR, f"{sym}_{FETCH_START}_{END}.csv")

def _load_cache(sym):
    p = _cache_path(sym)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty:
            return None
        s = df.iloc[:, 0].copy()
        s.name = sym
        if s.index[0] > pd.Timestamp(FETCH_START) + timedelta(days=10):
            print(f"    stale cache ({s.index[0].date()}) -> re-download")
            os.remove(p)
            return None
        return s
    except Exception:
        return None

def _fetch_one(sym, retries=4, delay=4.0):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(sym, start=FETCH_START, end=END,
                             progress=False, timeout=25, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("empty")
            s = df["Close"][sym].copy() if isinstance(df.columns, pd.MultiIndex) else df["Close"].copy()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.name = sym
            if s.notna().sum() < 50:
                raise ValueError(f"only {s.notna().sum()} rows")
            os.makedirs(CACHE_DIR, exist_ok=True)
            s.to_csv(_cache_path(sym))
            return s
        except Exception as e:
            if attempt == retries:
                print(f"    failed: {e}")
            time.sleep(delay * attempt + random.uniform(0.5, 2.0))
    return None

def fetch_prices():
    print(f"Fetching {len(ETFS)} ETFs | {FETCH_START} -> {END}")
    series = []
    for i, (sym, name, short) in enumerate(ETFS, 1):
        print(f"  [{i:>2}/{len(ETFS)}] {name:<16} ({sym})", end=" ... ", flush=True)
        s = _load_cache(sym)
        if s is not None:
            print(f"cache ({s.notna().sum()} rows)")
        else:
            s = _fetch_one(sym)
            if s is not None:
                print(f"ok ({s.notna().sum()} rows)")
            else:
                print("SKIP")
            time.sleep(random.uniform(1.0, 2.0))
        if s is not None:
            series.append(s)

    if not series:
        return pd.DataFrame()

    prices = pd.concat(series, axis=1).sort_index().ffill(limit=5)
    valid = [c for c in prices.columns if prices[c].notna().sum() > 100]
    prices = prices[valid].dropna(how="all")
    print(f"\n  {len(prices)} trading days, {len(valid)} instruments\n")
    return prices

# ── RS ENGINE ─────────────────────────────────────────────────────────────────
def period_return(series, idx_now, lookback):
    idx_past = idx_now - lookback
    if idx_past < 0:
        return None
    p0, p1 = series.iloc[idx_past], series.iloc[idx_now]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return None
    return float(p1 / p0 - 1.0)

def compute_rs(prices, idx, avail, lookback):
    rets = {s: period_return(prices[s], idx, lookback) for s in avail if s in prices.columns}
    valid = [s for s in rets if rets[s] is not None]
    scores = {}
    for sym in avail:
        if rets.get(sym) is None:
            scores[sym] = None
            continue
        peers = [rets[s] for s in valid if s != sym]
        scores[sym] = float(np.mean([rets[sym] - p for p in peers])) if peers else 0.0
    return scores, rets

def build_matrix(rets, avail):
    matrix = []
    for si, _, _ in ETFS:
        row = []
        for sj, _, _ in ETFS:
            if si == sj:
                row.append(0.0)
            elif si in avail and sj in avail and rets.get(si) is not None and rets.get(sj) is not None:
                row.append(round((rets[si] - rets[sj]) * 100, 3))
            else:
                row.append(None)
        matrix.append(row)
    return matrix

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
def run_backtest(prices, lookback):
    avail = prices.columns.tolist()

    # Signal = Friday close, Execute = Monday close (no look-ahead bias)
    pairs = []
    for f in pd.date_range(TRADE_START, END, freq="W-FRI"):
        si = prices.index.searchsorted(f, side="right") - 1
        ei = si + 1
        if 0 <= si < len(prices) and ei < len(prices):
            pairs.append((si, ei))

    if not pairs:
        return None

    cash = float(INITIAL)
    holdings = {}
    cur_top3 = []
    peak = float(INITIAL)
    total_trades = 0
    hold_count   = {e[0]: 0 for e in ETFS}
    instr_trades = {e[0]: 0 for e in ETFS}
    eq, nf, dd, log = [], [], [], []

    nifty_px0 = float(prices[NIFTY_SYM].iloc[pairs[0][1]]) if NIFTY_SYM in prices.columns else None

    last_scores = last_rets = last_matrix = None

    for wi, (si, ei) in enumerate(pairs):
        date_str = str(prices.index[si].date())
        exec_str = str(prices.index[ei].date())

        scores, rets = compute_rs(prices, si, avail, lookback)
        ranked   = sorted([(s,v) for s,v in scores.items() if v is not None], key=lambda x:-x[1])
        new_top3 = [s for s,_ in ranked[:TOP_N]]
        if len(new_top3) < TOP_N:
            continue

        needs_rebal = (wi == 0) or (set(new_top3) != set(cur_top3))
        exiting  = [s for s in cur_top3 if s not in new_top3]
        entering = [s for s in new_top3  if s not in cur_top3]

        if needs_rebal:
            for sym in exiting:
                if sym in holdings and holdings[sym] > 0:
                    px    = float(prices[sym].iloc[ei])
                    cash += holdings[sym] * px * (1 - COST_PCT)
                    instr_trades[sym] += 1; total_trades += 1
                    del holdings[sym]

            retained = sum(holdings[s] * float(prices[s].iloc[ei])
                           for s in holdings if s in prices.columns)
            target   = (cash + retained) / TOP_N

            for sym in new_top3:
                px   = float(prices[sym].iloc[ei])
                cval = holdings.get(sym, 0) * px
                if sym in entering or abs(cval - target) > target * 0.05:
                    if sym in holdings and holdings[sym] > 0:
                        cash += holdings[sym] * px * (1 - COST_PCT)
                        instr_trades[sym] += 1; total_trades += 1
                    cash         -= target
                    holdings[sym] = target * (1 - COST_PCT) / px
                    instr_trades[sym] += 1; total_trades += 1

        port = max(cash + sum(holdings[s] * float(prices[s].iloc[ei])
                              for s in holdings if s in prices.columns), 0)
        for sym in new_top3:
            hold_count[sym] = hold_count.get(sym, 0) + 1

        nifty_px = float(prices[NIFTY_SYM].iloc[ei]) if NIFTY_SYM in prices.columns else None
        nifty_v  = (nifty_px / nifty_px0 * INITIAL) if nifty_px and nifty_px0 else INITIAL

        if port > peak: peak = port
        ddown = (port - peak) / peak * 100

        eq.append({"x": date_str, "y": round(port, 2)})
        nf.append({"x": date_str, "y": round(nifty_v, 2)})
        dd.append({"x": date_str, "y": round(ddown, 3)})
        log.append({
            "date": date_str, "exec": exec_str,
            "top3": new_top3, "exiting": exiting, "entering": entering,
            "changed": needs_rebal and wi > 0, "capital": round(port, 2),
            "scores": {s: round(v,6) if v is not None else None for s,v in scores.items()},
            "rets":   {s: round(v,6) if v is not None else None for s,v in rets.items()},
        })
        cur_top3    = new_top3
        last_scores = scores
        last_rets   = rets
        last_matrix = build_matrix(rets, avail)

    return dict(eq=eq, nf=nf, dd=dd, log=log,
                trades=total_trades, hold_count=hold_count,
                instr_trades=instr_trades,
                last_scores=last_scores, last_rets=last_rets,
                last_matrix=last_matrix, avail=avail)

# ── STATS ─────────────────────────────────────────────────────────────────────
def calc_stats(eq, nf, dd, trades):
    if not eq:
        return None
    fv  = eq[-1]["y"]
    nfv = nf[-1]["y"]
    yrs = max((pd.Timestamp(END) - pd.Timestamp(TRADE_START)).days / 365.25, 0.05)
    tr  = (fv - INITIAL) / INITIAL * 100
    cagr= (pow(fv/INITIAL,   1/yrs) - 1) * 100
    nc  = (pow(nfv/INITIAL,  1/yrs) - 1) * 100
    wr  = [(eq[i]["y"]-eq[i-1]["y"])/eq[i-1]["y"] for i in range(1,len(eq))]
    shp = (np.mean(wr)/np.std(wr))*np.sqrt(52) if wr and np.std(wr)>0 else 0
    mdd = min(d["y"] for d in dd) if dd else 0
    return dict(final=round(fv,2), ret=round(tr,2), cagr=round(cagr,2),
                sharpe=round(shp,2), mdd=round(mdd,2), trades=trades,
                nifty_cagr=round(nc,2), alpha=round(cagr-nc,2), weeks=len(eq))

# ── HTML ──────────────────────────────────────────────────────────────────────
def build_html(results, best_lb):
    """results: dict of lookback -> {eq,nf,dd,log,...,stats}"""

    def p(v,d=1): return f"{'+' if v>0 else ''}{v:.{d}f}%"
    def inr(v):   return f"Rs {v/1e5:.2f}L"
    def cls(v):   return "green" if v>=0 else "red"

    br = results[best_lb]
    bs = br["stats"]
    avail = br["avail"]

    # ── Lookback comparison table ──────────────────────────────────────────
    cmp_html = '<table><thead><tr><th>Lookback</th><th>CAGR</th><th>Total Return</th><th>Sharpe</th><th>Max DD</th><th>Alpha vs Nifty</th><th>Trades</th></tr></thead><tbody>'
    for lb in LOOKBACKS:
        s = results[lb]["stats"]
        if s is None:
            cmp_html += f'<tr><td><b>{lb}d</b></td><td colspan="6" style="color:var(--muted)">No data</td></tr>'
            continue
        is_best = lb == best_lb
        bg = 'background:rgba(79,142,247,0.08);' if is_best else ''
        tag= ' <span class="bd bg">BEST</span>' if is_best else ''
        cmp_html += (f'<tr style="{bg}">'
                     f'<td><b>{lb}d</b>{tag}</td>'
                     f'<td class="{cls(s["cagr"])}">{p(s["cagr"])}</td>'
                     f'<td class="{cls(s["ret"])}">{p(s["ret"])}</td>'
                     f'<td class="{cls(s["sharpe"]-1)}">{s["sharpe"]:.2f}</td>'
                     f'<td class="red">{p(s["mdd"])}</td>'
                     f'<td class="{cls(s["alpha"])}">{p(s["alpha"])}</td>'
                     f'<td style="color:var(--muted)">{s["trades"]}</td></tr>')
    cmp_html += '</tbody></table>'

    # ── Rankings ──────────────────────────────────────────────────────────
    ls = br["last_scores"]; lr = br["last_rets"]
    rows = [(e,ls.get(e[0]),lr.get(e[0])) for e in ETFS
            if e[0] in avail and ls.get(e[0]) is not None]
    rows.sort(key=lambda x:-x[1])
    medals = ["1","2","3"]
    rank_html = '<table><thead><tr><th>Rank</th><th>ETF</th><th>RS Score</th><th>Return</th><th>Signal</th></tr></thead><tbody>'
    for i,(etf,sc,ret) in enumerate(rows):
        top = i < TOP_N
        med = medals[i] if i < 3 else str(i+1)
        rs  = f"{ret*100:+.2f}%" if ret is not None else "-"
        rc  = "green" if (ret or 0)>0 else "red"
        sig = f'<span class="bd {"bg" if top else "bm"}">{"LONG" if top else "OUT"}</span>'
        bg  = 'background:rgba(79,142,247,.06);' if top else ''
        rank_html += (f'<tr style="{bg}"><td>{med}</td>'
                      f'<td><b>{etf[1]}</b> <span style="color:var(--muted);font-size:.71rem">{etf[2]}</span></td>'
                      f'<td style="color:var(--accent)">{sc*100:+.3f}</td>'
                      f'<td class="{rc}">{rs}</td><td>{sig}</td></tr>')
    rank_html += '</tbody></table>'

    # ── Contrib ───────────────────────────────────────────────────────────
    hc = br["hold_count"]; it = br["instr_trades"]
    rows2 = sorted(ETFS, key=lambda e:-hc.get(e[0],0))
    cont_html = '<table><thead><tr><th>ETF</th><th>Weeks</th><th>% Time</th><th>Trades</th></tr></thead><tbody>'
    tw = bs["weeks"]
    for etf in rows2:
        wk = hc.get(etf[0],0); pt = round(wk/tw*100) if tw else 0
        bar = f'<div style="height:5px;width:{pt}px;max-width:80px;background:var(--accent);border-radius:3px;min-width:2px;display:inline-block"></div>'
        cont_html += f'<tr><td><b>{etf[1]}</b></td><td>{wk}</td><td>{bar} {pt}%</td><td style="color:var(--muted)">{it.get(etf[0],0)}</td></tr>'
    cont_html += '</tbody></table>'

    # ── Matrix ────────────────────────────────────────────────────────────
    lm = br["last_matrix"]
    vis = [e for e in ETFS if e[0] in avail]
    mat_html = '<table class="mx"><thead><tr><th>vs</th>'
    for e in vis: mat_html += f'<th>{e[2]}</th>'
    mat_html += '<th style="color:var(--accent)">Wins</th></tr></thead><tbody>'
    for i,ei in enumerate(ETFS):
        if ei[0] not in avail: continue
        row  = lm[i]
        wins = sum(1 for v in row if v is not None and v>0)
        tot  = sum(1 for v in row if v is not None and v!=0)
        mat_html += f'<tr><td class="rl">{ei[2]}</td>'
        for j,ej in enumerate(ETFS):
            if ej[0] not in avail: continue
            v = row[j]
            if i==j: mat_html += '<td style="background:var(--card2);color:var(--muted)">-</td>'
            elif v is None: mat_html += '<td style="color:var(--muted)">N/A</td>'
            else:
                inten = min(abs(v)/8,0.8)
                bg2 = f'rgba(34,197,94,{inten:.2f})' if v>0 else f'rgba(239,68,68,{inten:.2f})'
                clr = '#22c55e' if v>0 else '#ef4444'
                mat_html += f'<td style="background:{bg2};color:{clr}">{v:+.1f}%</td>'
        mat_html += f'<td style="color:var(--accent);font-weight:700">{wins}/{tot}</td></tr>'
    mat_html += '</tbody></table>'

    # ── Log ───────────────────────────────────────────────────────────────
    em = {e[0]:e for e in ETFS}
    log_html = ''
    for lg in reversed(br["log"]):
        tags = ''.join(f'<span class="tg">{em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["top3"])
        chg = ''
        if lg["changed"]:
            ex = ' '.join(f'<span class="bd br">- {em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["exiting"])
            en = ' '.join(f'<span class="bd bg">+ {em[s][2] if s in em else s.replace(".NS","")}</span>' for s in lg["entering"])
            chg = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:3px">{ex} {en}</div>'
        nc  = '' if lg["changed"] else '<span style="color:var(--muted);font-size:.72rem;margin-left:5px">no change</span>'
        c   = '1' if lg["changed"] else '0'
        log_html += (f'<div class="lr" data-c="{c}">'
                     f'<div class="ld">{lg["date"]}<span style="display:block;font-size:.68rem">exec {lg["exec"]}</span></div>'
                     f'<div class="lb2"><div style="display:flex;align-items:center;flex-wrap:wrap">{tags}{nc}</div>'
                     f'{chg}<div style="color:var(--muted);font-size:.72rem;margin-top:2px">Rs {lg["capital"]/1000:.1f}K</div>'
                     f'</div></div>\n')

    # ── Multi-lookback equity chart datasets ──────────────────────────────
    lb_colors = {15:'#f59e0b', 30:'#22c55e', 45:'#a78bfa', 55:'#fb923c', 65:'#4f8ef7'}
    multi_datasets = []
    for lb in LOOKBACKS:
        eq = results[lb]["eq"]
        if not eq: continue
        multi_datasets.append({
            "label": f"{lb}d RS",
            "data":  [{"x":d["x"],"y":d["y"]} for d in eq],
            "borderColor": lb_colors[lb],
            "backgroundColor": "transparent",
            "borderWidth": 2 if lb == best_lb else 1.5,
            "pointRadius": 0,
            "tension": 0.3,
            "borderDash": [] if lb == best_lb else [4,3],
        })
    # add nifty
    nf_data = br["nf"]
    multi_datasets.append({
        "label": "NiftyBees",
        "data":  [{"x":d["x"],"y":d["y"]} for d in nf_data],
        "borderColor": "#64748b",
        "backgroundColor": "transparent",
        "borderWidth": 1,
        "pointRadius": 0,
        "tension": 0.3,
        "borderDash": [6,3],
    })

    best_eq = json.dumps([{"x":d["x"],"y":d["y"]} for d in br["eq"]])
    best_nf = json.dumps([{"x":d["x"],"y":d["y"]} for d in br["nf"]])
    best_dd = json.dumps([{"x":d["x"],"y":d["y"]} for d in br["dd"]])
    multi_ds= json.dumps(multi_datasets)

    upd = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>India ETF RS Backtest</title>
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
.cc h3{{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}}
.chartbox{{position:relative;width:100%}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
@media(max-width:780px){{.g2{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
th{{background:var(--card2);color:var(--muted);font-weight:600;text-transform:uppercase;font-size:.68rem;padding:8px 11px;text-align:left;border-bottom:1px solid var(--border)}}
td{{padding:7px 11px;border-bottom:1px solid var(--border)}}
tr:last-child td{{border-bottom:none}}
.bd{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.71rem;font-weight:600}}
.bg{{background:rgba(34,197,94,.15);color:var(--green)}}
.br{{background:rgba(239,68,68,.15);color:var(--red)}}
.bm{{background:rgba(136,146,176,.1);color:var(--muted)}}
.st{{font-size:.93rem;font-weight:700;margin-bottom:13px;display:flex;align-items:center;gap:8px}}
.st::before{{content:'';display:block;width:4px;height:17px;background:var(--accent);border-radius:2px}}
.mw{{overflow-x:auto}}
.mx{{border-collapse:collapse;font-size:.7rem;white-space:nowrap}}
.mx th,.mx td{{padding:5px 6px;border:1px solid var(--border);text-align:center;min-width:50px}}
.mx th{{background:var(--card2);color:var(--muted);font-weight:600}}
.mx .rl{{font-weight:600;color:var(--text);background:var(--card2);text-align:left;padding-left:9px;min-width:62px}}
.tg{{display:inline-flex;background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.3);color:var(--accent);border-radius:5px;padding:2px 8px;font-size:.77rem;font-weight:600;margin:2px}}
.tabs{{display:flex;gap:4px;background:var(--card2);padding:4px;border-radius:8px;width:fit-content;margin-bottom:14px}}
.tab{{padding:5px 15px;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:500;color:var(--muted);transition:all .15s;border:none;background:none}}
.tab.active{{background:var(--accent);color:#fff}}
#log{{max-height:380px;overflow-y:auto}}
.lr{{display:flex;gap:11px;padding:7px 0;border-bottom:1px solid var(--border);font-size:.8rem}}
.ld{{color:var(--muted);min-width:90px;padding-top:2px;flex-shrink:0}}
.lb2{{flex:1}}
.upd{{text-align:right;color:var(--muted);font-size:.76rem;padding:8px 0 4px}}
</style>
</head><body>
<header>
  <div>
    <h1>India ETF Pairwise RS Backtest</h1>
    <div class="meta">{len(avail)} ETFs &middot; Lookbacks: {','.join(str(l)+'d' for l in LOOKBACKS)} &middot; Top-3 Long &middot; Weekly rebalance (Fri signal, Mon execute) &middot; Rs 10L &middot; Jan 2023&rarr;present</div>
  </div>
  <div class="meta">Updated: {upd}</div>
</header>
<div class="container">

<!-- STATS (best lookback = {best_lb}d) -->
<div class="sg">
  <div class="sc"><div class="lb">Final Value (best {best_lb}d)</div><div class="vl accent">{inr(bs["final"])}</div></div>
  <div class="sc"><div class="lb">Total Return</div><div class="vl {cls(bs["ret"])}">{p(bs["ret"])}</div></div>
  <div class="sc"><div class="lb">CAGR</div><div class="vl {cls(bs["cagr"])}">{p(bs["cagr"])}</div></div>
  <div class="sc"><div class="lb">Sharpe Ratio</div><div class="vl {cls(bs["sharpe"]-1)}">{bs["sharpe"]:.2f}</div></div>
  <div class="sc"><div class="lb">Max Drawdown</div><div class="vl red">{p(bs["mdd"])}</div></div>
  <div class="sc"><div class="lb">Total Trades</div><div class="vl">{bs["trades"]}</div></div>
  <div class="sc"><div class="lb">Nifty CAGR</div><div class="vl">{p(bs["nifty_cagr"])}</div></div>
  <div class="sc"><div class="lb">Alpha vs Nifty</div><div class="vl {cls(bs["alpha"])}">{p(bs["alpha"])}</div></div>
</div>

<!-- LOOKBACK COMPARISON TABLE -->
<div class="cc">
  <div class="st">Lookback Comparison — all periods vs NiftyBees</div>
  {cmp_html}
</div>

<!-- MULTI-LOOKBACK EQUITY CHART -->
<div class="cc">
  <h3>Equity Curve — All Lookback Periods vs NiftyBees</h3>
  <div class="chartbox" style="height:320px">
    <canvas id="mc"></canvas>
  </div>
</div>

<!-- BEST LOOKBACK EQUITY + DD -->
<div class="cc">
  <h3>Best Lookback ({best_lb}d) — Equity vs NiftyBees</h3>
  <div class="chartbox" style="height:280px">
    <canvas id="ec"></canvas>
  </div>
</div>
<div class="cc">
  <h3>Drawdown — Best Lookback ({best_lb}d)</h3>
  <div class="chartbox" style="height:150px">
    <canvas id="dc"></canvas>
  </div>
</div>

<div class="g2">
  <div class="cc"><div class="st">RS Rankings — {best_lb}d Lookback (last signal)</div>{rank_html}</div>
  <div class="cc"><div class="st">Time in Portfolio — {best_lb}d</div>{cont_html}</div>
</div>

<div class="cc">
  <div class="st">Pairwise RS Matrix — {best_lb}d (row outperforms column)</div>
  <div class="mw">{mat_html}</div>
</div>

<div class="cc">
  <div class="st">Weekly Log — {best_lb}d Lookback</div>
  <div class="tabs">
    <button class="tab active" onclick="sw('all',this)">All Weeks</button>
    <button class="tab" onclick="sw('chg',this)">Changes Only</button>
  </div>
  <div id="log">{log_html}</div>
</div>
<div class="upd">GitHub Actions &middot; yfinance &middot; Pairwise RS &middot; Signal: Fri close &rarr; Execute: Mon close</div>
</div>

<!-- Chart.js loaded at bottom, scripts after DOM is ready -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-date-fns/3.0.0/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
var eqRaw = {best_eq};
var nfRaw = {best_nf};
var ddRaw = {best_dd};
var multiDs = {multi_ds};

var gc = '#2e3250';

// Multi-lookback chart
new Chart(document.getElementById('mc'), {{
  type: 'line',
  data: {{ datasets: multiDs }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ labels: {{ color: '#8892b0', boxWidth: 12 }} }} }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'month' }},
            ticks: {{ color: '#8892b0', maxTicksLimit: 18 }}, grid: {{ color: gc }} }},
      y: {{ ticks: {{ color: '#8892b0', callback: function(v) {{ return 'Rs '+(v/1000).toFixed(0)+'K'; }} }},
            grid: {{ color: gc }} }}
    }}
  }}
}});

// Best-lookback equity chart
new Chart(document.getElementById('ec'), {{
  type: 'line',
  data: {{ datasets: [
    {{ label: 'RS Strategy ({best_lb}d)', data: eqRaw,
       borderColor: '#4f8ef7', backgroundColor: 'rgba(79,142,247,0.08)',
       borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
       parsing: false }},
    {{ label: 'NiftyBees', data: nfRaw,
       borderColor: '#f59e0b', backgroundColor: 'transparent',
       borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [5,4],
       parsing: false }}
  ]}},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ labels: {{ color: '#8892b0', boxWidth: 12 }} }},
      tooltip: {{ backgroundColor: '#1a1d27', titleColor: '#e2e8f0', bodyColor: '#8892b0',
        callbacks: {{ label: function(c) {{ return c.dataset.label + ': Rs ' + (c.raw.y/1000).toFixed(1) + 'K'; }} }} }} }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'month' }},
            ticks: {{ color: '#8892b0', maxTicksLimit: 18 }}, grid: {{ color: gc }} }},
      y: {{ ticks: {{ color: '#8892b0', callback: function(v) {{ return 'Rs '+(v/1000).toFixed(0)+'K'; }} }},
            grid: {{ color: gc }} }}
    }}
  }}
}});

// Drawdown chart
new Chart(document.getElementById('dc'), {{
  type: 'line',
  data: {{ datasets: [
    {{ label: 'Drawdown', data: ddRaw,
       borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.12)',
       borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true,
       parsing: false }}
  ]}},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }},
      tooltip: {{ backgroundColor: '#1a1d27',
        callbacks: {{ label: function(c) {{ return 'DD: '+c.raw.y.toFixed(2)+'%'; }} }} }} }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'month' }},
            ticks: {{ color: '#8892b0', maxTicksLimit: 18 }}, grid: {{ color: gc }} }},
      y: {{ ticks: {{ color: '#8892b0', callback: function(v) {{ return v.toFixed(1)+'%'; }} }},
            grid: {{ color: gc }} }}
    }}
  }}
}});

function sw(t, el) {{
  document.querySelectorAll('.tab').forEach(function(x) {{ x.classList.remove('active'); }});
  el.classList.add('active');
  document.querySelectorAll('.lr').forEach(function(r) {{
    r.style.display = (t==='all' || r.dataset.c==='1') ? 'flex' : 'none';
  }});
}}
</script>
</body></html>"""

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices = fetch_prices()
    if prices.empty:
        print("ERROR: No price data.", file=sys.stderr); sys.exit(1)

    results = {}
    for lb in LOOKBACKS:
        print(f"\n{'='*50}")
        print(f"Running backtest: lookback = {lb} days")
        print(f"{'='*50}")
        r = run_backtest(prices, lb)
        if r is None:
            print(f"  No results for {lb}d")
            results[lb] = {"eq":[],"nf":[],"dd":[],"log":[],"trades":0,
                           "hold_count":{},"instr_trades":{},"last_scores":None,
                           "last_rets":None,"last_matrix":None,"avail":[],"stats":None}
            continue
        r["stats"] = calc_stats(r["eq"], r["nf"], r["dd"], r["trades"])
        results[lb] = r
        s = r["stats"]
        if s:
            top3 = sorted([(sym,r["hold_count"].get(sym,0)) for sym in r["avail"]],
                          key=lambda x:-x[1])[:3]
            top3_names = [next((e[1] for e in ETFS if e[0]==s2),s2) for s2,_ in top3]
            print(f"  CAGR: {s['cagr']:+.1f}%  Sharpe: {s['sharpe']:.2f}  "
                  f"MaxDD: {s['mdd']:.1f}%  Alpha: {s['alpha']:+.1f}%")
            print(f"  Most held: {top3_names}")

    # Pick best lookback by Sharpe ratio
    valid_lbs = [lb for lb in LOOKBACKS if results[lb]["stats"] is not None]
    if not valid_lbs:
        print("ERROR: No valid results.", file=sys.stderr); sys.exit(1)
    best_lb = max(valid_lbs, key=lambda lb: results[lb]["stats"]["sharpe"])

    print(f"\n{'='*50}")
    print(f"BEST LOOKBACK: {best_lb} days (highest Sharpe)")
    print(f"{'='*50}")
    for lb in LOOKBACKS:
        s = results[lb]["stats"]
        if s:
            marker = " <-- BEST" if lb == best_lb else ""
            print(f"  {lb:>3}d | CAGR {s['cagr']:>+6.1f}% | Sharpe {s['sharpe']:.2f} | "
                  f"DD {s['mdd']:>5.1f}% | Alpha {s['alpha']:>+5.1f}%{marker}")

    os.makedirs("docs", exist_ok=True)
    html = build_html(results, best_lb)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport -> {OUT_PATH}")
