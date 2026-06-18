# 🇮🇳 India ETF — Pairwise RS Matrix Backtester

Automated weekly backtest of a **Relative Strength rotation strategy** across 12 NSE ETFs.  
Runs every Friday via **GitHub Actions**, publishes results to **GitHub Pages**.

## Strategy

| Parameter | Value |
|-----------|-------|
| Universe | 12 NSE ETFs (Gold, Silver, Nifty, Bank, IT, Pharma, Auto, PSU, Infra, Junior, GS Composite, Momentum) |
| Signal | 63-day pairwise RS — each ETF ranked against every other ETF |
| Portfolio | Long top-3 by RS score, equally weighted |
| Rebalance | Every Friday |
| Capital | ₹10,00,000 |
| Costs | 0.1% per trade (brokerage + slippage) |
| Backtest period | Jan 2023 → present (live-updated weekly) |

## ETFs Tracked

| Ticker | Name |
|--------|------|
| GOLDBEES.NS | GoldBees |
| SILVERBEES.NS | SilverBees |
| GSCOMP.NS | GS Composite |
| NIFTYBEES.NS | NiftyBees |
| JUNIORBEES.NS | JuniorBees (Nifty Next 50) |
| BANKBEES.NS | BankBees |
| ITMCAP.NS | ITBees |
| PSUBNKBEES.NS | PSU Bank BeES |
| PHARMABEES.NS | PharmaBees |
| AUTOBEES.NS | AutoBees |
| MOM100.NS | Momentum100 |
| INFRABEES.NS | InfraBees |

## Setup (5 minutes)

### 1. Create the repo

```bash
# Create a new GitHub repo named: etf-rs-backtest
# Then clone it and copy these files in
git clone https://github.com/YOUR_USERNAME/etf-rs-backtest
cd etf-rs-backtest
# copy backtest.py, requirements.txt, .github/workflows/backtest.yml
```

### 2. Enable GitHub Pages

- Go to repo **Settings → Pages**
- Source: **Deploy from a branch**
- Branch: `main` / folder: `/docs`
- Save → your live URL will be `https://YOUR_USERNAME.github.io/etf-rs-backtest`

### 3. Run it manually first

- Go to **Actions → ETF RS Backtest → Run workflow**
- Wait ~60 seconds
- Visit your GitHub Pages URL to see the report

### 4. It auto-runs every Friday at 4 AM UTC (9:30 AM IST)

No further action needed. The report updates itself weekly.

## Run locally

```bash
pip install -r requirements.txt
python backtest.py
# opens docs/index.html
```

## How the RS Score works

For each ETF `A`, we compute:

```
RS_score(A) = mean over all B≠A of [ return_A(63d) − return_B(63d) ]
```

This gives a **benchmark-free** ranking — Gold and Nifty are compared directly,  
not just against a single index. The top-3 instruments by RS score form the portfolio.

The **pairwise matrix** shows every A vs B comparison in a heatmap:  
- 🟢 Green = row instrument outperformed column over 63 days  
- 🔴 Red = row underperformed column
