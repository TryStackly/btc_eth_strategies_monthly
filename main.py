import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === 1. Download monthly closing prices for BTC and ETH (last ~4+ years) ===
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years to be safe

print("Downloading BTC and ETH monthly data...")
btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1mo', progress=False)['Close']
eth = yf.download('ETH-USD', start=start_date, end=end_date, interval='1mo', progress=False)['Close']

# === 2. Combine and make sure we have proper aligned dates ===
df = pd.concat([btc, eth], axis=1).dropna()
df.columns = ['BTC', 'ETH']

if len(df) < 12:
    raise ValueError("Not enough monthly data. Something went wrong with yfinance.")

dates = df.index
btc_prices = df['BTC'].values
eth_prices = df['ETH'].values

print(f"Got {len(df)} monthly data points from {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")

# === 3. Running All-Time-High ===
btc_ath = np.maximum.accumulate(btc_prices)
eth_ath = np.maximum.accumulate(eth_prices)

# Approximate circulating supply ratio ETH/BTC (good enough for this purpose)
supply_ratio_eth_to_btc = 120_000_000 / 19_700_000  # ≈6.09

# === 4. Simulation ===
monthly_budget = 500.0
n = len(dates)

# Holdings
btc_holding = {'ATH': 0.0, 'MC': 0.0, 'EQ': 0.0}
eth_holding = {'ATH': 0.0, 'MC': 0.0, 'EQ': 0.0}

# Portfolio value over time
value = {'ATH': [], 'MC': [], 'EQ': []}

for i in range(n):
    p_btc = btc_prices[i]
    p_eth = eth_prices[i]

    # ---- ATH Strategy ----
    dist_btc = 1 - p_btc / btc_ath[i]
    dist_eth = 1 - p_eth / eth_ath[i]
    total_dist = dist_btc + dist_eth + 1e-12
    w_btc_ath = dist_btc / total_dist
    w_eth_ath = dist_eth / total_dist

    btc_holding['ATH'] += (monthly_budget * w_btc_ath) / p_btc
    eth_holding['ATH'] += (monthly_budget * w_eth_ath) / p_eth

    # ---- Market Cap Strategy ----
    mc_ratio = (p_eth * supply_ratio_eth_to_btc) / p_btc
    w_btc_mc = 1 / (1 + mc_ratio)
    w_eth_mc = 1 - w_btc_mc

    btc_holding['MC'] += (monthly_budget * w_btc_mc) / p_btc
    eth_holding['MC'] += (monthly_budget * w_eth_mc) / p_eth

    # ---- Equal Weight ----
    btc_holding['EQ'] += (monthly_budget * 0.5) / p_btc
    eth_holding['EQ'] += (monthly_budget * 0.5) / p_eth

    # Record portfolio values
    value['ATH'].append(btc_holding['ATH'] * p_btc + eth_holding['ATH'] * p_eth)
    value['MC'].append(btc_holding['MC'] * p_btc + eth_holding['MC'] * p_eth)
    value['EQ'].append(btc_holding['EQ'] * p_btc + eth_holding['EQ'] * p_eth)

# === 5. Plot ===
plt.figure(figsize=(14, 7))
plt.plot(dates, value['ATH'], label='ATH Strategy', color='#00d26a', linewidth=3.5)
plt.plot(dates, value['MC'], label='Market Cap Strategy', color='#ff4444', linewidth=3)
plt.plot(dates, value['EQ'], label='Equal Weight (50/50)', color='#4488ff', linewidth=3)

plt.title('$500 Monthly DCA into BTC + ETH  →  Strategy Comparison', fontsize=18, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio Value (USD)', fontsize=12)
plt.legend(fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()

# Save perfect image for X/Twitter
plt.savefig('btc_eth_strategies_monthly.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nGraph saved as → btc_eth_strategies_monthly.png")

# === 6. Final results ===
total_invested = monthly_budget * n
print(f"\nTotal invested: ${total_invested:,.0f} over {n} months")
for name, short in [('ATH Strategy', 'ATH'), ('Market Cap Strategy', 'MC'), ('Equal Weight (50/50)', 'EQ')]:
    final = value[short][-1]
    roi = (final / total_invested - 1) * 100
    print(f"{name:25} → ${final:,.0f}   |   ROI +{roi:.1f}%")

plt.show()
