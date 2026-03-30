import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Stocks
stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# Step 2: Download data
data = yf.download(stocks, start="2022-01-01", end="2024-01-01")['Close']

# Step 3: Calculate returns
returns = data.pct_change()

# Step 4: Mean return & risk
mean_returns = returns.mean() * 252
risk = returns.std() * np.sqrt(252)

print("\n===== STOCK PERFORMANCE =====")
for stock in stocks:
    print(f"{stock}: Return = {mean_returns[stock]:.2f}, Risk = {risk[stock]:.2f}")

# ==============================
# 🚀 PORTFOLIO SIMULATION
# ==============================

num_portfolios = 1000

results = []
weights_list = []

for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    portfolio_return = np.sum(weights * mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    results.append([portfolio_return, portfolio_risk])
    weights_list.append(weights)

# Convert to DataFrame
results = pd.DataFrame(results, columns=['Return', 'Risk'])

# ==============================
# ⭐ FIND BEST PORTFOLIO (SHARPE)
# ==============================

# Sharpe Ratio (assuming risk-free rate = 0)
results['Sharpe'] = results['Return'] / results['Risk']

# Best portfolio
best_portfolio = results.loc[results['Sharpe'].idxmax()]
best_weights = weights_list[results['Sharpe'].idxmax()]

print("\n===== BEST PORTFOLIO =====")
print(f"Return: {best_portfolio['Return']:.2f}")
print(f"Risk: {best_portfolio['Risk']:.2f}")
print(f"Sharpe Ratio: {best_portfolio['Sharpe']:.2f}")

print("\nWeights:")
for i, stock in enumerate(stocks):
    print(f"{stock}: {best_weights[i]:.2f}")

# ==============================
# 📊 PLOT EFFICIENT FRONTIER
# ==============================

plt.figure(figsize=(10,6))

# All portfolios
plt.scatter(results['Risk'], results['Return'], alpha=0.5, label='Portfolios')

# Highlight best portfolio
plt.scatter(best_portfolio['Risk'], best_portfolio['Return'],
            color='red', marker='*', s=200, label='Best Portfolio')

plt.xlabel('Risk (Volatility)')
plt.ylabel('Return')
plt.title('Efficient Frontier')

plt.legend()
plt.show()