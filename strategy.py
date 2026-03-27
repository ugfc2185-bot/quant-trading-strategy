import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Step 1: Download data
data = yf.download("RELIANCE.NS", start="2022-01-01", end="2024-01-01")

# Step 2: Calculate Moving Averages
data['SMA50'] = data['Close'].rolling(50).mean()
data['SMA200'] = data['Close'].rolling(200).mean()

# Step 3: Create signals
data['Signal'] = 0
data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1

# Step 4: Generate buy/sell positions
data['Position'] = data['Signal'].diff()

# Step 5: Plot price + signals
plt.figure(figsize=(12,6))

plt.plot(data['Close'], label='Price', alpha=0.5)
plt.plot(data['SMA50'], label='SMA50')
plt.plot(data['SMA200'], label='SMA200')

# BUY signals
plt.scatter(data.index[data['Position'] == 1],
            data['Close'][data['Position'] == 1],
            marker='^', label='BUY')

# SELL signals
plt.scatter(data.index[data['Position'] == -1],
            data['Close'][data['Position'] == -1],
            marker='v', label='SELL')

plt.legend()
plt.title("Moving Average Trading Strategy")
plt.show()

# =========================
# 💸 BACKTESTING STARTS HERE
# =========================

# Step 6: Calculate returns
data['Returns'] = data['Close'].pct_change()

# Step 7: Strategy returns
data['Strategy_Returns'] = data['Returns'] * data['Signal'].shift(1)

# Step 8: Initial investment
initial = 100000

# Step 9: Portfolio growth
data['Portfolio'] = (1 + data['Strategy_Returns']).cumprod() * initial

# Step 10: Buy & Hold comparison
data['BuyHold'] = (1 + data['Returns']).cumprod() * initial

# Step 11: Plot portfolio comparison
plt.figure(figsize=(12,6))
plt.plot(data['Portfolio'], label='Strategy')
plt.plot(data['BuyHold'], label='Buy & Hold')
plt.legend()
plt.title("Strategy vs Market Performance")
plt.show()

# Step 12: Final results
final_value = data['Portfolio'].iloc[-1]
profit = final_value - initial

buyhold_value = data['BuyHold'].iloc[-1]
buyhold_profit = buyhold_value - initial

print("\n===== RESULTS =====")
print(f"Strategy Final Value: ₹{final_value:.2f}")
print(f"Strategy Profit: ₹{profit:.2f}")
print(f"Buy & Hold Value: ₹{buyhold_value:.2f}")
print(f"Buy & Hold Profit: ₹{buyhold_profit:.2f}")
# Sharpe Ratio
sharpe = (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()) * np.sqrt(252)

print(f"Sharpe Ratio: {sharpe:.2f}")