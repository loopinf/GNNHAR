import pickle
import numpy as np
import pandas as pd

# Load your 5-minute close crypto data
with open('crypto_5m_data.pkl', 'rb') as f:
    price_df = pickle.load(f)

# Calculate log returns
returns_df = np.log(price_df).diff().fillna(0)

# Compute 5-minute realized variance (squared returns)
squared_returns = returns_df ** 2

# Aggregate to 1-hour realized volatility
# Assuming crypto trades 24/7, so we have 12 5-min intervals per hour
hourly_rv = squared_returns.resample('1H').sum().sqrt()

# Save the processed hourly RV
hourly_rv.to_pickle('crypto_hourly_rv.pkl')

print(f"Processed hourly realized volatility for {len(hourly_rv.columns)} crypto assets")
print(f"Date range: {hourly_rv.index[0]} to {hourly_rv.index[-1]}")
