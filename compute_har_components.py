import pandas as pd
import numpy as np

def compute_har_components(rv_data, forecast_horizon=1):
    """
    Compute HAR components for the realized volatility
    """
    # Log transform the data (common in volatility modeling)
    log_rv = np.log(rv_data)
    
    # Calculate hourly, daily, weekly components
    hourly_rv = log_rv.shift(1)
    daily_rv = log_rv.shift(1).rolling(24).mean()  # 24-hour average
    weekly_rv = log_rv.shift(1).rolling(168).mean() # 168-hour (7-day) average
    
    # Align all data and drop NAs
    har_components = pd.concat([hourly_rv, daily_rv, weekly_rv], axis=1, 
                              keys=['hourly', 'daily', 'weekly'])
    
    # Target variable: h-step ahead volatility
    target = log_rv.shift(-forecast_horizon)
    
    # Clean data
    valid_idx = har_components.dropna().index
    valid_idx = valid_idx.intersection(target.dropna().index)
    
    X = har_components.loc[valid_idx]
    y = target.loc[valid_idx]
    
    return X, y

# Load the hourly RV data
hourly_rv = pd.read_pickle('crypto_hourly_rv.pkl')

# Compute HAR components for different forecast horizons
horizons = [1, 24, 168]  # 1-hour, 1-day, 1-week

for h in horizons:
    X, y = compute_har_components(hourly_rv, forecast_horizon=h)
    
    # Save processed data
    X.to_pickle(f'crypto_X_h{h}.pkl')
    y.to_pickle(f'crypto_y_h{h}.pkl')
    
    print(f"Processed data for {h}-hour ahead forecasting")
