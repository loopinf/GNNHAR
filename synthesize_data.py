import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_crypto_data(n_symbols=38, n_periods=2016, seed=42):
    """
    Generate synthetic 5-minute cryptocurrency price data with realistic dependencies
    
    Parameters:
    -----------
    n_symbols : int
        Number of cryptocurrency symbols to generate
    n_periods : int
        Number of 5-minute periods (default: 2016 = 1 week)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamps as index, symbols as columns, and prices as values
    """
    np.random.seed(seed)
    
    # Create symbols
    base_names = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX", "LINK"]
    symbols = []
    
    # Generate main symbols
    for name in base_names:
        symbols.append(name)
    
    # Generate additional symbols to reach n_symbols
    alt_prefixes = ["A", "B", "C", "D", "E", "L", "M", "N", "P", "S", "T", "U", "X", "Z"]
    i = 0
    while len(symbols) < n_symbols:
        symbols.append(f"{alt_prefixes[i % len(alt_prefixes)]}COIN{i+1}")
        i += 1
    
    symbols = symbols[:n_symbols]  # Ensure exactly n_symbols
    
    # Create timestamp index
    end_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(minutes=5*n_periods)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')[:-1]  # Exclude end
    
    # Initialize price dataframe with starting prices
    prices = pd.DataFrame(index=timestamps, columns=symbols)
    
    # Set initial prices (more realistic ranges)
    initial_prices = {
        'BTC': 40000 + np.random.normal(0, 1000),
        'ETH': 2200 + np.random.normal(0, 100),
        'XRP': 0.50 + np.random.normal(0, 0.05),
        'ADA': 0.40 + np.random.normal(0, 0.05),
        'SOL': 70 + np.random.normal(0, 5),
        'DOT': 6 + np.random.normal(0, 0.5),
        'DOGE': 0.08 + np.random.normal(0, 0.01),
        'AVAX': 18 + np.random.normal(0, 1),
        'LINK': 7 + np.random.normal(0, 0.5)
    }
    
    # Fill in remaining initial prices
    for s in symbols:
        if s not in initial_prices:
            if s.startswith('A'):
                initial_prices[s] = 5 + np.random.normal(0, 1)
            elif s.startswith('B'):
                initial_prices[s] = 2 + np.random.normal(0, 0.5)
            else:
                initial_prices[s] = 0.5 + np.random.normal(0, 0.1)
    
    # Define relationships between assets (VAR structure)
    # Format: {influenced_asset: [(influencing_asset, lag, strength), ...]}
    relationships = {
        'ETH': [('BTC', 1, 0.4)],  # ETH follows BTC with lag 1
        'XRP': [('BTC', 2, 0.2), ('ETH', 1, 0.3)],  # XRP follows both BTC and ETH
        'ADA': [('ETH', 1, 0.35)],
        'DOT': [('ETH', 1, 0.3), ('BTC', 2, 0.25)],
        'AVAX': [('SOL', 1, 0.3), ('ETH', 1, 0.2)],
        'LINK': [('ETH', 1, 0.4), ('BTC', 1, 0.2)],
        # Add relationships for some alt coins
        'ACOIN1': [('BTC', 1, 0.5)],
        'BCOIN1': [('ETH', 1, 0.4)],
        'CCOIN1': [('XRP', 1, 0.3)],
        'DCOIN1': [('ADA', 1, 0.25)],
        'ECOIN1': [('SOL', 1, 0.35)]
    }
    
    # Create log returns innovation series with volatility clustering (GARCH-like)
    innovations = {}
    volatilities = {}
    
    # Initialize volatility for each asset
    for s in symbols:
        volatilities[s] = np.zeros(n_periods)
        volatilities[s][0] = 0.001  # Initial volatility
        
        # Different assets have different base volatility
        if s == 'BTC':
            base_vol = 0.0008
        elif s in ['ETH', 'SOL', 'AVAX']:
            base_vol = 0.001
        else:
            base_vol = 0.0015
        
        # Generate volatility series with clustering
        for t in range(1, n_periods):
            volatilities[s][t] = 0.1*base_vol + 0.8*volatilities[s][t-1] + 0.1*(np.random.normal(0, 1)**2)*base_vol
            
        # Generate return innovations based on volatility    
        innovations[s] = np.random.normal(0, 1, n_periods) * volatilities[s]
    
    # Initialize log prices
    log_prices = {}
    for s in symbols:
        log_prices[s] = np.zeros(n_periods)
        log_prices[s][0] = np.log(initial_prices[s])
    
    # Generate price series with AR and cross-asset effects
    for t in range(1, n_periods):
        for s in symbols:
            # AR(1) component for all assets
            ar1_coef = 0.2 if s not in ['BTC', 'ETH'] else 0.15
            
            # Add AR(1) effect
            log_return = ar1_coef * (log_prices[s][t-1] - log_prices[s][t-2]) if t > 1 else 0
            
            # AR(2) component for some assets
            if t > 2 and s in ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']:
                ar2_coef = 0.05
                log_return += ar2_coef * (log_prices[s][t-2] - log_prices[s][t-3])
            
            # Add cross-asset effects (VAR structure)
            if s in relationships:
                for influencer, lag, strength in relationships[s]:
                    if t > lag:
                        influencer_return = log_prices[influencer][t-lag] - log_prices[influencer][t-lag-1]
                        log_return += strength * influencer_return
            
            # Add innovation term
            log_return += innovations[s][t]
            
            # Update log price
            log_prices[s][t] = log_prices[s][t-1] + log_return
    
    # Convert log prices to actual prices
    for s in symbols:
        prices[s] = np.exp(log_prices[s])
    
    print(f"Generated synthetic 5-minute price data for {n_symbols} crypto assets")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Sample prices:")
    print(prices.iloc[:5, :5])  # Show first 5 rows and columns
    
    return prices

def visualize_data(prices):
    """Visualize some of the synthetic data"""
    # Plot prices for major assets
    main_assets = ['BTC', 'ETH', 'XRP', 'SOL', 'AVAX']
    plt.figure(figsize=(12, 6))
    for asset in main_assets:
        if asset in prices.columns:
            # Normalize to start at 100 for comparison
            plt.plot(prices.index, prices[asset]/prices[asset].iloc[0]*100, label=asset)
    plt.title('Normalized Price Trends (Start=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('synthetic_prices.png')
    
    # Plot correlation matrix
    returns = np.log(prices).diff().dropna()
    plt.figure(figsize=(16, 12))
    corr = returns.iloc[:, :15].corr()  # First 15 assets for readability
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Return Correlations Between Assets')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    print("Visualization saved to 'synthetic_prices.png' and 'correlation_matrix.png'")

if __name__ == "__main__":
    # Generate synthetic data
    prices = generate_synthetic_crypto_data(n_symbols=38, n_periods=2016)  # 1 week of 5-min data
    
    # Save to pickle file
    with open('synthetic_crypto_data.pkl', 'wb') as f:
        pickle.dump(prices, f)
    print("Saved synthetic data to 'synthetic_crypto_data.pkl'")
    
    # Visualize data (if matplotlib and seaborn are available)
    try:
        visualize_data(prices)
    except Exception as e:
        print(f"Visualization failed: {e}")