# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the original StrategyBacktester.

This script shows how to:
1. Generate sample financial data with trading signals.
2. Initialize the StrategyBacktester.
3. Run a backtest on the data.
4. Print the performance metrics.
5. Plot the results.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path to allow for absolute imports.
# This is necessary if you run the script directly from the examples folder.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from CoreQuantUtilities.backtester.backtester import StrategyBacktester

def generate_data_with_signals():
    """Generates a sample DataFrame with OHLCV data and a 'Signals' column."""
    print("Generating sample data...")
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', '2022-12-31', freq='D')
    n = len(dates)
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.75)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': close_prices - np.random.randn(n) * 0.25,
        'high': close_prices + np.abs(np.random.randn(n) * 0.5),
        'low': close_prices - np.abs(np.random.randn(n) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, n),
        'Signals': np.random.choice([-1, 0, 1], n, p=[0.05, 0.9, 0.05])
    })
    print("Sample data generated.")
    return sample_data

def main():
    """Main function to run the backtester example."""
    
    # 1. Generate data
    data = generate_data_with_signals()
    
    # 2. Initialize the backtester with desired parameters
    print("\nInitializing StrategyBacktester...")
    vectorized_backtester = StrategyBacktester(
        commission=0.001,
        slippage=0.001,
        time_horizon='1d'
    )
    
    # 3. Run the backtest
    print("Running backtest...")
    vectorized_backtester.backtest(data, signal_col='Signals')
    
    # 4. Print performance metrics
    vectorized_backtester.print_metrics()
    
    # 5. Plot the results
    print("\nDisplaying plot...")
    vectorized_backtester.plot_results()

if __name__ == '__main__':
    main()