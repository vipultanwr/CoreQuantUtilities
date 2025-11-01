import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StrategyBacktester:
    def __init__(self, commission=0.001, slippage=0.001, time_horizon='1d'):
        """
        Initialize the backtester
        
        Parameters:
        commission: Commission rate (0.001 = 0.1%)
        slippage: Slippage rate (0.001 = 0.1%)
        time_horizon: The data frequency for annualization.
                      Supported values: '1m', '5m', '15m', '30m', '1h', '1d', '1w', '1mo'.
        """
        self.commission = commission
        self.slippage = slippage
        self.time_horizon = time_horizon
        self.periods_per_year = self._get_periods_per_year(time_horizon)
        self.results = None
        self.trades = []
        
    def _get_periods_per_year(self, time_horizon):
        """Maps time horizon string to periods per year."""
        TRADING_DAYS_PER_YEAR = 252
        TRADING_HOURS_PER_DAY = 6.5  # For US equities; adjust for other markets

        horizon_map = {
            '1m': TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 60,
            '5m': TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * (60 / 5),
            '15m': TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * (60 / 15),
            '30m': TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * (60 / 30),
            '1h': TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY,
            '1d': TRADING_DAYS_PER_YEAR,
            '1w': 52,
            '1mo': 12,
        }
        if time_horizon not in horizon_map:
            raise ValueError(f"Unsupported time_horizon: '{time_horizon}'. Supported values are {list(horizon_map.keys())}")
        return horizon_map[time_horizon]
        
    def backtest(self, df, price_col='close', signal_col='Signals', date_col='date'):
        """
        Run the backtest on the provided DataFrame
        
        Parameters:
        df: DataFrame with OHLCV data and signals
        price_col: Column name for price data
        signal_col: Column name for signals (1=buy, -1=sell, 0=hold)
        date_col: Column name for date/datetime data
        """
        # Make a copy to avoid modifying original data
        data = df.copy()        
        # Normalize column names to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]
        price_col = price_col.lower()
        signal_col = signal_col.lower()
        date_col = date_col.lower()
        data = data.sort_values(date_col).reset_index(drop=True)
        
        # Initialize tracking variables
        position = 0  # Current position: 1=long, -1=short, 0=neutral
        portfolio_value = 1.0 # Start with 1.0 to represent 100% of initial capital for percentage tracking
        entry_price = 0
        
        # Lists to store results
        portfolio_values = []
        positions = []
        returns = []
        trades = []
        
        for i in range(len(data)):
            current_price = data.iloc[i][price_col]
            signal = data.iloc[i][signal_col]
            
            # Calculate portfolio value based on percentage change
            if position == 1:  # Long position
                portfolio_value *= (1 + (current_price - entry_price) / entry_price)
            elif position == -1:  # Short position
                portfolio_value *= (1 + (entry_price - current_price) / entry_price)
            
            # Process signals
            if signal == 1 and position != 1:  # Buy signal
                if position == -1:  # Close short position
                    # Calculate PnL as a percentage of the capital allocated to the trade
                    pnl_percentage = (entry_price - current_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.iloc[i][date_col],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'SHORT',
                        'pnl_percentage': pnl_percentage,
                        'return': pnl_percentage
                    })
                    portfolio_value *= (1 + pnl_percentage) # Apply PnL to portfolio
                
                # Open long position
                entry_price = current_price * (1 + self.slippage)
                entry_date = data.iloc[i][date_col]
                position = 1
                # Apply commission as a percentage deduction from portfolio value
                portfolio_value *= (1 - self.commission)
                
            elif signal == -1 and position != -1:  # Sell signal
                if position == 1:  # Close long position
                    # Calculate PnL as a percentage of the capital allocated to the trade
                    pnl_percentage = (current_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.iloc[i][date_col],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'LONG',
                        'pnl_percentage': pnl_percentage,
                        'return': pnl_percentage
                    })
                    portfolio_value *= (1 + pnl_percentage) # Apply PnL to portfolio
                
                # Open short position
                entry_price = current_price * (1 - self.slippage)
                entry_date = data.iloc[i][date_col]
                position = -1
                # Apply commission as a percentage deduction from portfolio value
                portfolio_value *= (1 - self.commission)
            
            # Store values
            portfolio_values.append(portfolio_value)
            positions.append(position)
            
            # Calculate returns
            if i == 0:
                returns.append(0)
            else:
                returns.append((portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1])
        
        # Close any remaining position
        if position != 0:
            current_price = data.iloc[-1][price_col]
            if position == 1:
                pnl_percentage = (current_price - entry_price) / entry_price
            else:
                pnl_percentage = (entry_price - current_price) / entry_price
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': data.iloc[-1][date_col],
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': 'LONG' if position == 1 else 'SHORT',
                'pnl_percentage': pnl_percentage,
                'return': pnl_percentage
            })
            portfolio_value *= (1 + pnl_percentage) # Apply PnL to portfolio
        
        # Create results DataFrame
        results = data.copy()
        results['portfolio_value'] = portfolio_values
        results['position'] = positions
        results['returns'] = returns
        results['cumulative_returns'] = (1 + pd.Series(returns)).cumprod() - 1
        
        # Store results
        self.results = results
        self.trades = pd.DataFrame(trades)
        
        return results
    
    def calculate_metrics(self, time_horizon=None):
        """Calculate comprehensive performance metrics"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        returns = self.results['returns'].dropna()
        portfolio_values = self.results['portfolio_value']

        if time_horizon:
            annualization_factor = self._get_periods_per_year(time_horizon)
        else:
            # Use the one from initialization
            annualization_factor = self.periods_per_year
        
        # Basic metrics
        total_return = portfolio_values.iloc[-1] - 1 # Since portfolio_values start at 1.0
        annualized_return = (1 + total_return) ** (annualization_factor / len(returns)) - 1 if len(returns) > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(annualization_factor)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win/Loss analysis
        if len(self.trades) > 0:
            winning_trades = self.trades[self.trades['pnl_percentage'] > 0]
            losing_trades = self.trades[self.trades['pnl_percentage'] < 0]
            win_rate = len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0
            avg_win_percentage = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
            avg_loss_percentage = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(avg_win_percentage / avg_loss_percentage) if avg_loss_percentage != 0 else 0
        else:
            win_rate = 0
            avg_win_percentage = 0
            avg_loss_percentage = 0
            profit_factor = 0
        
        # Additional metrics
        sortino_ratio = annualized_return / (returns[returns < 0].std() * np.sqrt(annualization_factor)) if len(returns[returns < 0]) > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Sortino Ratio': f"{sortino_ratio:.2f}",
            'Calmar Ratio': f"{calmar_ratio:.2f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Trades': len(self.trades),
            'Profit Factor': f"{profit_factor:.2f}",
            'Average Win Percentage': f"{avg_win_percentage:.2%}",
            'Average Loss Percentage': f"{avg_loss_percentage:.2%}"
        }
        
        return metrics
    
    def plot_results(self, figsize=(15, 12), date_col='date'):
        """Create comprehensive performance plots"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Strategy Performance Analysis', fontsize=16, fontweight='bold')
        results_df = self.results.copy()
        
        # 1. Portfolio Value Over Time (now Cumulative Returns)
        ax1 = axes[0, 0]
        ax1.plot(results_df[date_col], results_df['portfolio_value'] * 100, 
                label='Strategy Cumulative Returns', linewidth=2, color='blue')
        ax1.set_title('Cumulative Returns Over Time')
        ax1.set_ylabel('Cumulative Returns (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Returns
        ax2 = axes[0, 1]
        ax2.plot(results_df[date_col], results_df['cumulative_returns'] * 100, 
                color='green', linewidth=2)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Returns (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = axes[1, 0]
        returns = results_df['returns'].dropna()
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
        ax3.fill_between(results_df[date_col].iloc[drawdown.index], drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = axes[1, 1]
        monthly_returns = results_df.set_index(date_col)['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
        
        if len(monthly_returns) > 1:
            # Create a simple bar chart for monthly returns
            ax4.bar(range(len(monthly_returns)), monthly_returns * 100, 
                   color=['green' if x > 0 else 'red' for x in monthly_returns])
            ax4.set_title('Monthly Returns')
            ax4.set_ylabel('Monthly Return (%)')
            ax4.set_xticks(range(0, len(monthly_returns), max(1, len(monthly_returns)//10)))
            ax4.set_xticklabels([monthly_returns.index[i] for i in range(0, len(monthly_returns), max(1, len(monthly_returns)//10))], 
                               rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns')
        
        plt.tight_layout()
        plt.show()
    
    def print_metrics(self):
        """Print performance metrics in a formatted table"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print("STRATEGY PERFORMANCE METRICS")
        print("="*50)
        
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        
        print("\n" + "="*50)
        
        if len(self.trades) > 0:
            print("RECENT TRADES:")
            print("-"*50)
            print(self.trades.tail().to_string(index=False))
        
        return metrics

# Example usage with your data
def run_backtest_example():
    """
    Example of how to use the backtester with your data
    """
    # Sample data (replace with your actual DataFrame)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    # Create sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.02)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': close_prices + np.random.randn(n) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n) * 1.5),
        'low': close_prices - np.abs(np.random.randn(n) * 1.5),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n),
        'Signals': np.random.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1])  # 10% buy, 10% sell, 80% hold
    })
    
    # Initialize and run backtester
    backtester = StrategyBacktester(
        commission=0.001,   # 0.1% commission
        slippage=0.001,     # 0.1% slippage
        time_horizon='1d'   # Specify data frequency
    )
    
    # Run backtest
    results = backtester.backtest(sample_data, price_col='close', signal_col='Signals')
    
    # Print metrics
    backtester.print_metrics()
    
    # Plot results
    backtester.plot_results()
    
    return backtester

# Uncomment the line below to run the example
# backtester = run_backtest_example()
