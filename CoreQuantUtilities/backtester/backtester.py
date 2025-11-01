import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StrategyBacktester:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.001):
        """
        Initialize the backtester
        
        Parameters:
        initial_capital: Starting capital for the strategy
        commission: Commission rate (0.001 = 0.1%)
        slippage: Slippage rate (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None
        self.trades = []
        
    def backtest(self, df, price_col='close', signal_col='Signals'):
        """
        Run the backtest on the provided DataFrame
        
        Parameters:
        df: DataFrame with OHLCV data and signals
        price_col: Column name for price data
        signal_col: Column name for signals (1=buy, -1=sell, 0=hold)
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        data = data.sort_values('date').reset_index(drop=True)
        
        # Initialize tracking variables
        position = 0  # Current position: 1=long, -1=short, 0=neutral
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        entry_price = 0
        
        # Lists to store results
        portfolio_values = []
        positions = []
        cash_values = []
        returns = []
        trades = []
        
        for i in range(len(data)):
            current_price = data.iloc[i][price_col]
            signal = data.iloc[i][signal_col]
            
            # Calculate portfolio value
            if position == 0:
                portfolio_value = cash
            elif position == 1:  # Long position
                portfolio_value = cash + (current_price - entry_price) * (cash / entry_price)
            elif position == -1:  # Short position
                portfolio_value = cash - (current_price - entry_price) * (cash / entry_price)
            
            # Process signals
            if signal == 1 and position != 1:  # Buy signal
                if position == -1:  # Close short position
                    pnl = (entry_price - current_price) * (cash / entry_price)
                    cash += pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.iloc[i]['date'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'SHORT',
                        'pnl': pnl,
                        'return': pnl / self.initial_capital
                    })
                
                # Open long position
                entry_price = current_price * (1 + self.slippage)
                entry_date = data.iloc[i]['date']
                position = 1
                # Apply commission
                cash *= (1 - self.commission)
                
            elif signal == -1 and position != -1:  # Sell signal
                if position == 1:  # Close long position
                    pnl = (current_price - entry_price) * (cash / entry_price)
                    cash += pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.iloc[i]['date'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'LONG',
                        'pnl': pnl,
                        'return': pnl / self.initial_capital
                    })
                
                # Open short position
                entry_price = current_price * (1 - self.slippage)
                entry_date = data.iloc[i]['date']
                position = -1
                # Apply commission
                cash *= (1 - self.commission)
            
            # Store values
            portfolio_values.append(portfolio_value)
            positions.append(position)
            cash_values.append(cash)
            
            # Calculate returns
            if i == 0:
                returns.append(0)
            else:
                returns.append((portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1])
        
        # Close any remaining position
        if position != 0:
            current_price = data.iloc[-1][price_col]
            if position == 1:
                pnl = (current_price - entry_price) * (cash / entry_price)
            else:
                pnl = (entry_price - current_price) * (cash / entry_price)
            
            cash += pnl
            trades.append({
                'entry_date': entry_date,
                'exit_date': data.iloc[-1]['date'],
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': 'LONG' if position == 1 else 'SHORT',
                'pnl': pnl,
                'return': pnl / self.initial_capital
            })
        
        # Create results DataFrame
        results = data.copy()
        results['portfolio_value'] = portfolio_values
        results['position'] = positions
        results['cash'] = cash_values
        results['returns'] = returns
        results['cumulative_returns'] = (1 + pd.Series(returns)).cumprod() - 1
        
        # Store results
        self.results = results
        self.trades = pd.DataFrame(trades)
        
        return results
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        returns = self.results['returns'].dropna()
        portfolio_values = self.results['portfolio_value']
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win/Loss analysis
        if len(self.trades) > 0:
            winning_trades = self.trades[self.trades['pnl'] > 0]
            losing_trades = self.trades[self.trades['pnl'] < 0]
            win_rate = len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Additional metrics
        sortino_ratio = annualized_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
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
            'Average Win': f"${avg_win:.2f}",
            'Average Loss': f"${avg_loss:.2f}",
            'Final Portfolio Value': f"${portfolio_values.iloc[-1]:.2f}"
        }
        
        return metrics
    
    def plot_results(self, figsize=(15, 12)):
        """Create comprehensive performance plots"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        ax1.plot(self.results['date'], self.results['portfolio_value'], 
                label='Strategy', linewidth=2, color='blue')
        ax1.plot(self.results['date'], [self.initial_capital] * len(self.results), 
                label='Initial Capital', linestyle='--', color='red', alpha=0.7)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Returns
        ax2 = axes[0, 1]
        ax2.plot(self.results['date'], self.results['cumulative_returns'] * 100, 
                color='green', linewidth=2)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Returns (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = axes[1, 0]
        returns = self.results['returns'].dropna()
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
        ax3.fill_between(range(len(drawdown)), drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = axes[1, 1]
        monthly_returns = self.results.set_index('date')['returns'].resample('M').apply(
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
        initial_capital=100000,
        commission=0.001,  # 0.1% commission
        slippage=0.001     # 0.1% slippage
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
