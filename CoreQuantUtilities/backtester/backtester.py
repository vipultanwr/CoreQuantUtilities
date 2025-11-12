# -*- coding: utf-8 -*-
"""
Created on Sun Nov 2 21:27:21 2025

@author: Vipul Tanwar

A Backtrader-based backtesting engine that mirrors the interface of the custom backtester.
"""

import backtrader as bt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class PandasDataWithSignal(bt.feeds.PandasData):
    """
    Custom data feed that includes a 'signal' line from the DataFrame.
    """
    lines = ('signal',)
    # Map the 'signal' line to a column in the DataFrame.
    # -1 indicates the last column, but we will pass the column name via params.
    params = (('signal', 'signal'),)

class SignalStrategy(bt.Strategy):
    """
    A generic strategy that trades based on an external signal column.
    Signal: 1 for Buy, -1 for Sell, 0 for Hold.
    """
    params = (
        ('holding_period', 1), # Default holding period of 1 bar (trade on next bar)
    )

    def __init__(self):
        self.signal = self.datas[0].lines.signal
        self.order = None
        self.entry_bar = 0 # Bar number of the last position entry

    def notify_order(self, order):
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None # Reset order status so we can place another one

    def next(self):
        # Check if an order is pending ... if so, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in a position and if the holding period has passed
        if self.position and (len(self) < self.entry_bar + self.p.holding_period):
            return # Still in holding period, do nothing
        if self.order:
            return

        current_signal = self.signal[0]

        if current_signal == 1:  # Buy Signal
            # If we are not in a long position, buy.
            # This will also close a short position and open a long one.
            if self.position.size <= 0:
                self.entry_bar = len(self) # Record entry bar
                self.order = self.buy()

        elif current_signal == -1:  # Sell Signal
            # If we are not in a short position, sell.
            # This will also close a long position and open a short one.
            if self.position.size >= 0:
                self.entry_bar = len(self) # Record entry bar
                self.order = self.sell()

class StrategyBacktester:
    def __init__(self, commission=0.001, slippage=0.001, time_horizon='1d', initial_cash=100000.0, holding_period=1, sizing_percent=98):
        """
        Initialize the backtrader wrapper.

        Parameters:
        commission (float): Commission rate for trades (e.g., 0.001 for 0.1%).
        slippage (float): Slippage per trade (e.g., 0.001 for 0.1%).
        time_horizon (str): The data frequency for annualization.
        initial_cash (float): Starting portfolio value.
        holding_period (int): Minimum number of bars to hold a position.
        sizing_percent (int): Percentage of portfolio to use for trades (e.g., 98 for 98%).
        """
        self.commission = commission
        self.slippage = slippage
        self.time_horizon = time_horizon
        self.holding_period = holding_period
        self.sizing_percent = sizing_percent
        self.initial_cash = initial_cash
        self.results = None
        self.cerebro = None
        self.run_info = None

    def _get_periods_per_year(self, time_horizon):
        """Maps time horizon string to periods per year."""
        TRADING_DAYS_PER_YEAR = 252
        TRADING_HOURS_PER_DAY = 6.5

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
        return horizon_map.get(time_horizon, TRADING_DAYS_PER_YEAR)

    def backtest(self, df, price_col='close', signal_col='Signals', date_col='date'):
        """
        Run the backtest using backtrader.

        Parameters:
        df (pd.DataFrame): DataFrame with OHLCV data and signals.
        price_col (str): Not used by backtrader directly, but kept for interface consistency.
        signal_col (str): Column name for signals (1=buy, -1=sell, 0=hold).
        date_col (str): Column name for date/datetime data.
        """
        data = df.copy()
        data.columns = [col.lower() for col in data.columns]
        signal_col = signal_col.lower()
        date_col = date_col.lower()

        # Prepare data for backtrader
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col)

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume', signal_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Input DataFrame must contain column: {col}")

        # Create a cerebro instance
        self.cerebro = bt.Cerebro()

        # Add data feed
        # Use the custom data feed and map the signal_col to its 'signal' line
        data_feed = PandasDataWithSignal(dataname=data, datetime=None, signal=signal_col)
        self.cerebro.adddata(data_feed)

        # Add strategy
        # The strategy now knows to look for the 'signal' line, so no param is needed.
        self.cerebro.addstrategy(SignalStrategy, holding_period=self.holding_period)

        # Set initial capital and broker settings
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        # Backtrader's slippage is % based, so 0.1% is 0.1
        self.cerebro.broker.set_slippage_perc(perc=self.slippage * 100)
        self.cerebro.addsizer(bt.sizers.AllInSizer, percents=self.sizing_percent)

        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1, factor=self._get_periods_per_year(self.time_horizon), annualize=True)
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.NoTimeFrame)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

        # Only add PyFolio analyzer if there are potential trades to avoid errors on no-trade backtests
        has_trading_signals = (data[signal_col] != 0).any()
        if has_trading_signals:
            self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        # Run the backtest
        self.run_info = self.cerebro.run()
        
        # Extract results for compatibility, handling the case where PyFolio was not added
        if has_trading_signals:
            pyfolio_analyzer = self.run_info[0].analyzers.getbyname('pyfolio')
            returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
        else:
            # Create empty structures if no trades were possible
            returns = pd.Series(0.0, index=data.index)
            positions = pd.DataFrame(columns=['amount', 'value'])
            transactions = pd.DataFrame(columns=['amount', 'price', 'symbol', 'value'])

        self.results = pd.DataFrame(index=returns.index)
        self.results['returns'] = returns
        self.results['portfolio_value'] = (1 + returns).cumprod() * self.initial_cash
        self.results['cumulative_returns'] = self.results['portfolio_value'] / self.initial_cash - 1
        
        # Check if positions DataFrame is not empty AND has the 'amount' column
        if not positions.empty and 'amount' in positions.columns:
            self.results['position'] = positions['amount'].reindex(self.results.index, fill_value=0)
        else:
            self.results['position'] = 0
        
        self.trades = transactions

        return self.results

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics from backtrader analyzers."""
        if not self.run_info:
            raise ValueError("No backtest results available. Run backtest() first.")

        analyzers = self.run_info[0].analyzers
        trade_analyzer = analyzers.tradeanalyzer.get_analysis()
        
        periods_per_year = self._get_periods_per_year(self.time_horizon)
        
        total_return = (self.cerebro.broker.getvalue() / self.initial_cash) - 1
        # Manually calculate annualized return for the given time horizon
        annualized_return = (1 + total_return) ** (periods_per_year / len(self.results)) - 1 if len(self.results) > 0 else 0
        
        win_rate = trade_analyzer.won.total / trade_analyzer.total.total if 'won' in trade_analyzer and trade_analyzer.total.total > 0 else 0
        avg_win = trade_analyzer.won.pnl.average if 'won' in trade_analyzer and trade_analyzer.won.total > 0 else 0
        avg_loss = trade_analyzer.lost.pnl.average if 'lost' in trade_analyzer and trade_analyzer.lost.total > 0 else 0
        profit_factor = abs(trade_analyzer.won.pnl.total / trade_analyzer.lost.pnl.total) if 'won' in trade_analyzer and 'lost' in trade_analyzer and trade_analyzer.lost.pnl.total != 0 else float('inf')

        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': "N/A in bt", # PyFolio can calculate this
            'Sharpe Ratio': f"{analyzers.sharpe.get_analysis().get('sharperatio', 0):.2f}",
            'Sortino Ratio': "N/A in bt", # PyFolio can calculate this
            'Calmar Ratio': "N/A in bt", # PyFolio can calculate this
            'Maximum Drawdown': f"{analyzers.drawdown.get_analysis().max.drawdown / 100:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Trades': trade_analyzer.total.total,
            'Profit Factor': f"{profit_factor:.2f}",
            'Average Win': f"{avg_win:.2f}",
            'Average Loss': f"{avg_loss:.2f}"
        }
        return metrics

    def plot_results(self, **kwargs):
        """Plot results using backtrader's plotting feature."""
        if not self.cerebro:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        if self.trades.empty:
            print("\nPlotting skipped: No trades were executed during the backtest.")
            return
        self.cerebro.plot(**kwargs)

    def print_metrics(self):
        """Print performance metrics in a formatted table."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print("BACKTRADER STRATEGY PERFORMANCE METRICS")
        print("="*50)
        
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        
        print("\n" + "="*50)


# Example usage
def run_backtest_example():
    """Example of how to use the BacktraderWrapper."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': close_prices - np.random.randn(n) * 0.25,
        'high': close_prices + np.abs(np.random.randn(n) * 0.5),
        'low': close_prices - np.abs(np.random.randn(n) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, n),
        'Signals': np.random.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1])
    })
    
    bt_backtester = StrategyBacktester(
        commission=0.001,
        slippage=0.001,
        time_horizon='1d'
    )
    
    results = bt_backtester.backtest(sample_data, signal_col='Signals')
    
    bt_backtester.print_metrics()
    bt_backtester.plot_results(style='candlestick')

if __name__ == '__main__':
    run_backtest_example()