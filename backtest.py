
import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional
import pandas as pd


try:
    from alpaca_trade_api import REST
except ImportError:
    REST = None


class Backtest:
    """
    Backtesting engine that uses Alpaca historical data.
    
    Example usage:
        def my_strategy(df, i, positions):
            # Simple moving average crossover
            if i < 20:
                return None
            sma_short = df['close'].iloc[i-5:i].mean()
            sma_long = df['close'].iloc[i-20:i].mean()
            if sma_short > sma_long and not positions:
                return ('buy', 10)
            elif sma_short < sma_long and positions:
                return ('sell', 10)
            return None
        
        bt = Backtest('AAPL', '2023-01-01', '2023-12-31')
        results = bt.run(my_strategy, initial_capital=10000)
        print(results)
    """
    
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets"
    ):
        """
        Initialize the backtester.
        
        Args:
            symbol: Stock symbol to backtest (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            api_key: Alpaca API key (defaults to env var)
            secret_key: Alpaca secret key (defaults to env var)
            base_url: Alpaca API base URL
        """
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        
        # Use provided keys or fall back to environment variables
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = base_url
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials missing. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

        if REST is None:
            raise ImportError("alpaca-trade-api is required for backtesting")
        
        self.api = REST(self.api_key, self.secret_key, base_url=self.base_url, api_version='v2')
        self.data: Optional[pd.DataFrame] = None
        
    def fetch_data(self, timeframe: str = "1Day") -> pd.DataFrame:
        """
        Fetch historical bar data from Alpaca.
        
        Args:
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            bars = self.api.get_bars(
                self.symbol,
                timeframe,
                start=self.start_date,
                end=self.end_date,
                adjustment='all',
                feed='iex'
            ).df
            
            if bars.empty:
                raise ValueError(f"No data found for {self.symbol} in the given date range")
            
            # Flatten multi-index if present
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=0, drop=True)
            
            self.data = bars.reset_index()
            self.data.columns = [c.lower() for c in self.data.columns]
            
            # Save to JSON
            try:
                if not os.path.exists('data'):
                    os.makedirs('data')
                
                filename = f"data/{self.symbol}_{self.start_date}_{self.end_date}.json"
                # Store dates as strings for JSON compatibility
                self.data.to_json(filename, orient='records', date_format='iso')
                print(f"[Backtest] Data saved to {filename}")
            except Exception as e:
                print(f"[Backtest] Warning: Could not save data to JSON: {e}")
                
            return self.data
            
        except Exception as e:
            print(f"[Backtest] Error fetching data: {e}")
            raise
    
    def run(
        self,
        strategy: Callable[[pd.DataFrame, int, List], Optional[tuple]],
        initial_capital: float = 10000.0,
        commission: float = 0.0
    ) -> Dict:
        """
        Run the backtest with the given strategy.
        
        Args:
            strategy: A callable that takes (dataframe, current_index, current_positions)
                     and returns either None (no action) or ('buy'/'sell', quantity)
            initial_capital: Starting cash amount
            commission: Commission per trade (flat fee)
            
        Returns:
            Dictionary with backtest results
        """
        if self.data is None:
            self.fetch_data()
        
        df = self.data
        cash = initial_capital
        positions = []  # List of (qty, buy_price) tuples
        trades = []
        portfolio_values = []
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
            
            # Calculate current portfolio value
            holdings_value = sum(qty * current_price for qty, _ in positions)
            portfolio_value = cash + holdings_value
            portfolio_values.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'holdings_value': holdings_value,
                'price': current_price
            })
            
            # Get strategy signal
            signal = strategy(df, i, positions)
            
            if signal is not None:
                action, qty = signal
                
                if action == 'buy' and cash >= current_price * qty + commission:
                    cost = current_price * qty + commission
                    cash -= cost
                    positions.append((qty, current_price))
                    trades.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'price': current_price,
                        'qty': qty,
                        'cost': cost
                    })
                    
                elif action == 'sell' and positions:
                    # Sell FIFO
                    sell_qty = min(qty, sum(q for q, _ in positions))
                    remaining = sell_qty
                    profit = 0
                    new_positions = []
                    
                    for pos_qty, buy_price in positions:
                        if remaining <= 0:
                            new_positions.append((pos_qty, buy_price))
                        elif pos_qty <= remaining:
                            profit += (current_price - buy_price) * pos_qty
                            remaining -= pos_qty
                        else:
                            profit += (current_price - buy_price) * remaining
                            new_positions.append((pos_qty - remaining, buy_price))
                            remaining = 0
                    
                    positions = new_positions
                    revenue = current_price * sell_qty - commission
                    cash += revenue
                    trades.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'price': current_price,
                        'qty': sell_qty,
                        'revenue': revenue,
                        'profit': profit
                    })
        
        # Final portfolio value
        final_price = df['close'].iloc[-1]
        final_holdings = sum(qty * final_price for qty, _ in positions)
        final_value = cash + final_holdings
        
        # Calculate metrics
        total_return = (final_value - initial_capital) / initial_capital * 100
        num_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
        
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': initial_capital,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'win_rate': round(winning_trades / num_trades * 100, 2) if num_trades > 0 else 0,
            'trades': trades,
            'portfolio_history': portfolio_values
        }


# ----- Pre-built strategies -----

def sma_crossover_strategy(short_period: int = 10, long_period: int = 30):
    """
    Returns a Simple Moving Average crossover strategy function.
    Buys when short SMA crosses above long SMA, sells when it crosses below.
    """
    prev_short = None
    prev_long = None
    
    def strategy(df: pd.DataFrame, i: int, positions: List) -> Optional[tuple]:
        nonlocal prev_short, prev_long
        
        if i < long_period:
            return None
        
        short_sma = df['close'].iloc[i - short_period:i].mean()
        long_sma = df['close'].iloc[i - long_period:i].mean()
        
        signal = None
        
        # Crossover detection
        if prev_short is not None and prev_long is not None:
            # Golden cross (buy signal)
            if prev_short <= prev_long and short_sma > long_sma and not positions:
                signal = ('buy', 10)
            # Death cross (sell signal)
            elif prev_short >= prev_long and short_sma < long_sma and positions:
                total_qty = sum(q for q, _ in positions)
                signal = ('sell', total_qty)
        
        prev_short = short_sma
        prev_long = long_sma
        
        return signal
    
    return strategy


def rsi_strategy(period: int = 14, oversold: float = 30, overbought: float = 70):
    """
    Returns an RSI-based strategy function.
    Buys when RSI drops below oversold level, sells when it rises above overbought.
    """
    def strategy(df: pd.DataFrame, i: int, positions: List) -> Optional[tuple]:
        if i < period + 1:
            return None
        
        # Calculate RSI
        deltas = df['close'].diff().iloc[i - period:i]
        gains = deltas.where(deltas > 0, 0).mean()
        losses = (-deltas.where(deltas < 0, 0)).mean()
        
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        
        # Trading signals
        if rsi < oversold and not positions:
            return ('buy', 10)
        elif rsi > overbought and positions:
            total_qty = sum(q for q, _ in positions)
            return ('sell', total_qty)
        
        return None
    
    return strategy
