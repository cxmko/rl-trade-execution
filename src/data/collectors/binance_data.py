import os
import time
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from typing import Optional, List, Tuple


class BinanceDataCollector:
    """
    Class to collect historical BTCUSDT data from Binance
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the Binance client
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self.client = Client(api_key, api_secret)
        
    def get_historical_klines(self, 
                             symbol: str = 'BTCUSDT',
                             interval: str = '1m',
                             start_date: str = '2023-01-01',
                             end_date: str = '2024-12-31') -> pd.DataFrame:
        """
        Download historical kline data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            
        Returns:
            DataFrame containing the historical data
        """
        print(f"Downloading {symbol} data from {start_date} to {end_date}...")
        
        # Convert dates to milliseconds timestamp
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Initialize an empty list to store kline data
        all_klines = []
        
        # Binance API has limit on data points per request
        # We'll fetch in chunks of 1000 klines (approximately 16.6 hours for 1m data)
        current_ts = start_ts
        
        while current_ts < end_ts:
            # Calculate chunk end timestamp (1000 candles forward or end_ts)
            chunk_end_ts = min(current_ts + (1000 * 60 * 1000), end_ts)  # 1000 minutes in ms
            
            try:
                klines = self.client.get_historical_klines(
                    symbol=symbol, 
                    interval=interval,
                    start_str=current_ts,
                    end_str=chunk_end_ts,
                    limit=1000
                )
                all_klines.extend(klines)
                
                # Update current timestamp for next iteration
                current_ts = chunk_end_ts
                
                # Respect API rate limits
                time.sleep(1)
                print(f"Downloaded data until {datetime.fromtimestamp(chunk_end_ts/1000)}")
                
            except Exception as e:
                print(f"Error downloading data: {e}")
                # Wait longer on error
                time.sleep(5)
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                   
        df = pd.DataFrame(all_klines, columns=columns)
        
        # Process the timestamp columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert string values to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def download_and_save_data(self,
                              symbol: str = 'BTCUSDT',
                              interval: str = '1m',
                              save_path: str = '../../data/raw',
                              train_period: Tuple[str, str] = ('2023-01-01', '2023-12-31'),
                              test_period: Tuple[str, str] = ('2024-01-01', '2024-12-31')) -> None:
        """
        Download and save training and testing datasets
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            save_path: Path to save the data
            train_period: Tuple of (start_date, end_date) for training data
            test_period: Tuple of (start_date, end_date) for testing data
        """
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Download training data
        train_start, train_end = train_period
        train_df = self.get_historical_klines(symbol, interval, train_start, train_end)
        train_filename = f"{save_path}/{symbol}_{interval}_train_{train_start}_to_{train_end}.csv"
        train_df.to_csv(train_filename)
        print(f"Training data saved to {train_filename}")
        
        # Download testing data
        test_start, test_end = test_period
        test_df = self.get_historical_klines(symbol, interval, test_start, test_end)
        test_filename = f"{save_path}/{symbol}_{interval}_test_{test_start}_to_{test_end}.csv"
        test_df.to_csv(test_filename)
        print(f"Testing data saved to {test_filename}")


if __name__ == "__main__":
    # Example usage
    collector = BinanceDataCollector()
    collector.download_and_save_data()