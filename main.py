from src.data.collectors.binance_data import BinanceDataCollector
import os




def download_data():
    """Download market data from Binance"""
    collector = BinanceDataCollector()
    
    # Download BTCUSDT data for 2023 (training) and 2024 (testing)
    collector.download_and_save_data(
        symbol='BTCUSDT',
        interval='1m',
        save_path='data/raw',
        train_period=('2023-01-01', '2023-12-31'),
        test_period=('2024-01-01', '2024-12-31')
    )



if __name__ == "__main__":
    
    print("\nDownloading data...")
    download_data()
    



    