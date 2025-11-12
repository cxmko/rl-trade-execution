#from src.data.collectors.binance_data import BinanceDataCollector
from src.data.processors.feature_engineering import FeatureEngineer
import os


def setup_project():
    """Set up project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


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


def process_data():
    """Process and engineer features for the raw data"""
    engineer = FeatureEngineer()
    
    # Process the downloaded data
    train_df, test_df = engineer.process_data(
        "data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv",
        "data/raw/BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv"
    )
    
    # Save processed data
    train_df.to_csv("data/processed/BTCUSDT_1m_train_processed.csv")
    test_df.to_csv("data/processed/BTCUSDT_1m_test_processed.csv")
    
    print("Data processing complete. Processed files saved to data/processed/")


if __name__ == "__main__":
    print("Setting up project structure...")
    #setup_project()
    
    print("\nDownloading data...")
    #download_data()
    
    print("\nProcessing data...")
    process_data()


    