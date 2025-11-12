import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """
    Class to engineer features for the reinforcement learning model
    """
    def __init__(self):
        pass
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Calculate local volatility (High - Low)
        result['volatility'] = result['high'] - result['low']
        
        # Calculate MACD
        result['ema12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema26'] = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = result['ema12'] - result['ema26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate candle position
        result['candle_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
        
        # Calculate time features (assuming index is datetime)
        # Convert hour of day to sin and cos for cyclical feature
        if isinstance(result.index, pd.DatetimeIndex):
            hours = result.index.hour + result.index.minute/60
            result['time_sin'] = np.sin(2 * np.pi * hours / 24.0)
            result['time_cos'] = np.cos(2 * np.pi * hours / 24.0)
        
        return result
    
    def normalize_features(self, train_df: pd.DataFrame, 
                         test_df: Optional[pd.DataFrame] = None, 
                         features: Optional[List[str]] = None) -> tuple:
        """
        Normalize features using min-max scaling based on training data
        
        Args:
            train_df: Training dataframe
            test_df: Testing dataframe (optional)
            features: List of features to normalize (if None, normalize all numeric features)
            
        Returns:
            Tuple of (normalized_train_df, normalized_test_df)
        """
        # If no features specified, use all numeric columns
        if features is None:
            features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Make copies to avoid modifying originals
        train_normalized = train_df.copy()
        test_normalized = test_df.copy() if test_df is not None else None
        
        # Store normalization parameters
        self.normalization_params = {}
        
        # Normalize each feature
        for feature in features:
            if feature in train_df.columns:
                # Get min and max from training data only
                min_val = train_df[feature].min()
                max_val = train_df[feature].max()
                
                # Store the parameters for this feature
                self.normalization_params[feature] = {'min': min_val, 'max': max_val}
                
                # Skip normalization if max equals min (constant feature)
                if max_val == min_val:
                    continue
                
                # Normalize training data
                train_normalized[feature] = 1e-6+(train_df[feature] - min_val) / (max_val - min_val)
                
                # Normalize test data using same parameters if provided
                if test_normalized is not None and feature in test_normalized:
                    test_normalized[feature] = (test_df[feature] - min_val) / (max_val - min_val)
        
        if test_normalized is not None:
            return train_normalized, test_normalized
        else:
            return train_normalized
    
    def process_data(self, train_path: str, test_path: str) -> tuple:
        """
        Load, engineer features, and normalize data
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to testing data CSV
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        # Load data
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Add technical indicators
        train_df = self.add_technical_indicators(train_df)
        test_df = self.add_technical_indicators(test_df)
        
        # Normalize features
        train_normalized, test_normalized = self.normalize_features(
            train_df, test_df
        )
        
        return train_normalized, test_normalized


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    # Replace with actual paths to your data
    # train_df, test_df = engineer.process_data(
    #     "../../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv",
    #     "../../data/raw/BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv"
    # )