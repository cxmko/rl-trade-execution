import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# Set style for academic report
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def load_data():
    """Load and preprocess the 2023 and 2024 datasets."""
    print("Loading data...")
    df_2023 = pd.read_csv('../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv')
    df_2024 = pd.read_csv('../data/raw/BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv')
    
    # Convert timestamps
    df_2023['timestamp'] = pd.to_datetime(df_2023['timestamp'])
    df_2024['timestamp'] = pd.to_datetime(df_2024['timestamp'])
    
    df_2023.set_index('timestamp', inplace=True)
    df_2024.set_index('timestamp', inplace=True)
    
    return df_2023, df_2024

def calculate_market_stats(df, name):
    """Calculate key market statistics."""
    # Resample to 4H for relevant execution horizon stats
    df_4h = df['close'].resample('4H').last()
    returns = df_4h.pct_change().dropna()
    
    total_ret = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    annual_vol = returns.std() * np.sqrt(365 * 6) # 6 periods of 4H per day
    max_dd = ((df['close'] / df['close'].cummax()) - 1).min()
    kurt = stats.kurtosis(returns)
    
    return {
        'name': name,
        'total_return': total_ret,
        'volatility': annual_vol,
        'max_drawdown': max_dd,
        'kurtosis': kurt,
        'returns': returns
    }

def plot_comparative_trends(df_2023, df_2024):
    """
    Figure 1: Normalized Cumulative Return Comparison
    Shows the trend difference clearly by starting both at 0%.
    """
    print("Generating Comparative Trend Plot...")
    
    # Normalize to percentage change from start of year
    norm_23 = (df_2023['close'] / df_2023['close'].iloc[0] - 1) * 100
    norm_24 = (df_2024['close'] / df_2024['close'].iloc[0] - 1) * 100
    
    stats_23 = calculate_market_stats(df_2023, "2023")
    stats_24 = calculate_market_stats(df_2024, "2024")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Lines
    ax.plot(norm_23.index, norm_23, color='#2c3e50', label='2023 (Train)', linewidth=1.5, alpha=0.8)
    ax.plot(norm_24.index, norm_24, color='#e74c3c', label='2024 (Test)', linewidth=1.5, alpha=0.9)
    
    # Add Mean/Final markers
    ax.text(norm_23.index[-1], norm_23.iloc[-1], f" +{stats_23['total_return']:.1%}", 
            color='#2c3e50', fontweight='bold', va='center')
    ax.text(norm_24.index[-1], norm_24.iloc[-1], f" +{stats_24['total_return']:.1%}", 
            color='#e74c3c', fontweight='bold', va='center')
    
    # Add Stats Box
    stats_text = (
        f"2023 Stats:\n"
        f"• Volatility: {stats_23['volatility']:.1%}\n"
        f"• Max DD: {stats_23['max_drawdown']:.1%}\n\n"
        f"2024 Stats:\n"
        f"• Volatility: {stats_24['volatility']:.1%}\n"
        f"• Max DD: {stats_24['max_drawdown']:.1%}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_title('Cumulative Return Comparison: 2023 vs 2024', fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../report/sample/data_regime_split.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_return_distributions(df_2023, df_2024):
    """
    Figure 2: Distribution of 4-Hour Returns
    Shows the "Fat Tails" and Kurtosis difference.
    """
    print("Generating Return Distribution Plot...")
    
    stats_23 = calculate_market_stats(df_2023, "2023")
    stats_24 = calculate_market_stats(df_2024, "2024")
    
    r23 = stats_23['returns'] * 100 # Convert to %
    r24 = stats_24['returns'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # KDE Plots (Kernel Density Estimate)
    sns.kdeplot(r23, color='#2c3e50', fill=True, alpha=0.3, label=f"2023 (Kurtosis: {stats_23['kurtosis']:.2f})", ax=ax)
    sns.kdeplot(r24, color='#e74c3c', fill=True, alpha=0.3, label=f"2024 (Kurtosis: {stats_24['kurtosis']:.2f})", ax=ax)
    
    ax.set_title('Distribution of 4-Hour Returns (Execution Horizon)', fontweight='bold')
    ax.set_xlabel('4-Hour Return (%)')
    ax.set_ylabel('Density')
    ax.set_xlim(-5, 5) # Focus on the core distribution
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../report/sample/data_return_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_volatility_density(df_2023, df_2024):
    """
    Figure 3: Volatility Regime Density
    Instead of a time series, show the DISTRIBUTION of volatility.
    """
    print("Generating Volatility Density Plot...")
    
    window = 240 # 4 hours
    
    # Calculate rolling volatility (annualized)
    vol_23 = df_2023['close'].pct_change().rolling(window).std() * np.sqrt(365*24*60)
    vol_24 = df_2024['close'].pct_change().rolling(window).std() * np.sqrt(365*24*60)
    
    # Drop NaNs
    vol_23 = vol_23.dropna()
    vol_24 = vol_24.dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.kdeplot(vol_23, color='#2c3e50', fill=True, alpha=0.3, label=f"2023 Mean Vol: {vol_23.mean():.1%}", ax=ax)
    sns.kdeplot(vol_24, color='#e74c3c', fill=True, alpha=0.3, label=f"2024 Mean Vol: {vol_24.mean():.1%}", ax=ax)
    
    ax.set_title('Distribution of Realized Volatility Regimes', fontweight='bold')
    ax.set_xlabel('Annualized Rolling Volatility')
    ax.set_ylabel('Frequency (Density)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../report/sample/data_volatility_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('../report/sample', exist_ok=True)
    
    df23, df24 = load_data()
    
    plot_comparative_trends(df23, df24)
    plot_return_distributions(df23, df24)
    plot_volatility_density(df23, df24)