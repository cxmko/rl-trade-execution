import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(".."))

# --- CONFIGURATION ---
DATA_PATH = "../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv"

VOL_WINDOW = 5       # Fast signal
AVG_WINDOW = 60      # Slow baseline
MAX_IMPACT_CAP = 0.05   # 5% Hard cap
LAMBDA_CANDIDATES = [0.002, 0.003, 0.004, 0.005, 0.01]

def prepare_data_metrics(df):
    """Compute environment metrics on the full dataset."""
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    print(f"📊 Loaded RAW data. Rows: {len(df)}")
    print("📊 Computing rolling metrics...")
    
    # 1. Calculate Returns
    df['returns'] = df['close'].pct_change()
    
    # 2. Realized Volatility (5m window)
    df['realized_vol'] = df['returns'].rolling(window=VOL_WINDOW).std().fillna(0)
    
    # 3. Rolling Sigma (60m baseline of realized vol)
    df['rolling_sigma'] = df['realized_vol'].rolling(window=AVG_WINDOW).mean().fillna(0)
    
    # 4. Rolling Volume (60m mean)
    df['rolling_volume'] = df['volume'].rolling(window=AVG_WINDOW).mean().fillna(0)
    
    # Drop NaNs from warmup
    df = df.dropna()
    return df

def analyze_min_metrics(df):
    """Finds data-driven floors for Volume and Sigma."""
    print("\n🔍 ANALYZING DATA FLOORS...")
    
    # --- VOLUME ---
    # Filter strictly positive to find real floor
    pos_vol = df['rolling_volume'][df['rolling_volume'] > 0.0]
    min_vol = pos_vol.min()
    p01_vol = pos_vol.quantile(0.01)
    
    print(f"   📉 Volume (60m avg): Min Observed = {min_vol:.4f} BTC | 1st Percentile = {p01_vol:.4f} BTC")
    
    # --- SIGMA ---
    # Filter strictly positive
    pos_sigma = df['rolling_sigma'][df['rolling_sigma'] > 0.0]
    min_sigma = pos_sigma.min()
    p01_sigma = pos_sigma.quantile(0.01)
    
    print(f"   📉 Sigma  (60m avg): Min Observed = {min_sigma:.8f}     | 1st Percentile = {p01_sigma:.8f}")
    
    return min_vol, min_sigma

def get_regime_samples(df):
    """Extracts representative rows for 9 market regimes + Extreme Edge Cases."""
    print("\n🔍 Extracting extreme market scenarios...")
    
    # Define Percentiles
    vol_low = df['realized_vol'].quantile(0.01)
    vol_high = df['realized_vol'].quantile(0.99)
    
    liq_low = df['rolling_volume'].quantile(0.01)
    liq_high = df['rolling_volume'].quantile(0.99)
    
    scenarios = []
    
    # --- 1. THE 9 REGIMES MATRIX ---
    vol_levels = {
        'Low Vol': (df['realized_vol'] <= vol_low),
        'Med Vol': (df['realized_vol'] > vol_low) & (df['realized_vol'] < vol_high),
        'High Vol': (df['realized_vol'] >= vol_high)
    }
    
    liq_levels = {
        'Low Liq': (df['rolling_volume'] <= liq_low),
        'Med Liq': (df['rolling_volume'] > liq_low) & (df['rolling_volume'] < liq_high),
        'High Liq': (df['rolling_volume'] >= liq_high)
    }
    
    for v_name, v_mask in vol_levels.items():
        for l_name, l_mask in liq_levels.items():
            mask = v_mask & l_mask
            subset = df[mask]
            
            if not subset.empty:
                # Pick extreme point
                if 'High' in v_name:
                    sample = subset.sort_values('realized_vol', ascending=False).iloc[0]
                elif 'Low' in l_name:
                    sample = subset.sort_values('rolling_volume', ascending=True).iloc[0]
                else:
                    sample = subset.sample(1, random_state=42).iloc[0]
                
                scenarios.append({
                    'Type': 'Regime',
                    'Name': f"{v_name} / {l_name}",
                    'realized_vol': sample['realized_vol'],
                    'rolling_sigma': sample['rolling_sigma'],
                    'rolling_volume': sample['rolling_volume']
                })

    # --- 2. EXTREME EDGE CASES ---
    
    # Absolute Min Volume
    pos_vol = df[df['rolling_volume'] > 0.0001]
    if not pos_vol.empty:
        min_vol_row = df.loc[pos_vol['rolling_volume'].idxmin()]
        scenarios.append({
            'Type': 'Edge',
            'Name': '💀 ABSOLUTE MIN LIQUIDITY',
            'realized_vol': min_vol_row['realized_vol'],
            'rolling_sigma': min_vol_row['rolling_sigma'],
            'rolling_volume': min_vol_row['rolling_volume']
        })
        
    # Absolute Min Sigma (The "Flatline" Market)
    pos_sigma = df[df['rolling_sigma'] > 1e-9]
    if not pos_sigma.empty:
        min_sigma_row = df.loc[pos_sigma['rolling_sigma'].idxmin()]
        scenarios.append({
            'Type': 'Edge',
            'Name': '💤 ABSOLUTE MIN VOLATILITY',
            'realized_vol': min_sigma_row['realized_vol'],
            'rolling_sigma': min_sigma_row['rolling_sigma'],
            'rolling_volume': min_sigma_row['rolling_volume']
        })
    
    # Max Volatility Spike
    df['vol_ratio'] = df['realized_vol'] / (df['rolling_sigma'] + 1e-9)
    max_spike_row = df.loc[df['vol_ratio'].idxmax()]
    scenarios.append({
        'Type': 'Edge',
        'Name': '⚡ MAX VOLATILITY SHOCK',
        'realized_vol': max_spike_row['realized_vol'],
        'rolling_sigma': max_spike_row['rolling_sigma'],
        'rolling_volume': max_spike_row['rolling_volume']
    })

    return scenarios

def calculate_impact(scenario, quantity, lambda_0, min_vol, min_sigma):
    """Replicates the environment's impact logic exactly."""
    
    # 1. Safety Clamps (Using Data-Driven Floors)
    eff_sigma = max(scenario['rolling_sigma'], min_sigma)
    eff_volume = max(scenario['rolling_volume'], min_vol)
    
    # 2. Factors
    vol_factor = 1 + (scenario['realized_vol'] / eff_sigma)
    quantity_factor = (quantity / eff_volume) ** 0.5
    
    # 3. Raw Impact
    raw_impact = lambda_0 * vol_factor * quantity_factor
    
    # 4. Cap
    clipped_impact = min(raw_impact, MAX_IMPACT_CAP)
    
    return clipped_impact * 10000 # Return in bps

def run_test():
    # Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find file at {DATA_PATH}")
        return

    df = prepare_data_metrics(df)
    
    # Analyze Floors
    real_min_vol, real_min_sigma = analyze_min_metrics(df)
    
    # Set Safety Floors (Slightly buffered or raw min)
    # For volume, we decided 0.01 is safe for BTC.
    # For sigma, we use the real min or 1e-6.
    TEST_MIN_VOL = max(real_min_vol, 0.01)
    TEST_MIN_SIGMA = max(real_min_sigma, 1e-7) 
    
    # Get Scenarios
    scenarios = get_regime_samples(df)
    
    # Quantities to test
    quantities = [1.0, 100.0] 
    
    print(f"\n{'='*100}")
    print(f"TESTING REAL DATA SCENARIOS (RAW DATA)")
    print(f"Min Rolling Vol Floor:   {TEST_MIN_VOL:.4f}")
    print(f"Min Rolling Sigma Floor: {TEST_MIN_SIGMA:.8f}")
    print(f"Max Impact Cap:          {MAX_IMPACT_CAP*100}%")
    print(f"{'='*100}\n")

    for qty in quantities:
        print(f"\n📦 ORDER SIZE: {qty} BTC")
        print(f"{'-'*100}")
        
        # Header
        header = f"{'Scenario':<30} | {'Vol':<8} | {'Sigma':<9} | {'Liq':<8} |"
        for lam in LAMBDA_CANDIDATES:
            header += f" λ={lam:<5} |"
        print(header)
        print(f"{'-'*100}")
        
        for sc in scenarios:
            row_str = f"{sc['Name']:<30} | {sc['realized_vol']*100:<7.2f}% | {sc['rolling_sigma']:.7f} | {sc['rolling_volume']:<8.2f} |"
            
            for lam in LAMBDA_CANDIDATES:
                bps = calculate_impact(sc, qty, lam, TEST_MIN_VOL, TEST_MIN_SIGMA)
                
                # Color coding
                marker = ""
                if bps >= MAX_IMPACT_CAP * 10000: marker = "🛑" 
                elif bps > 100: marker = "🔥" 
                elif bps < 1: marker = "❄️" 
                
                row_str += f" {bps:>6.1f} {marker:<2}|"
            
            print(row_str)

if __name__ == "__main__":
    run_test()