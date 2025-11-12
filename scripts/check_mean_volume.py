import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(".."))

# Charger les données
data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
print(f"Chargement des données depuis {data_path}...")
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Calculer le volume moyen
mean_volume = data['volume'].mean()

print(f"\n=== Statistiques de Volume ===")
print(f"Volume moyen: {mean_volume:.2f}")
print(f"Volume médian: {data['volume'].median():.2f}")
print(f"Volume min: {data['volume'].min():.2f}")
print(f"Volume max: {data['volume'].max():.2f}")
print(f"Écart-type: {data['volume'].std():.2f}")

# Calculer la volatilité réalisée moyenne aussi
vol_window = 20
returns = np.diff(np.log(data['close'].values))
realized_vols = []

for i in range(vol_window, len(returns)):
    window_returns = returns[i-vol_window:i]
    realized_vols.append(np.std(window_returns))

mean_realized_vol = np.mean(realized_vols)

print(f"\n=== Statistiques de Volatilité Réalisée ===")
print(f"Volatilité réalisée moyenne: {mean_realized_vol:.6f}")
print(f"Volatilité réalisée médiane: {np.median(realized_vols):.6f}")
print(f"Volatilité réalisée min: {np.min(realized_vols):.6f}")
print(f"Volatilité réalisée max: {np.max(realized_vols):.6f}")

# Estimer l'impact pour différentes quantités
print(f"\n=== Estimation d'Impact (lambda_0=0.1, alpha=0.5) ===")
lambda_0 = 0.0005
alpha = 0.5
test_quantities = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0]  # En unités de mean_volume

for q_factor in test_quantities:
    quantity = q_factor * mean_volume
    vol_factor = 1 + (mean_realized_vol / mean_realized_vol)  # = 2
    quantity_factor = (quantity / mean_volume) ** alpha
    temp_impact = lambda_0 * vol_factor * quantity_factor
    
    print(f"Quantité = {q_factor:.2f} × mean_volume → Impact temporaire ≈ {temp_impact:.4f}")

