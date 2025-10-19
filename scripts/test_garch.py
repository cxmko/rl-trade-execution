import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ajouter le dossier parent au path pour importer les modules du projet
sys.path.append(os.path.abspath(".."))
from src.environment.garch_simulator import GarchCalibrator, GarchSimulator

def calculate_realized_volatility(prices, window=20):
    """
    Calcule la volatilité réalisée basée sur l'écart-type des rendements récents
    """
    # Convertir en array numpy si c'est une série
    if isinstance(prices, pd.Series):
        prices = prices.values
        
    # Calculer les rendements logarithmiques
    returns = np.diff(np.log(prices))
    
    # Initialiser le tableau de volatilités
    volatility = np.zeros(len(prices))
    
    # Pour chaque point, calculer la volatilité sur la fenêtre précédente
    for i in range(1, len(prices)):
        # Déterminer la taille de la fenêtre disponible
        window_size = min(window, i)
        
        if window_size > 0:
            # Extraire les rendements dans la fenêtre
            window_returns = returns[i-window_size:i]
            # Calculer l'écart-type
            volatility[i] = np.std(window_returns) if len(window_returns) > 0 else 0
    
    return volatility

def sample_and_visualize_garch():
    """Échantillonne et visualise les données simulées par le modèle GARCH"""
    
    # 1. Charger les données historiques
    print("Chargement des données historiques...")
    data_path = 'C:/Users/Cameron/OneDrive/Bureau/Ecole/IPP/DS/projet/rl-trade-execution/data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 2. Calibrer le modèle GARCH
    print("Calibration du modèle GARCH...")
    calibrator = GarchCalibrator()
    params = calibrator.fit(data['close'])
    
    # Obtenir les derniers prix et volatilité pour initialiser le simulateur
    initial_price = data['close'].iloc[-1]
    initial_volatility = np.sqrt(calibrator.results.conditional_volatility.iloc[-1]**2)
    
    # 3. Générer plusieurs trajectoires simulées
    n_simulations = 5  # Nombre de trajectoires
    n_steps = 1440     # Nombre de pas (1 jour de données minutes)
    
    print(f"Génération de {n_simulations} trajectoires de prix simulées...")
    simulated_paths = []
    simulated_vols = []
    real_paths = []     # Pour stocker les trajectoires réelles
    start_indices = []  # Pour stocker les indices de départ
    # Choisir un point de départ aléatoire dans les données d'entraînement
    # (en évitant le début pour avoir suffisamment d'historique pour la volatilité conditionnelle)
    # Éviter aussi la fin pour avoir suffisamment de données futures pour la trajectoire réelle
    random_idx = np.random.randint(100, len(data) - n_steps - 1)
    start_indices.append(random_idx)
    
    # Extraire prix et recalibrer le modèle GARCH autour de ce point
    random_price = data['close'].iloc[random_idx]
    
    # Calibrer le GARCH sur une fenêtre autour du point choisi
    window_start = max(0, random_idx - 5000)
    window_data = data.iloc[window_start:random_idx]
    
    local_calibrator = GarchCalibrator()
    local_params = local_calibrator.fit(window_data['close'])
    local_volatility = np.sqrt(local_calibrator.results.conditional_volatility.iloc[-1]**2)
    
    for i in range(n_simulations):
        print(f"Simulation {i+1}: Point de départ aléatoire à l'index {random_idx}, prix={random_price:.2f}")
        
        # Créer le simulateur avec ces paramètres
        simulator = GarchSimulator(local_params, random_price, local_volatility)
        prices, vols = simulator.simulate_path(n_steps)
        
        # Stocker les résultats
        simulated_paths.append(prices)
        simulated_vols.append(vols)
        
    # Extraire la trajectoire réelle correspondante
    if random_idx + n_steps < len(data):
        real_trajectory = data['close'].iloc[random_idx:random_idx + n_steps + 1].values
        real_paths.append(real_trajectory)
    else:
        # Si on atteint la fin des données, prendre ce qui est disponible
        real_trajectory = data['close'].iloc[random_idx:].values
        # Compléter avec la dernière valeur pour avoir la même longueur
        padding = np.full(n_steps + 1 - len(real_trajectory), real_trajectory[-1])
        real_trajectory = np.concatenate([real_trajectory, padding])
        real_paths.append(real_trajectory)  

    # 4. Créer un tableau des données simulées
    # Créer un index de dates pour les données simulées
    start_date = datetime.now()
    dates = [start_date + timedelta(minutes=i) for i in range(n_steps + 1)]
    
    # Créer un DataFrame pour les données simulées
    simulated_df = pd.DataFrame(index=dates)
    
    # Ajouter les prix simulés et réels
    for i in range(n_simulations):
        simulated_df[f'price_sim_{i+1}'] = simulated_paths[i]
    simulated_df[f'price_real_{1}'] = real_paths[0]  # Ajouter la trajectoire réelle
    
    # Calculer et ajouter la volatilité réalisée pour chaque simulation
    for i in range(n_simulations):
        prices = simulated_df[f'price_sim_{i+1}'].values
        realized_vol = calculate_realized_volatility(prices)
        simulated_df[f'vol_realized_sim_{i+1}'] = realized_vol
    
    # Calculer et ajouter la volatilité réalisée pour les données réelles
    real_prices = simulated_df[f'price_real_{1}'].values
    real_realized_vol = calculate_realized_volatility(real_prices)
    simulated_df['vol_realized_real'] = real_realized_vol
    
    # 5. Afficher un échantillon du tableau
    print("\nAperçu des données simulées avec volatilité réalisée:")
    print(simulated_df.head(10))
    
    # 6. Visualiser les trajectoires de prix simulées et réelles
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    # Trajectoires de prix
    ax1 = axs[0]
    
    # Tracer les trajectoires simulées
    for i in range(n_simulations):
        ax1.plot(simulated_df.index, simulated_df[f'price_sim_{i+1}'], 
                 label=f'Simulation {i+1}', alpha=0.7)
    
    # Tracer la trajectoire réelle avec des tirets
    ax1.plot(simulated_df.index, simulated_df[f'price_real_{1}'], 
            linestyle='--', label=f'Réel', alpha=0.7, color='black', linewidth=2)
    
    ax1.set_title('Trajectoires de prix simulées vs réelles (GARCH)')
    ax1.set_ylabel('Prix')
    ax1.legend()
    
    # Trajectoires de volatilité GARCH (on garde pour référence)
    ax2 = axs[1]
    for i in range(n_simulations):
        ax2.plot(simulated_df.index, simulated_vols[i], 
                 label=f'Vol GARCH Sim {i+1}', alpha=0.7)
    
    ax2.set_title('Trajectoires de volatilité GARCH (modèle interne)')
    ax2.set_ylabel('Volatilité GARCH')
    ax2.legend()
    
    # Trajectoires de volatilité réalisée
    ax3 = axs[2]
    
    # Tracer les volatilités réalisées simulées
    for i in range(n_simulations):
        ax3.plot(simulated_df.index, simulated_df[f'vol_realized_sim_{i+1}'], 
                 label=f'Vol Réalisée Sim {i+1}', alpha=0.7)
    
    # Tracer la volatilité réalisée réelle
    ax3.plot(simulated_df.index, simulated_df['vol_realized_real'], 
            linestyle='--', label=f'Vol Réalisée Réelle', alpha=0.9, color='black', linewidth=2)
    
    ax3.set_title('Volatilité réalisée (écart-type des rendements sur fenêtre glissante)')
    ax3.set_ylabel('Volatilité Réalisée')
    ax3.legend()
    
    # Ajuster l'espacement entre les sous-graphiques
    fig.subplots_adjust(hspace=0.3)
    
    # 7. Comparer les distributions des rendements
    plt.figure(figsize=(12, 6))
    
    # Histogramme des rendements réels
    historical_returns = np.diff(np.log(data['close'].iloc[-n_steps:].values))
    plt.hist(historical_returns, bins=50, alpha=0.5, color='blue', 
             density=True, label='Rendements historiques')
    
    # Histogrammes des rendements simulés
    simulated_returns = []
    for i in range(n_simulations):
        prices = simulated_df[f'price_sim_{i+1}'].values
        returns = np.diff(np.log(prices))
        simulated_returns.append(returns)
    
    all_sim_returns = np.concatenate(simulated_returns)
    plt.hist(all_sim_returns, bins=50, alpha=0.5, color='red', 
             density=True, label='Rendements simulés')
    
    plt.title('Comparaison des distributions des rendements')
    plt.xlabel('Rendement logarithmique')
    plt.ylabel('Densité')
    plt.legend()
    
    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
    
    return simulated_df

if __name__ == "__main__":
    df_simulated = sample_and_visualize_garch()
    
    # Sauvegarder les données simulées si nécessaire
    # Créer le dossier sample s'il n'existe pas
    os.makedirs("../sample", exist_ok=True)
    df_simulated.to_csv("../sample/garch_simulated_samples.csv")
    print("Données simulées avec volatilité réalisée sauvegardées dans ../sample/garch_simulated_samples.csv")