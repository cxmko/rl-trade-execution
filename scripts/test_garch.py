import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from src.environment.garch_simulator import (
    GarchCalibrator, 
    GarchSimulator, 
    VolumeModelCalibrator,
    calibrate_full_model
)

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.2)

def calculate_realized_volatility(prices, window=20):
    """Calcule la volatilité réalisée"""
    if isinstance(prices, pd.Series):
        prices = prices.values
        
    returns = np.diff(np.log(prices))
    volatility = np.zeros(len(prices))
    
    for i in range(1, len(prices)):
        window_size = min(window, i)
        if window_size > 0:
            window_returns = returns[i-window_size:i]
            volatility[i] = np.std(window_returns) if len(window_returns) > 0 else 0
    
    return volatility

def compute_statistics(simulated_data, real_data):
    """Calcule les statistiques comparatives"""
    mean_real = np.mean(real_data)
    median_real = np.median(real_data)
    std_real = np.std(real_data)
    
    eps = 1e-10
    
    stats_dict = {
        'mean_diff': np.abs(np.mean(simulated_data) - mean_real),
        'mean_rel_error': np.abs(np.mean(simulated_data) - mean_real) / (mean_real + eps) * 100,
        'median_diff': np.abs(np.median(simulated_data) - median_real),
        'median_rel_error': np.abs(np.median(simulated_data) - median_real) / (median_real + eps) * 100,
        'std_diff': np.abs(np.std(simulated_data) - std_real),
        'std_rel_error': np.abs(np.std(simulated_data) - std_real) / (std_real + eps) * 100,
        'min_diff': np.abs(np.min(simulated_data) - np.min(real_data)),
        'max_diff': np.abs(np.max(simulated_data) - np.max(real_data)),
        'skewness_diff': np.abs(pd.Series(simulated_data).skew() - pd.Series(real_data).skew()),
        'kurtosis_diff': np.abs(pd.Series(simulated_data).kurtosis() - pd.Series(real_data).kurtosis()),
    }
    
    for key in stats_dict:
        if 'rel_error' in key:
            stats_dict[key] = min(stats_dict[key], 1000.0)
    
    return stats_dict

def plot_distributions_and_correlations(all_sim_returns, all_real_returns, 
                                      all_sim_volumes, all_real_volumes,
                                      all_sim_vols, all_real_vols):
    """Génère des graphiques détaillés de distribution et corrélation"""
    
    print("\nGénération des graphiques de distribution et corrélation...")
    
    # 1. Distributions (Histograms + KDE)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparaison des Distributions: Réel vs Simulé', fontsize=16, fontweight='bold')
    
    # Returns
    sns.histplot(all_real_returns, stat="density", bins=100, color="black", alpha=0.3, label="Réel", ax=axes[0,0], kde=True)
    sns.histplot(all_sim_returns, stat="density", bins=100, color="blue", alpha=0.3, label="Simulé", ax=axes[0,0], kde=True)
    axes[0,0].set_title("Distribution des Rendements (Log-Returns)")
    axes[0,0].set_xlim([-0.01, 0.01]) # Zoom on center
    axes[0,0].legend()

    # Returns (Log Scale for Tails)
    sns.histplot(all_real_returns, stat="density", bins=100, color="black", alpha=0.3, label="Réel", ax=axes[1,0], kde=False)
    sns.histplot(all_sim_returns, stat="density", bins=100, color="blue", alpha=0.3, label="Simulé", ax=axes[1,0], kde=False)
    axes[1,0].set_yscale('log')
    axes[1,0].set_title("Queues de Distribution (Log Scale)")
    axes[1,0].set_xlim([-0.02, 0.02])
    
    # Volumes
    sns.histplot(all_real_volumes, stat="density", bins=50, color="black", alpha=0.3, label="Réel", ax=axes[0,1], kde=True)
    sns.histplot(all_sim_volumes, stat="density", bins=50, color="orange", alpha=0.3, label="Simulé", ax=axes[0,1], kde=True)
    axes[0,1].set_title("Distribution des Volumes")
    axes[0,1].set_xlim([0, np.percentile(all_real_volumes, 99)]) # Ignore extreme outliers for plot
    axes[0,1].legend()
    
    # Volatility
    sns.histplot(all_real_vols, stat="density", bins=50, color="black", alpha=0.3, label="Réel", ax=axes[0,2], kde=True)
    sns.histplot(all_sim_vols, stat="density", bins=50, color="green", alpha=0.3, label="Simulé", ax=axes[0,2], kde=True)
    axes[0,2].set_title("Distribution de la Volatilité Réalisée")
    axes[0,2].legend()
    
    # QQ Plot (Returns)
    stats.probplot(all_real_returns, dist="norm", plot=axes[1,1])
    axes[1,1].get_lines()[0].set_color('black')
    axes[1,1].get_lines()[0].set_alpha(0.5)
    axes[1,1].get_lines()[0].set_label('Réel')
    
    # Overlay Simulated QQ (Manual approximation for visual comparison)
    # Note: stats.probplot doesn't support easy overlay, so we just plot Real QQ to check normality
    # Ideally we want Sim QQ to match Real QQ.
    axes[1,1].set_title("QQ-Plot (Réel vs Normal)")
    
    # Volume vs Volatility Correlation
    # Sample 5000 points to avoid clutter
    idx = np.random.choice(len(all_sim_volumes), min(5000, len(all_sim_volumes)), replace=False)
    
    axes[1,2].scatter(np.array(all_sim_vols)[idx], np.array(all_sim_volumes)[idx], alpha=0.1, color='blue', label='Simulé')
    axes[1,2].set_title("Corrélation Volume vs Volatilité")
    axes[1,2].set_xlabel("Volatilité")
    axes[1,2].set_ylabel("Volume")
    axes[1,2].set_xscale('log')
    axes[1,2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('../sample/garch_distributions.png', dpi=150)
    print("✓ Distributions sauvegardées: ../sample/garch_distributions.png")
    plt.close()
    
    # 2. Autocorrelation Plot (Volatility Clustering)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate ACF for squared returns (proxy for volatility clustering)
    max_lag = 50
    
    def get_acf(series, lags):
        return [pd.Series(series).autocorr(lag=i) for i in range(lags)]
    
    real_sq_ret = np.array(all_real_returns)**2
    sim_sq_ret = np.array(all_sim_returns)**2
    
    acf_real = get_acf(real_sq_ret, max_lag)
    acf_sim = get_acf(sim_sq_ret, max_lag)
    
    ax.plot(range(max_lag), acf_real, label='Réel (Carré des rendements)', color='black', linewidth=2)
    ax.plot(range(max_lag), acf_sim, label='Simulé (Carré des rendements)', color='blue', linestyle='--', linewidth=2)
    
    ax.set_title("Autocorrélation de la Volatilité (Clustering)")
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Autocorrélation")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../sample/garch_autocorr.png', dpi=150)
    print("✓ Autocorrélation sauvegardée: ../sample/garch_autocorr.png")
    plt.close()


def run_test_mode(data, n_tests=100, n_steps=60):
    """Mode test: effectue plusieurs simulations et compare avec les données réelles"""
    print(f"\n{'='*80}")
    print(f"MODE TEST: {n_tests} simulations de {n_steps} pas")
    print(f"{'='*80}\n")
    
    print("Calibration du modèle GARCH global...")
    garch_params, volume_params = calibrate_full_model(data)
    
    all_price_stats = []
    all_volume_stats = []
    all_volatility_stats = []
    all_returns_stats = []
    
    # Arrays for distribution plotting
    dist_sim_returns = []
    dist_real_returns = []
    dist_sim_volumes = []
    dist_real_volumes = []
    dist_sim_vols = []
    dist_real_vols = []
    
    n_failed = 0
    n_unstable = 0
    
    print(f"\nExécution de {n_tests} simulations...")
    for test_idx in tqdm(range(n_tests)):
        random_idx = np.random.randint(100, len(data) - n_steps - 1)
        
        window_start = max(0, random_idx - 5000)
        window_data = data.iloc[window_start:random_idx]
        
        try:
            # Calibrer localement
            local_calibrator = GarchCalibrator()
            local_params = local_calibrator.fit(window_data['close'])
            
            if local_params['alpha'] + local_params['beta'] >= 0.999:
                n_unstable += 1
                local_params = garch_params
                
            random_price = data['close'].iloc[random_idx]
            local_volatility = np.sqrt(local_calibrator.results.conditional_volatility.iloc[-1]**2)
            local_volatility = min(local_volatility, 0.1)
            
            simulator = GarchSimulator(local_params, random_price, local_volatility, volume_params)
            sim_prices, sim_vols, sim_volumes = simulator.simulate_path(n_steps)
            
            if np.any(np.isnan(sim_prices)) or np.any(np.isinf(sim_prices)):
                n_failed += 1
                continue
            
            if np.max(sim_prices) > random_price * 10 or np.min(sim_prices) < random_price * 0.1:
                n_failed += 1
                continue
            
            real_prices = data['close'].iloc[random_idx:random_idx + n_steps + 1].values
            real_volumes = data['volume'].iloc[random_idx:random_idx + n_steps].values
            
            sim_returns = np.diff(np.log(sim_prices))
            real_returns = np.diff(np.log(real_prices))
            
            if np.any(np.isnan(sim_returns)) or np.any(np.isinf(sim_returns)):
                n_failed += 1
                continue
            
            sim_realized_vol = calculate_realized_volatility(sim_prices)
            real_realized_vol = calculate_realized_volatility(real_prices)
            
            # Collect data for distributions
            dist_sim_returns.extend(sim_returns)
            dist_real_returns.extend(real_returns)
            dist_sim_volumes.extend(sim_volumes)
            dist_real_volumes.extend(real_volumes)
            dist_sim_vols.extend(sim_realized_vol[1:])
            dist_real_vols.extend(real_realized_vol[1:])
            
            price_stats = compute_statistics(sim_prices, real_prices)
            volume_stats = compute_statistics(sim_volumes, real_volumes)
            volatility_stats = compute_statistics(sim_realized_vol[1:], real_realized_vol[1:])
            returns_stats = compute_statistics(sim_returns, real_returns)
            
            if any(np.isnan(list(price_stats.values())) + np.isinf(list(price_stats.values()))):
                n_failed += 1
                continue
            
            all_price_stats.append(price_stats)
            all_volume_stats.append(volume_stats)
            all_volatility_stats.append(volatility_stats)
            all_returns_stats.append(returns_stats)
            
        except Exception as e:
            n_failed += 1
            continue
    
    # Afficher résultats
    print(f"\n{'='*80}")
    print(f"STATISTIQUES DE SIMULATION")
    print(f"{'='*80}")
    print(f"Simulations réussies: {len(all_price_stats)} / {n_tests}")
    print(f"Simulations échouées: {n_failed}")
    print(f"Modèles instables: {n_unstable}")
    
    if len(all_price_stats) == 0:
        print("\n❌ ERREUR: Aucune simulation réussie!")
        return None
    
    print("\n" + "="*80)
    print("RÉSULTATS AGRÉGÉS SUR LES TESTS")
    print("="*80)
    
    def print_aggregated_stats(stats_list, category_name):
        print(f"\n{category_name}:")
        print("-" * 60)
        
        metrics = list(stats_list[0].keys())
        for metric in metrics:
            values = [s[metric] for s in stats_list]
            values_array = np.array(values)
            mean_temp = np.mean(values_array)
            std_temp = np.std(values_array)
            mask = np.abs(values_array - mean_temp) < 3 * std_temp
            values_filtered = values_array[mask]
            
            if len(values_filtered) == 0:
                values_filtered = values_array
            
            mean_val = np.mean(values_filtered)
            std_val = np.std(values_filtered)
            min_val = np.min(values_filtered)
            max_val = np.max(values_filtered)
            
            if 'rel_error' in metric:
                print(f"  {metric:25s}: {mean_val:8.2f}% ± {std_val:6.2f}% (min: {min_val:6.2f}%, max: {max_val:6.2f}%)")
            else:
                print(f"  {metric:25s}: {mean_val:8.4f} ± {std_val:6.4f} (min: {min_val:6.4f}, max: {max_val:6.4f})")
    
    print_aggregated_stats(all_price_stats, "PRIX")
    print_aggregated_stats(all_volume_stats, "VOLUMES")
    print_aggregated_stats(all_volatility_stats, "VOLATILITÉ RÉALISÉE")
    print_aggregated_stats(all_returns_stats, "RENDEMENTS")
    
    create_summary_plot(all_price_stats, all_volume_stats, all_volatility_stats, all_returns_stats)
    
    # Generate Distribution Plots
    plot_distributions_and_correlations(
        dist_sim_returns, dist_real_returns,
        dist_sim_volumes, dist_real_volumes,
        dist_sim_vols, dist_real_vols
    )
    
    return {
        'price_stats': all_price_stats,
        'volume_stats': all_volume_stats,
        'volatility_stats': all_volatility_stats,
        'returns_stats': all_returns_stats
    }

def create_summary_plot(price_stats, volume_stats, volatility_stats, returns_stats):
    """Crée un graphique récapitulatif"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse comparative: Simulé vs Réel', fontsize=16, fontweight='bold')
    
    def plot_metrics(ax, stats_list, title, color):
        metrics = ['mean_rel_error', 'median_rel_error', 'std_rel_error']
        labels = ['Erreur Moyenne\n(%)', 'Erreur Médiane\n(%)', 'Erreur Écart-type\n(%)']
        
        values = []
        errors = []
        for metric in metrics:
            metric_values = [s[metric] for s in stats_list]
            values.append(np.mean(metric_values))
            errors.append(np.std(metric_values))
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7, color=color, edgecolor='black')
        
        ax.set_ylabel('Erreur relative (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err,
                   f'{val:.1f}%\n±{err:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plot_metrics(axes[0, 0], price_stats, 'PRIX', 'steelblue')
    plot_metrics(axes[0, 1], volume_stats, 'VOLUMES', 'coral')
    plot_metrics(axes[1, 0], volatility_stats, 'VOLATILITÉ RÉALISÉE', 'mediumseagreen')
    plot_metrics(axes[1, 1], returns_stats, 'RENDEMENTS', 'mediumpurple')
    
    plt.tight_layout()
    plt.savefig('../sample/garch_test_summary.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: ../sample/garch_test_summary.png")
    plt.close()

def sample_and_visualize_garch():
    """Mode visualisation"""
    print("Chargement des données...")
    data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    print("Calibration des modèles...")
    garch_params, volume_params = calibrate_full_model(data)
    
    n_simulations = 5
    n_steps = 1440
    
    random_idx = np.random.randint(100, len(data) - n_steps - 1)
    random_price = data['close'].iloc[random_idx]
    
    window_start = max(0, random_idx - 5000)
    window_data = data.iloc[window_start:random_idx]
    
    local_calibrator = GarchCalibrator()
    local_params = local_calibrator.fit(window_data['close'])
    local_volatility = np.sqrt(local_calibrator.results.conditional_volatility.iloc[-1]**2)
    
    simulated_paths = []
    simulated_vols = []
    simulated_volumes = []
    
    for i in range(n_simulations):
        print(f"Simulation {i+1}/{n_simulations}")
        simulator = GarchSimulator(local_params, random_price, local_volatility, volume_params)
        prices, vols, volumes = simulator.simulate_path(n_steps)
        
        simulated_paths.append(prices)
        simulated_vols.append(vols)
        simulated_volumes.append(volumes)
    
    # Données réelles
    real_trajectory = data['close'].iloc[random_idx:random_idx + n_steps + 1].values
    real_volume_trajectory = data['volume'].iloc[random_idx:random_idx + n_steps].values
    
    # Calculer la volatilité réalisée sur la trajectoire réelle
    real_volatility = calculate_realized_volatility(real_trajectory)
    
    # Graphiques
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    for i in range(n_simulations):
        axs[0].plot(simulated_paths[i], label=f'Simulation {i+1}', alpha=0.7)
    axs[0].plot(real_trajectory, '--', label='Réel', color='black', linewidth=2)
    axs[0].set_title('Trajectoires de prix (GARCH)')
    axs[0].set_ylabel('Prix')
    axs[0].legend()
    
    for i in range(n_simulations):
        axs[1].plot(simulated_vols[i], label=f'Vol Sim {i+1}', alpha=0.7)
    
    # Ajouter la courbe de volatilité réelle
    axs[1].plot(real_volatility, '--', label='Réel (Réalisée)', color='black', linewidth=2)
    
    axs[1].set_title('Volatilité GARCH vs Réalisée')
    axs[1].set_ylabel('Volatilité')
    axs[1].legend()
    
    for i in range(n_simulations):
        axs[2].plot(simulated_volumes[i], label=f'Vol Sim {i+1}', alpha=0.7)
    axs[2].plot(real_volume_trajectory, '--', label='Réel', color='black', linewidth=2)
    axs[2].set_title('Volumes (Modèle bimodal)')
    axs[2].set_ylabel('Volume')
    axs[2].set_xlabel('Pas de temps')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('../sample/garch_simulation.png', dpi=150)
    print("\n✓ Visualisation sauvegardée: ../sample/garch_simulation.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test du simulateur GARCH')
    parser.add_argument('--mode', type=str, default='visualize', 
                       choices=['visualize', 'test'],
                       help='Mode: visualize ou test')
    parser.add_argument('--n_tests', type=int, default=10000,
                       help='Nombre de tests')
    parser.add_argument('--n_steps', type=int, default=240,
                       help='Nombre de pas')
    
    args = parser.parse_args()
    
    os.makedirs("../sample", exist_ok=True)
    
    if args.mode == 'visualize':
        sample_and_visualize_garch()
    elif args.mode == 'test':
        data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        results = run_test_mode(data, n_tests=args.n_tests, n_steps=args.n_steps)