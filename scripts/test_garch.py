import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    stats = {
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
    
    for key in stats:
        if 'rel_error' in key:
            stats[key] = min(stats[key], 1000.0)
    
    return stats


def run_test_mode(data, n_tests=100, n_steps=60):
    """Mode test: effectue plusieurs simulations et compare avec les données réelles"""
    print(f"\n{'='*80}")
    print(f"MODE TEST: {n_tests} simulations de {n_steps} pas")
    print(f"{'='*80}\n")
    
    # Calibrer les modèles GARCH et volume
    print("Calibration du modèle GARCH global...")
    garch_params, volume_params = calibrate_full_model(data)
    
    all_price_stats = []
    all_volume_stats = []
    all_volatility_stats = []
    all_returns_stats = []
    
    all_sim_volumes = []
    all_real_volumes = []
    
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
            
            # Créer le simulateur avec modèle volume
            simulator = GarchSimulator(local_params, random_price, local_volatility, volume_params)
            sim_prices, sim_vols, sim_volumes = simulator.simulate_path(n_steps)
            
            if np.any(np.isnan(sim_prices)) or np.any(np.isinf(sim_prices)):
                n_failed += 1
                continue
            
            if np.max(sim_prices) > random_price * 10 or np.min(sim_prices) < random_price * 0.1:
                n_failed += 1
                continue
            
            # Données réelles
            real_prices = data['close'].iloc[random_idx:random_idx + n_steps + 1].values
            real_volumes = data['volume'].iloc[random_idx:random_idx + n_steps].values
            
            all_sim_volumes.extend(sim_volumes)
            all_real_volumes.extend(real_volumes)
            
            # Calculs
            sim_returns = np.diff(np.log(sim_prices))
            real_returns = np.diff(np.log(real_prices))
            
            if np.any(np.isnan(sim_returns)) or np.any(np.isinf(sim_returns)):
                n_failed += 1
                continue
            
            sim_realized_vol = calculate_realized_volatility(sim_prices)
            real_realized_vol = calculate_realized_volatility(real_prices)
            
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
    
    # Statistiques volumes
    print("\n" + "="*80)
    print("STATISTIQUES DÉTAILLÉES DES VOLUMES")
    print("="*80)
    
    all_sim_volumes = np.array(all_sim_volumes)
    all_real_volumes = np.array(all_real_volumes)
    
    print("\nVolumes réels:")
    print(f"  Moyenne: {np.mean(all_real_volumes):.2f}")
    print(f"  Médiane: {np.median(all_real_volumes):.2f}")
    print(f"  Écart-type: {np.std(all_real_volumes):.2f}")
    print(f"  Min: {np.min(all_real_volumes):.2f}")
    print(f"  Max: {np.max(all_real_volumes):.2f}")
    
    print("\nVolumes simulés:")
    print(f"  Moyenne: {np.mean(all_sim_volumes):.2f}")
    print(f"  Médiane: {np.median(all_sim_volumes):.2f}")
    print(f"  Écart-type: {np.std(all_sim_volumes):.2f}")
    print(f"  Min: {np.min(all_sim_volumes):.2f}")
    print(f"  Max: {np.max(all_sim_volumes):.2f}")
    
    print("\nDifférence relative:")
    print(f"  Moyenne: {(np.mean(all_sim_volumes) - np.mean(all_real_volumes)) / np.mean(all_real_volumes) * 100:.2f}%")
    print(f"  Écart-type: {(np.std(all_sim_volumes) - np.std(all_real_volumes)) / np.std(all_real_volumes) * 100:.2f}%")
    
    create_summary_plot(all_price_stats, all_volume_stats, all_volatility_stats, all_returns_stats)
    
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
    axs[1].set_title('Volatilité GARCH')
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
    parser.add_argument('--n_tests', type=int, default=100,
                       help='Nombre de tests')
    parser.add_argument('--n_steps', type=int, default=60,
                       help='Nombre de pas')
    
    args = parser.parse_args()
    
    os.makedirs("../sample", exist_ok=True)
    
    if args.mode == 'visualize':
        sample_and_visualize_garch()
    elif args.mode == 'test':
        data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        results = run_test_mode(data, n_tests=args.n_tests, n_steps=args.n_steps)