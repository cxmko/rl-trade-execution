import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sys.path.append(os.path.abspath(".."))

def analyze_volume_distribution(data):
    """
    Analyse complète et détaillée de la distribution des volumes
    """
    print("="*80)
    print("ANALYSE APPROFONDIE DE LA DISTRIBUTION DES VOLUMES")
    print("="*80)
    
    volumes = data['volume'].values
    
    # ========== STATISTIQUES DE BASE ==========
    print("\n" + "="*80)
    print("1. STATISTIQUES DESCRIPTIVES DE BASE")
    print("="*80)
    
    print(f"\nNombre d'observations: {len(volumes):,}")
    print(f"Moyenne: {np.mean(volumes):.2f}")
    print(f"Médiane: {np.median(volumes):.2f}")
    print(f"Écart-type: {np.std(volumes):.2f}")
    print(f"Variance: {np.var(volumes):.2f}")
    print(f"Min: {np.min(volumes):.2f}")
    print(f"Max: {np.max(volumes):.2f}")
    print(f"Étendue (Range): {np.max(volumes) - np.min(volumes):.2f}")
    print(f"Coefficient de variation: {np.std(volumes) / np.mean(volumes):.2f}")
    
    # Moments d'ordre supérieur
    print(f"\nSkewness (asymétrie): {stats.skew(volumes):.4f}")
    print(f"Kurtosis (aplatissement): {stats.kurtosis(volumes):.4f}")
    
    # Test de normalité
    _, p_value = stats.normaltest(volumes)
    print(f"\nTest de normalité (D'Agostino-Pearson):")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Normal? {'Non' if p_value < 0.05 else 'Oui'} (seuil 5%)")
    
    # ========== QUANTILES DÉTAILLÉS ==========
    print("\n" + "="*80)
    print("2. ANALYSE PAR QUANTILES")
    print("="*80)
    
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    print("\nQuantiles:")
    for p in percentiles:
        val = np.percentile(volumes, p)
        count = np.sum(volumes <= val)
        pct = count / len(volumes) * 100
        print(f"  P{p:5.1f}: {val:12.2f}  ({count:7,} obs = {pct:5.2f}%)")
    
    # Écarts inter-quartiles
    q25, q50, q75 = np.percentile(volumes, [25, 50, 75])
    iqr = q75 - q25
    print(f"\nÉcart inter-quartile (IQR): {iqr:.2f}")
    print(f"Ratio Q75/Q25: {q75/q25:.2f}")
    print(f"Ratio Q50/Q25: {q50/q25:.2f}")
    
    # ========== ANALYSE DES VALEURS EXTRÊMES ==========
    print("\n" + "="*80)
    print("3. ANALYSE DES VALEURS EXTRÊMES")
    print("="*80)
    
    # Outliers selon la règle IQR
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers_low = volumes < lower_bound
    outliers_high = volumes > upper_bound
    
    print(f"\nRègle IQR (1.5 * IQR):")
    print(f"  Borne inférieure: {lower_bound:.2f}")
    print(f"  Borne supérieure: {upper_bound:.2f}")
    print(f"  Outliers bas: {np.sum(outliers_low):,} ({np.sum(outliers_low)/len(volumes)*100:.2f}%)")
    print(f"  Outliers hauts: {np.sum(outliers_high):,} ({np.sum(outliers_high)/len(volumes)*100:.2f}%)")
    
    # Top volumes
    print(f"\n10 plus gros volumes:")
    top_volumes = sorted(volumes, reverse=True)[:10]
    for i, v in enumerate(top_volumes, 1):
        print(f"  {i:2d}. {v:12.2f}  (ratio au médian: {v/q50:.1f}x)")
    
    # Distribution des zéros et valeurs très faibles
    zeros = np.sum(volumes == 0)
    near_zeros = np.sum((volumes > 0) & (volumes < 1))
    very_low = np.sum((volumes >= 1) & (volumes < 10))
    
    print(f"\nValeurs très faibles:")
    print(f"  Exactement 0: {zeros:,} ({zeros/len(volumes)*100:.2f}%)")
    print(f"  Entre 0 et 1: {near_zeros:,} ({near_zeros/len(volumes)*100:.2f}%)")
    print(f"  Entre 1 et 10: {very_low:,} ({very_low/len(volumes)*100:.2f}%)")
    
    # ========== ANALYSE PAR FENÊTRES D'1 HEURE ==========
    print("\n" + "="*80)
    print("4. ANALYSE PAR FENÊTRES D'1 HEURE (60 minutes)")
    print("="*80)
    
    # Créer des fenêtres de 60 minutes
    n_windows = len(volumes) // 60
    hourly_stats = []
    
    for i in range(n_windows):
        window = volumes[i*60:(i+1)*60]
        hourly_stats.append({
            'mean': np.mean(window),
            'median': np.median(window),
            'std': np.std(window),
            'min': np.min(window),
            'max': np.max(window),
            'sum': np.sum(window),
            'range': np.max(window) - np.min(window)
        })
    
    hourly_df = pd.DataFrame(hourly_stats)
    
    print(f"\nNombre de fenêtres d'1 heure analysées: {n_windows:,}")
    print("\nStatistiques MOYENNES par fenêtre d'1 heure:")
    for col in hourly_df.columns:
        print(f"  {col:10s}: {hourly_df[col].mean():10.2f} ± {hourly_df[col].std():8.2f}")
    
    print("\nStatistiques des fenêtres d'1 heure (quartiles):")
    print(hourly_df.describe().T[['min', '25%', '50%', '75%', 'max']])
    
    # Variabilité entre fenêtres
    print(f"\nVariabilité inter-fenêtres:")
    print(f"  CV de la moyenne horaire: {hourly_df['mean'].std() / hourly_df['mean'].mean():.2f}")
    print(f"  CV de l'écart-type horaire: {hourly_df['std'].std() / hourly_df['std'].mean():.2f}")
    
    # ========== ANALYSE DE STATIONNARITÉ ==========
    print("\n" + "="*80)
    print("5. ANALYSE DE STATIONNARITÉ")
    print("="*80)
    
    # Découper en 10 périodes
    n_periods = 10
    period_size = len(volumes) // n_periods
    period_stats = []
    
    for i in range(n_periods):
        start = i * period_size
        end = min((i+1) * period_size, len(volumes))
        period = volumes[start:end]
        period_stats.append({
            'period': i+1,
            'mean': np.mean(period),
            'std': np.std(period),
            'median': np.median(period),
            'max': np.max(period)
        })
    
    period_df = pd.DataFrame(period_stats)
    
    print("\nStatistiques par période (10 périodes égales):")
    print(period_df.to_string(index=False))
    
    print(f"\nStabilité temporelle:")
    print(f"  Écart-type de la moyenne: {period_df['mean'].std():.2f}")
    print(f"  Ratio max/min des moyennes: {period_df['mean'].max() / period_df['mean'].min():.2f}")
    print(f"  Écart-type de l'écart-type: {period_df['std'].std():.2f}")
    
    # ========== ANALYSE DE LA FORME DE DISTRIBUTION ==========
    print("\n" + "="*80)
    print("6. TESTS D'AJUSTEMENT DE DISTRIBUTIONS")
    print("="*80)
    
    # Tester plusieurs distributions
    distributions = {
        'normal': stats.norm,
        'lognormal': stats.lognorm,
        'exponential': stats.expon,
        'gamma': stats.gamma,
        'weibull': stats.weibull_min,
    }
    
    print("\nTests de Kolmogorov-Smirnov (ajustement de distribution):")
    best_dist = None
    best_p = 0
    
    for name, dist in distributions.items():
        try:
            # Ajuster la distribution
            if name == 'lognormal':
                # Log-normale nécessite log(volumes > 0)
                vol_positive = volumes[volumes > 0]
                params = dist.fit(vol_positive)
                ks_stat, p_value = stats.kstest(vol_positive, lambda x: dist.cdf(x, *params))
            else:
                params = dist.fit(volumes)
                ks_stat, p_value = stats.kstest(volumes, lambda x: dist.cdf(x, *params))
            
            print(f"  {name:15s}: KS={ks_stat:.4f}, p-value={p_value:.6f}")
            
            if p_value > best_p:
                best_p = p_value
                best_dist = name
        except:
            print(f"  {name:15s}: Échec de l'ajustement")
    
    print(f"\nMeilleure distribution: {best_dist} (p-value={best_p:.6f})")
    
    # ========== ANALYSE LOG-TRANSFORMATION ==========
    print("\n" + "="*80)
    print("7. ANALYSE EN LOG (pour modélisation log-log)")
    print("="*80)
    
    vol_positive = volumes[volumes > 0]
    log_volumes = np.log(vol_positive)
    
    print(f"\nVolumes positifs: {len(vol_positive):,} ({len(vol_positive)/len(volumes)*100:.2f}%)")
    print(f"\nLog(Volume) - Statistiques:")
    print(f"  Moyenne: {np.mean(log_volumes):.4f}")
    print(f"  Médiane: {np.median(log_volumes):.4f}")
    print(f"  Écart-type: {np.std(log_volumes):.4f}")
    print(f"  Min: {np.min(log_volumes):.4f}")
    print(f"  Max: {np.max(log_volumes):.4f}")
    print(f"  Skewness: {stats.skew(log_volumes):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(log_volumes):.4f}")
    
    # Test de normalité sur log
    _, p_value_log = stats.normaltest(log_volumes)
    print(f"\nTest de normalité sur log(Volume):")
    print(f"  p-value: {p_value_log:.6f}")
    print(f"  Normal? {'Oui' if p_value_log > 0.05 else 'Non'} (seuil 5%)")
    
    # ========== ANALYSE DES CLUSTERS ==========
    print("\n" + "="*80)
    print("8. ANALYSE DES REGIMES / CLUSTERS")
    print("="*80)
    
    # K-means simple pour identifier des régimes naturels
    from sklearn.cluster import KMeans
    
    # Essayer 2, 3, 4 clusters
    for n_clusters in [2, 3, 4]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(volumes.reshape(-1, 1))
        
        print(f"\n{n_clusters} clusters (K-means):")
        for i in range(n_clusters):
            cluster_vols = volumes[labels == i]
            print(f"  Cluster {i+1}: n={len(cluster_vols):7,} ({len(cluster_vols)/len(volumes)*100:5.2f}%)")
            print(f"    Moyenne: {np.mean(cluster_vols):10.2f}")
            print(f"    Médiane: {np.median(cluster_vols):10.2f}")
            print(f"    Écart-type: {np.std(cluster_vols):10.2f}")
            print(f"    Min: {np.min(cluster_vols):10.2f}, Max: {np.max(cluster_vols):10.2f}")
    
    # ========== GRAPHIQUES ==========
    print("\n" + "="*80)
    print("9. GÉNÉRATION DES GRAPHIQUES")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Histogramme général
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(volumes, bins=100, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Volume')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des volumes (échelle linéaire)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogramme log
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(log_volumes, bins=100, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('log(Volume)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution de log(Volume)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.boxplot(volumes, vert=True)
    ax3.set_ylabel('Volume')
    ax3.set_title('Box plot des volumes')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. QQ plot (normalité)
    ax4 = fig.add_subplot(gs[1, 0])
    stats.probplot(volumes, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normal) - Volume')
    ax4.grid(True, alpha=0.3)
    
    # 5. QQ plot log (normalité)
    ax5 = fig.add_subplot(gs[1, 1])
    stats.probplot(log_volumes, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normal) - log(Volume)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Évolution temporelle
    ax6 = fig.add_subplot(gs[1, 2])
    sample_size = min(10000, len(volumes))
    ax6.plot(volumes[:sample_size], alpha=0.5, linewidth=0.5)
    ax6.set_xlabel('Temps (minutes)')
    ax6.set_ylabel('Volume')
    ax6.set_title(f'Série temporelle (premiers {sample_size:,} points)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Distribution par fenêtre d'1h (box plots)
    ax7 = fig.add_subplot(gs[2, 0])
    sample_windows = min(50, len(hourly_df))
    hourly_sample = hourly_df.iloc[:sample_windows]
    ax7.boxplot([hourly_sample['mean'], hourly_sample['median'], 
                 hourly_sample['std'], hourly_sample['max']],
                labels=['Moyenne', 'Médiane', 'Écart-type', 'Max'])
    ax7.set_ylabel('Valeur')
    ax7.set_title('Distribution des stats horaires')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Évolution des stats par période
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(period_df['period'], period_df['mean'], 'o-', label='Moyenne')
    ax8.plot(period_df['period'], period_df['median'], 's-', label='Médiane')
    ax8.fill_between(period_df['period'], 
                     period_df['mean'] - period_df['std'],
                     period_df['mean'] + period_df['std'],
                     alpha=0.3, label='±1 std')
    ax8.set_xlabel('Période')
    ax8.set_ylabel('Volume')
    ax8.set_title('Évolution temporelle des statistiques')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Histogramme log avec zoom sur queues
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(volumes, bins=100, edgecolor='black', alpha=0.7)
    ax9.set_yscale('log')
    ax9.set_xlabel('Volume')
    ax9.set_ylabel('Fréquence (échelle log)')
    ax9.set_title('Distribution (échelle log-y)')
    ax9.grid(True, alpha=0.3)
    
    # 10. ECDF (fonction de répartition empirique)
    ax10 = fig.add_subplot(gs[3, 0])
    sorted_vols = np.sort(volumes)
    ecdf = np.arange(1, len(sorted_vols)+1) / len(sorted_vols)
    ax10.plot(sorted_vols, ecdf, linewidth=1)
    ax10.set_xlabel('Volume')
    ax10.set_ylabel('Probabilité cumulative')
    ax10.set_title('ECDF (Fonction de répartition empirique)')
    ax10.grid(True, alpha=0.3)
    
    # 11. Log-Log ECDF (pour voir queues)
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(sorted_vols, 1 - ecdf, linewidth=1)
    ax11.set_xscale('log')
    ax11.set_yscale('log')
    ax11.set_xlabel('Volume (log)')
    ax11.set_ylabel('P(X > x) (log)')
    ax11.set_title('Tail distribution (log-log)')
    ax11.grid(True, alpha=0.3)
    
    # 12. Autocorrélation
    ax12 = fig.add_subplot(gs[3, 2])
    from pandas.plotting import autocorrelation_plot
    # Prendre un échantillon pour ne pas surcharger
    vol_sample = volumes[::10]  # 1 point sur 10
    autocorrelation_plot(pd.Series(vol_sample), ax=ax12)
    ax12.set_title('Autocorrélation des volumes')
    ax12.set_xlim([0, 1000])
    
    plt.suptitle('ANALYSE COMPLÈTE DE LA DISTRIBUTION DES VOLUMES', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('../sample/volume_complete_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graphiques sauvegardés: ../sample/volume_complete_analysis.png")
    
    # ========== RECOMMANDATIONS POUR MODÉLISATION ==========
    print("\n" + "="*80)
    print("10. RECOMMANDATIONS POUR LA MODÉLISATION")
    print("="*80)
    
    # Analyser la proportion de valeurs dans différents régimes
    p10 = np.percentile(volumes, 10)
    p50 = np.percentile(volumes, 50)
    p90 = np.percentile(volumes, 90)
    p99 = np.percentile(volumes, 99)
    
    very_low = np.sum(volumes <= p10)
    low = np.sum((volumes > p10) & (volumes <= p50))
    medium = np.sum((volumes > p50) & (volumes <= p90))
    high = np.sum((volumes > p90) & (volumes <= p99))
    extreme = np.sum(volumes > p99)
    
    print("\nRépartition suggérée par quantiles:")
    print(f"  Très faible (0-P10):   {very_low:7,} obs ({very_low/len(volumes)*100:5.2f}%)  [0 - {p10:.2f}]")
    print(f"  Faible (P10-P50):      {low:7,} obs ({low/len(volumes)*100:5.2f}%)  [{p10:.2f} - {p50:.2f}]")
    print(f"  Moyen (P50-P90):       {medium:7,} obs ({medium/len(volumes)*100:5.2f}%)  [{p50:.2f} - {p90:.2f}]")
    print(f"  Élevé (P90-P99):       {high:7,} obs ({high/len(volumes)*100:5.2f}%)  [{p90:.2f} - {p99:.2f}]")
    print(f"  Extrême (>P99):        {extreme:7,} obs ({extreme/len(volumes)*100:5.2f}%)  [> {p99:.2f}]")
    
    print("\nCaractéristiques clés pour le modèle:")
    print(f"  - Distribution très asymétrique (skewness={stats.skew(volumes):.2f})")
    print(f"  - Queues lourdes (kurtosis={stats.kurtosis(volumes):.2f})")
    print(f"  - Log-transformation {'améliore' if p_value_log > p_value else 'ne change pas'} la normalité")
    print(f"  - Variabilité {'forte' if hourly_df['std'].std() / hourly_df['std'].mean() > 0.5 else 'modérée'} entre fenêtres horaires")
    print(f"  - Ratio volume max/médian: {np.max(volumes)/np.median(volumes):.1f}x")
    
    return {
        'volumes': volumes,
        'hourly_stats': hourly_df,
        'period_stats': period_df,
        'log_volumes': log_volumes,
        'best_distribution': best_dist
    }


if __name__ == "__main__":
    os.makedirs("../sample", exist_ok=True)
    
    print("Chargement des données...")
    data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    print(f"Données chargées: {len(data):,} observations")
    print(f"Période: {data.index[0]} à {data.index[-1]}")
    
    results = analyze_volume_distribution(data)
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)
    print("\nFichiers générés:")
    print("  - ../sample/volume_complete_analysis.png (12 graphiques)")
    print("\nUtilisez ces résultats pour concevoir un modèle volume adapté !")