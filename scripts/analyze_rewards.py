import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='The optimizer returned code 4')

sys.path.append(os.path.abspath(".."))
from src.environment.execution_env import OptimalExecutionEnv
from src.models.ppo_agent import PPOAgent


class RewardAnalyzer:
    """Analyseur détaillé des récompenses pendant l'entraînement"""
    
    def __init__(self):
        self.episode_rewards = []           # Récompense totale par épisode
        self.step_rewards = []              # Récompenses individuelles (tous épisodes)
        self.episode_step_rewards = []      # Liste des récompenses par épisode
        self.episode_revenues = []          # Revenu total par épisode
        self.episode_lengths = []           # Nombre de pas effectifs par épisode
        self.episode_inventories = []       # Inventaire final par épisode
        self.episode_impacts = []           # Impact moyen par épisode
        
        # Statistiques par pas de temps
        self.timestep_rewards = {i: [] for i in range(60)}
        
    def record_episode(self, rewards: list, revenue: float, length: int, 
                      inventory: float, avg_impact: float):
        """Enregistrer les données d'un épisode"""
        self.episode_rewards.append(sum(rewards))
        self.episode_step_rewards.append(rewards)
        self.step_rewards.extend(rewards)
        self.episode_revenues.append(revenue)
        self.episode_lengths.append(length)
        self.episode_inventories.append(inventory)
        self.episode_impacts.append(avg_impact)
        
        # Enregistrer récompenses par timestep
        for t, r in enumerate(rewards):
            if t < 60:
                self.timestep_rewards[t].append(r)
    
    def get_statistics(self):
        """Calculer les statistiques globales"""
        return {
            'reward_mean': np.mean(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'reward_min': np.min(self.episode_rewards),
            'reward_max': np.max(self.episode_rewards),
            'reward_median': np.median(self.episode_rewards),
            'reward_q25': np.percentile(self.episode_rewards, 25),
            'reward_q75': np.percentile(self.episode_rewards, 75),
            'step_reward_mean': np.mean(self.step_rewards),
            'step_reward_std': np.std(self.step_rewards),
            'step_reward_min': np.min(self.step_rewards),
            'step_reward_max': np.max(self.step_rewards),
            'revenue_mean': np.mean(self.episode_revenues),
            'revenue_std': np.std(self.episode_revenues),
            'length_mean': np.mean(self.episode_lengths),
            'inventory_mean': np.mean(self.episode_inventories),
            'impact_mean': np.mean(self.episode_impacts)
        }
    
    def plot_detailed_analysis(self, save_path='../sample/reward_analysis.png'):
        """Créer une visualisation complète des récompenses"""
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        stats = self.get_statistics()
        n_episodes = len(self.episode_rewards)
        
        # ═══════════════════════════════════════════════════════════════
        # 1. ÉVOLUTION DE LA RÉCOMPENSE TOTALE PAR ÉPISODE
        # ═══════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, :])
        episodes = np.arange(1, n_episodes + 1)
        
        # Courbe brute
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='steelblue', 
                label='Récompense brute', linewidth=1)
        
        # Moyenne mobile (fenêtre 10)
        if n_episodes >= 10:
            window = min(10, n_episodes // 5)
            ma = pd.Series(self.episode_rewards).rolling(window=window, min_periods=1).mean()
            ax1.plot(episodes, ma, color='darkblue', linewidth=2.5, 
                    label=f'Moyenne mobile ({window} ép.)')
        
        # Ligne médiane
        ax1.axhline(y=stats['reward_median'], color='red', linestyle='--', 
                   alpha=0.7, linewidth=1.5, label=f"Médiane: {stats['reward_median']:.2e}")
        
        ax1.set_xlabel('Épisode', fontsize=12)
        ax1.set_ylabel('Récompense Totale', fontsize=12)
        ax1.set_title(f'Évolution de la Récompense Totale par Épisode ({n_episodes} épisodes)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # ═══════════════════════════════════════════════════════════════
        # 2. DISTRIBUTION DES RÉCOMPENSES TOTALES
        # ═══════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.episode_rewards, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=stats['reward_mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Moyenne: {stats['reward_mean']:.2e}")
        ax2.axvline(x=stats['reward_median'], color='orange', linestyle='--', 
                   linewidth=2, label=f"Médiane: {stats['reward_median']:.2e}")
        ax2.set_xlabel('Récompense Totale', fontsize=11)
        ax2.set_ylabel('Fréquence', fontsize=11)
        ax2.set_title('Distribution des Récompenses Totales', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # ═══════════════════════════════════════════════════════════════
        # 3. BOXPLOT DES RÉCOMPENSES
        # ═══════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[1, 1])
        bp = ax3.boxplot([self.episode_rewards], vert=True, patch_artist=True,
                         labels=['Épisodes'])
        bp['boxes'][0].set_facecolor('lightblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        ax3.set_ylabel('Récompense Totale', fontsize=11)
        ax3.set_title('Boxplot des Récompenses', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Ajouter annotations
        text_stats = f"Min: {stats['reward_min']:.2e}\n"
        text_stats += f"Q1: {stats['reward_q25']:.2e}\n"
        text_stats += f"Med: {stats['reward_median']:.2e}\n"
        text_stats += f"Q3: {stats['reward_q75']:.2e}\n"
        text_stats += f"Max: {stats['reward_max']:.2e}"
        ax3.text(1.15, 0.5, text_stats, transform=ax3.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ═══════════════════════════════════════════════════════════════
        # 4. DISTRIBUTION DES RÉCOMPENSES PAR PAS
        # ═══════════════════════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(self.step_rewards, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax4.axvline(x=stats['step_reward_mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Moyenne: {stats['step_reward_mean']:.2e}")
        ax4.set_xlabel('Récompense par Pas', fontsize=11)
        ax4.set_ylabel('Fréquence', fontsize=11)
        ax4.set_title('Distribution des Récompenses par Pas', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # ═══════════════════════════════════════════════════════════════
        # 5. RÉCOMPENSE MOYENNE PAR TIMESTEP
        # ═══════════════════════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[2, :])
        
        # Calculer moyenne et écart-type par timestep
        timesteps = []
        means = []
        stds = []
        for t in range(60):
            if len(self.timestep_rewards[t]) > 0:
                timesteps.append(t)
                means.append(np.mean(self.timestep_rewards[t]))
                stds.append(np.std(self.timestep_rewards[t]))
        
        timesteps = np.array(timesteps)
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot avec intervalle de confiance
        ax5.plot(timesteps, means, color='darkgreen', linewidth=2.5, label='Moyenne')
        ax5.fill_between(timesteps, means - stds, means + stds, 
                         alpha=0.3, color='green', label='±1 std')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax5.set_xlabel('Pas de Temps', fontsize=12)
        ax5.set_ylabel('Récompense Moyenne', fontsize=12)
        ax5.set_title('Récompense Moyenne par Pas de Temps (avec écart-type)', 
                     fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, 59])
        ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # ═══════════════════════════════════════════════════════════════
        # 6. CORRÉLATION RÉCOMPENSE vs REVENU
        # ═══════════════════════════════════════════════════════════════
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.scatter(self.episode_rewards, self.episode_revenues, 
                   alpha=0.5, s=20, color='steelblue')
        
        # Ligne de régression
        z = np.polyfit(self.episode_rewards, self.episode_revenues, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(self.episode_rewards), max(self.episode_rewards), 100)
        ax6.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Régression: y={z[0]:.2e}x+{z[1]:.2e}')
        
        # Corrélation
        corr = np.corrcoef(self.episode_rewards, self.episode_revenues)[0, 1]
        ax6.text(0.05, 0.95, f'Corrélation: {corr:.3f}', transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax6.set_xlabel('Récompense Totale', fontsize=11)
        ax6.set_ylabel('Revenu (USDT)', fontsize=11)
        ax6.set_title('Corrélation Récompense vs Revenu', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax6.ticklabel_format(style='plain', axis='y')
        
        # ═══════════════════════════════════════════════════════════════
        # 7. CORRÉLATION RÉCOMPENSE vs NOMBRE DE PAS
        # ═══════════════════════════════════════════════════════════════
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.scatter(self.episode_rewards, self.episode_lengths, 
                   alpha=0.5, s=20, color='coral')
        
        corr = np.corrcoef(self.episode_rewards, self.episode_lengths)[0, 1]
        ax7.text(0.05, 0.95, f'Corrélation: {corr:.3f}', transform=ax7.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax7.set_xlabel('Récompense Totale', fontsize=11)
        ax7.set_ylabel('Nombre de Pas Effectifs', fontsize=11)
        ax7.set_title('Corrélation Récompense vs Pas', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # ═══════════════════════════════════════════════════════════════
        # 8. STATISTIQUES TEXTUELLES
        # ═══════════════════════════════════════════════════════════════
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        stats_text = f"""
╔═══════════════════════════════════════╗
║     STATISTIQUES DES RÉCOMPENSES      ║
╚═══════════════════════════════════════╝

📊 RÉCOMPENSES TOTALES (par épisode)
  Moyenne:        {stats['reward_mean']:>15.4e}
  Écart-type:     {stats['reward_std']:>15.4e}
  Médiane:        {stats['reward_median']:>15.4e}
  Min:            {stats['reward_min']:>15.4e}
  Max:            {stats['reward_max']:>15.4e}
  Q1:             {stats['reward_q25']:>15.4e}
  Q3:             {stats['reward_q75']:>15.4e}

📈 RÉCOMPENSES PAR PAS
  Moyenne:        {stats['step_reward_mean']:>15.4e}
  Écart-type:     {stats['step_reward_std']:>15.4e}
  Min:            {stats['step_reward_min']:>15.4e}
  Max:            {stats['step_reward_max']:>15.4e}

💰 MÉTRIQUES LIÉES
  Revenu moyen:   {stats['revenue_mean']:>15,.2f} USDT
  Pas moyen:      {stats['length_mean']:>15.1f}
  Inv. final:     {stats['inventory_mean']:>15.2f} BTC
  Impact moyen:   {stats['impact_mean']:>15.2f} bps

───────────────────────────────────────────
Total épisodes: {n_episodes}
Total pas:      {len(self.step_rewards)}
        """
        
        ax8.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Analyse des récompenses sauvegardée: {save_path}")
        plt.show()
    
    def print_summary(self):
        """Afficher un résumé dans la console"""
        stats = self.get_statistics()
        n_episodes = len(self.episode_rewards)
        
        print(f"\n{'='*80}")
        print(f"📊 ANALYSE DES RÉCOMPENSES - RÉSUMÉ")
        print(f"{'='*80}")
        print(f"\n🎯 RÉCOMPENSES TOTALES (par épisode)")
        print(f"  Nombre d'épisodes:      {n_episodes:>10}")
        print(f"  Moyenne:                {stats['reward_mean']:>15.4e}")
        print(f"  Écart-type:             {stats['reward_std']:>15.4e}")
        print(f"  CV (coef. variation):   {stats['reward_std']/abs(stats['reward_mean']):>15.2%}")
        print(f"  Médiane:                {stats['reward_median']:>15.4e}")
        print(f"  Min:                    {stats['reward_min']:>15.4e}")
        print(f"  Max:                    {stats['reward_max']:>15.4e}")
        print(f"  Écart (Max-Min):        {stats['reward_max'] - stats['reward_min']:>15.4e}")
        
        print(f"\n📈 RÉCOMPENSES PAR PAS")
        print(f"  Nombre de pas total:    {len(self.step_rewards):>10}")
        print(f"  Moyenne:                {stats['step_reward_mean']:>15.4e}")
        print(f"  Écart-type:             {stats['step_reward_std']:>15.4e}")
        print(f"  Min:                    {stats['step_reward_min']:>15.4e}")
        print(f"  Max:                    {stats['step_reward_max']:>15.4e}")
        
        print(f"\n💰 MÉTRIQUES CORRÉLÉES")
        print(f"  Revenu moyen:           {stats['revenue_mean']:>15,.2f} USDT")
        print(f"  Écart-type revenu:      {stats['revenue_std']:>15,.2f} USDT")
        print(f"  Pas effectifs moyen:    {stats['length_mean']:>15.1f}")
        print(f"  Inventaire final moyen: {stats['inventory_mean']:>15.2f} BTC")
        print(f"  Impact moyen:           {stats['impact_mean']:>15.2f} bps")
        
        print(f"\n🔍 ORDRE DE GRANDEUR")
        magnitude = int(np.floor(np.log10(abs(stats['reward_mean'])))) if stats['reward_mean'] != 0 else 0
        print(f"  Ordre de grandeur:      10^{magnitude}")
        print(f"  Récompense typique:     ~{stats['reward_mean']/10**magnitude:.2f} × 10^{magnitude}")
        
        print(f"\n{'='*80}\n")


def mini_training_with_reward_analysis(
    data_path: str,
    n_episodes: int = 100,
    analyze_every: int = 20,
    horizon_steps: int = 60,
    initial_inventory: float = 1000
):
    """
    Mini-entraînement avec analyse détaillée des récompenses
    
    Args:
        data_path: Chemin vers les données
        n_episodes: Nombre d'épisodes d'entraînement
        analyze_every: Fréquence d'analyse (tous les X épisodes)
        horizon_steps: Nombre de pas par épisode
        initial_inventory: Inventaire initial
    """
    
    print(f"\n{'='*80}")
    print(f"🔬 MINI-ENTRAÎNEMENT AVEC ANALYSE DES RÉCOMPENSES")
    print(f"{'='*80}")
    print(f"  Épisodes:           {n_episodes}")
    print(f"  Analyse tous les:   {analyze_every} épisodes")
    print(f"  Horizon:            {horizon_steps} pas")
    print(f"  Inventaire:         {initial_inventory} BTC")
    print(f"{'='*80}\n")
    
    # Créer l'environnement
    print("Initialisation de l'environnement...")
    env = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.0005,
        alpha=0.5,
        delta=0.2,
        random_start_prob=0
    )
    
    # Créer l'agent
    print("Initialisation de l'agent PPO...")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.3,
        lambda_gae=0.95,
        hidden_dims=[256, 256, 128],
        device='cpu'
    )
    
    # Créer l'analyseur
    analyzer = RewardAnalyzer()
    
    # Entraînement avec tracking
    print(f"\nDébut de l'entraînement ({n_episodes} épisodes)...\n")
    
    for episode in tqdm(range(n_episodes), desc="Entraînement"):
        state, _ = env.reset()
        done = False
        episode_rewards_list = []
        episode_impacts_list = []
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Stocker la récompense
            episode_rewards_list.append(reward)
            
            # Stocker l'impact si vente
            if info['quantity_sold'] > 1e-6:
                episode_impacts_list.append(info['temp_impact_relative'] * 10000)
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
        
        # Enregistrer l'épisode
        analyzer.record_episode(
            rewards=episode_rewards_list,
            revenue=info['total_revenue'],
            length=len([r for r in episode_rewards_list if r != 0]),  # Approximation
            inventory=info['inventory_remaining'],
            avg_impact=np.mean(episode_impacts_list) if episode_impacts_list else 0
        )
        
        # Mise à jour PPO
        if terminated:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state, deterministic=True)
        
        if (episode + 1) % 80 == 0:
            agent.update(next_value=next_value, epochs=4, batch_size=32)
        
        # Analyse intermédiaire
        if (episode + 1) % analyze_every == 0:
            print(f"\n{'─'*80}")
            print(f"📊 Analyse intermédiaire @ Épisode {episode + 1}/{n_episodes}")
            print(f"{'─'*80}")
            
            recent_rewards = analyzer.episode_rewards[-(analyze_every):]
            recent_revenues = analyzer.episode_revenues[-(analyze_every):]
            
            print(f"  Récompense (moy):    {np.mean(recent_rewards):>15.4e}")
            print(f"  Récompense (std):    {np.std(recent_rewards):>15.4e}")
            print(f"  Revenu (moy):        {np.mean(recent_revenues):>15,.2f} USDT")
            print(f"  Revenu (std):        {np.std(recent_revenues):>15,.2f} USDT")
            print(f"{'─'*80}\n")
    
    # Analyse finale
    print(f"\n{'='*80}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*80}\n")
    
    analyzer.print_summary()
    analyzer.plot_detailed_analysis(save_path='../sample/mini_training_reward_analysis.png')
    
    return agent, env, analyzer


def diagnose_reward_inconsistency(
    data_path: str,
    n_test_episodes: int = 20,
    horizon_steps: int = 60,
    initial_inventory: float = 1000
):
    """
    Diagnostiquer pourquoi la récompense totale est positive alors que la performance vs TWAP est négative
    
    Ce script simule plusieurs scénarios et affiche TOUS les détails de calcul
    """
    
    print(f"\n{'='*100}")
    print(f"🔬 DIAGNOSTIC DÉTAILLÉ : Incohérence Récompense vs Performance TWAP")
    print(f"{'='*100}\n")
    
    # Créer l'environnement
    env = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.0005,
        alpha=0.5,
        delta=0.2,
        random_start_prob=0.0  # Départ fixe pour reproductibilité
    )
    
    # Créer un agent aléatoire (pour tester la mécanique)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.3,
        lambda_gae=0.95,
        hidden_dims=[256, 256, 128],
        device='cpu'
    )
    
    # ═══════════════════════════════════════════════════════════════
    # SCÉNARIO 1 : Agent qui suit une stratégie simple
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'─'*100}")
    print(f"📋 SCÉNARIO 1 : Agent vend 5% à chaque pas (stratégie fixe)")
    print(f"{'─'*100}\n")
    
    for episode_idx in range(n_test_episodes):
        state, _ = env.reset()
        
        episode_rewards = []
        episode_agent_revenues = []
        episode_twap_revenues = []
        episode_normalizers = []
        episode_relative_gains = []
        episode_inventories = []
        episode_prices = []
        
        step = 0
        done = False
        
        print(f"\n{'▼'*100}")
        print(f"ÉPISODE {episode_idx + 1}")
        print(f"{'▼'*100}")
        print(f"Prix initial: {env.initial_price:.2f} USDT")
        print(f"Inventaire initial: {env.initial_inventory:.2f} BTC")
        print(f"Horizon: {horizon_steps} pas")
        
        while not done:
            # Action : vendre 5% (index 5)
            action = 5
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Stocker les données
            episode_rewards.append(reward)
            episode_agent_revenues.append(info['agent_revenue'])
            episode_twap_revenues.append(info['twap_revenue'])
            episode_normalizers.append(info['normalizer'])
            episode_relative_gains.append(info['relative_gain'])
            episode_inventories.append(info['inventory_remaining'])
            episode_prices.append(info['current_price'])
            
            # Afficher détails UNIQUEMENT pour les 5 premiers et 5 derniers pas
            if step < 5 or step >= horizon_steps - 5:
                print(f"\n  ┌─ PAS {step + 1}/{horizon_steps} {'─'*80}")
                print(f"  │ État:")
                print(f"  │   Inventaire restant:      {info['inventory_remaining']:>12.4f} BTC")
                print(f"  │   Temps restant:           {info['time_remaining']:>12} pas")
                print(f"  │   Prix actuel:             {info['current_price']:>12,.2f} USDT")
                print(f"  │")
                print(f"  │ Action Agent:")
                print(f"  │   Quantité vendue:         {info['quantity_sold']:>12.4f} BTC")
                print(f"  │   Prix d'exécution:        {info['execution_price']:>12,.2f} USDT")
                print(f"  │   Impact temporaire:       {info['temp_impact_relative']*10000:>12.4f} bps")
                print(f"  │   Revenu agent:            {info['agent_revenue']:>12,.2f} USDT")
                print(f"  │")
                print(f"  │ TWAP Hypothétique:")
                print(f"  │   Quantité TWAP:           {info['twap_revenue']/info['current_price'] if info['current_price'] > 0 else 0:>12.4f} BTC (estimé)")
                print(f"  │   Revenu TWAP:             {info['twap_revenue']:>12,.2f} USDT")
                print(f"  │")
                print(f"  │ Calcul Récompense:")
                print(f"  │   Gain relatif:            {info['relative_gain']:>12,.2f} USDT")
                print(f"  │   Normalizer:              {info['normalizer']:>12,.2f}")
                print(f"  │   Récompense brute:        {info['relative_gain']/info['normalizer'] if info['normalizer'] > 0 else 0:>12.6f}")
                print(f"  │   Récompense finale:       {reward:>12.6f}")
                print(f"  └{'─'*85}")
            
            elif step == 5:
                print(f"\n  [...] (pas 6 à {horizon_steps - 5} omis pour lisibilité)")
            
            state = next_state
            step += 1
        
        # ═══════════════════════════════════════════════════════════════
        # RÉSUMÉ DE L'ÉPISODE
        # ═══════════════════════════════════════════════════════════════
        
        total_reward = sum(episode_rewards)
        total_agent_revenue = sum(episode_agent_revenues)
        total_twap_revenue = sum(episode_twap_revenues)
        
        print(f"\n{'▲'*100}")
        print(f"RÉSUMÉ ÉPISODE {episode_idx + 1}")
        print(f"{'▲'*100}")
        
        print(f"\n📊 REVENUS TOTAUX:")
        print(f"  Agent:                     {total_agent_revenue:>15,.2f} USDT")
        print(f"  TWAP (hypothétique):       {total_twap_revenue:>15,.2f} USDT")
        print(f"  Différence:                {total_agent_revenue - total_twap_revenue:>15,.2f} USDT")
        
        print(f"\n🎁 RÉCOMPENSES:")
        print(f"  Somme récompenses:         {total_reward:>15.4f}")
        print(f"  Récompense moyenne/pas:    {np.mean(episode_rewards):>15.6f}")
        print(f"  Récompense médiane/pas:    {np.median(episode_rewards):>15.6f}")
        print(f"  Récompense min:            {np.min(episode_rewards):>15.6f}")
        print(f"  Récompense max:            {np.max(episode_rewards):>15.6f}")
        
        print(f"\n🎯 PERFORMANCE VS TWAP:")
        perf_vs_twap = ((total_agent_revenue - total_twap_revenue) / total_twap_revenue) * 100
        print(f"  Performance relative:      {perf_vs_twap:>15.4f} %")
        
        print(f"\n📦 INVENTAIRE:")
        print(f"  Initial:                   {initial_inventory:>15.2f} BTC")
        print(f"  Final:                     {episode_inventories[-1]:>15.2f} BTC")
        print(f"  Vendu:                     {initial_inventory - episode_inventories[-1]:>15.2f} BTC")  # ✅ CORRECTION
        print(f"  Taux de complétion:        {(1 - episode_inventories[-1]/initial_inventory)*100:>15.2f} %")  # ✅ CORRECTION
      
        print(f"\n💹 PRIX:")
        print(f"  Initial:                   {env.initial_price:>15,.2f} USDT")
        print(f"  Final:                     {episode_prices[-1]:>15,.2f} USDT")
        print(f"  Variation:                 {(episode_prices[-1]/env.initial_price - 1)*100:>15.2f} %")
        
        # ═══════════════════════════════════════════════════════════════
        # ANALYSE DÉTAILLÉE DES NORMALIZERS
        # ═══════════════════════════════════════════════════════════════
        
        print(f"\n🔍 ANALYSE DES NORMALIZERS:")
        print(f"  Moyenne:                   {np.mean(episode_normalizers):>15,.2f}")
        print(f"  Écart-type:                {np.std(episode_normalizers):>15,.2f}")
        print(f"  Min:                       {np.min(episode_normalizers):>15,.2f}")
        print(f"  Max:                       {np.max(episode_normalizers):>15,.2f}")
        print(f"  Somme totale:              {np.sum(episode_normalizers):>15,.2f}")
        
        # ═══════════════════════════════════════════════════════════════
        # RECONSTRUCTION MANUELLE DU TWAP
        # ═══════════════════════════════════════════════════════════════
        
        print(f"\n{'─'*100}")
        print(f"🧮 RECONSTRUCTION MANUELLE DU CALCUL TWAP")
        print(f"{'─'*100}\n")
        
        # Simuler TWAP réel (comme dans calculate_twap_performance)
        env.reset()  # Reset pour avoir même état initial
        
        twap_total_revenue_simulated = 0.0
        twap_inventory = initial_inventory
        
        print(f"Simulation TWAP (premiers 5 pas):")
        for t in range(min(5, horizon_steps)):
            time_remaining = horizon_steps - t
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.vol_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.vol_window)
            
            # TWAP adaptatif
            twap_quantity = twap_inventory / time_remaining
            
            # Impact
            twap_impact = env._calculate_temporary_impact(
                twap_quantity, realized_vol, rolling_sigma, rolling_volume
            )
            
            twap_execution_price = current_price * (1 - twap_impact)
            twap_revenue_step = twap_quantity * twap_execution_price
            twap_total_revenue_simulated += twap_revenue_step
            twap_inventory -= twap_quantity
            
            print(f"  Pas {t+1}: qty={twap_quantity:.4f}, price={current_price:.2f}, impact={twap_impact*10000:.4f} bps, revenue={twap_revenue_step:,.2f}")
            
            # Avancer le simulateur
            perm_impact = env._calculate_permanent_impact(twap_impact)
            perm_impact = np.clip(perm_impact, 0, 0.005)
            next_price, next_vol, next_volume = env.garch_simulator.step()
            next_price = next_price * (1 - perm_impact * 0.5)
            if next_price > env.initial_price * 2 or next_price < env.initial_price * 0.5:
                next_price = current_price * np.random.uniform(0.99, 1.01)
            env.prices_history.append(next_price)
            env.volumes_history.append(next_volume)
            new_vol = env._calculate_realized_volatility(np.array(env.prices_history))
            env.realized_vols_history.append(new_vol)
        
        print(f"\n  [... simulation complète omise ...]")
        
        # Comparer avec la somme des TWAP hypothétiques
        print(f"\n📊 COMPARAISON TWAP:")
        print(f"  TWAP simulé (reconstruction):  {twap_total_revenue_simulated:>15,.2f} USDT (premiers 5 pas)")
        print(f"  TWAP hypothétique (step()):    {total_twap_revenue:>15,.2f} USDT (somme totale)")
        print(f"  Différence:                    {abs(twap_total_revenue_simulated - total_twap_revenue):>15,.2f} USDT")
        
        # ═══════════════════════════════════════════════════════════════
        # VÉRIFICATION COHÉRENCE
        # ═══════════════════════════════════════════════════════════════
        
        print(f"\n{'─'*100}")
        print(f"✅ VÉRIFICATIONS DE COHÉRENCE")
        print(f"{'─'*100}\n")
        
        # Vérification 1 : Reward doit être proportionnel à (agent_rev - twap_rev)
        expected_sign_reward = np.sign(total_agent_revenue - total_twap_revenue)
        actual_sign_reward = np.sign(total_reward)
        
        print(f"1. Signe de la récompense:")
        print(f"   Agent vs TWAP:             {total_agent_revenue - total_twap_revenue:>15,.2f} USDT → signe: {'+' if expected_sign_reward > 0 else '-' if expected_sign_reward < 0 else '0'}")
        print(f"   Récompense totale:         {total_reward:>15.4f} → signe: {'+' if actual_sign_reward > 0 else '-' if actual_sign_reward < 0 else '0'}")
        print(f"   ✅ Cohérent" if expected_sign_reward == actual_sign_reward else f"   ❌ INCOHÉRENT !")
        
        # Vérification 2 : Reconstruction manuelle de la récompense
        reconstructed_reward = sum([
            (agent_rev - twap_rev) / norm if norm > 0 else 0
            for agent_rev, twap_rev, norm in zip(episode_agent_revenues, episode_twap_revenues, episode_normalizers)
        ])
        
        print(f"\n2. Reconstruction de la récompense:")
        print(f"   Récompense calculée (step): {total_reward:>15.4f}")
        print(f"   Récompense reconstruite:    {reconstructed_reward:>15.4f}")
        print(f"   Différence:                 {abs(total_reward - reconstructed_reward):>15.6f}")
        print(f"   ✅ Cohérent (diff < 1e-4)" if abs(total_reward - reconstructed_reward) < 1e-4 else f"   ❌ DIVERGENCE !")
        
        # Vérification 3 : Impact du normalizer
        avg_normalizer = np.mean(episode_normalizers)
        expected_reward_scale = (total_agent_revenue - total_twap_revenue) / (avg_normalizer * horizon_steps)
        
        print(f"\n3. Échelle de la récompense (approximation):")
        print(f"   Différence revenus:        {total_agent_revenue - total_twap_revenue:>15,.2f} USDT")
        print(f"   Normalizer moyen:          {avg_normalizer:>15,.2f}")
        print(f"   Nombre de pas:             {horizon_steps:>15}")
        print(f"   Récompense attendue (≈):   {expected_reward_scale:>15.4f}")
        print(f"   Récompense réelle:         {total_reward:>15.4f}")
        print(f"   Ratio:                     {total_reward / expected_reward_scale if abs(expected_reward_scale) > 1e-6 else 0:>15.2f}x")
        
        print(f"\n{'='*100}\n")
        
        # Arrêter après 3 épisodes pour ne pas surcharger
        if episode_idx >= 2:
            print(f"\n[Diagnostic limité à 3 épisodes pour lisibilité. Relancer pour plus de tests.]\n")
            break
    
    print(f"\n{'='*100}")
    print(f"🎯 FIN DU DIAGNOSTIC")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    
    # ═══════════════════════════════════════════════════════════════
    # OPTION 1 : Mini-entraînement classique (déjà présent)
    # ═══════════════════════════════════════════════════════════════
    
    agent, env, analyzer = mini_training_with_reward_analysis(
        data_path=data_path,
        n_episodes=1000,
        analyze_every=100,
        horizon_steps=60,
        initial_inventory=1000
    )
    """
    
    # ═══════════════════════════════════════════════════════════════
    # OPTION 2 : DIAGNOSTIC DÉTAILLÉ (NOUVEAU)
    # ═══════════════════════════════════════════════════════════════
    diagnose_reward_inconsistency(
        data_path=data_path,
        n_test_episodes=3,  # Nombre de scénarios à tester
        horizon_steps=60,
        initial_inventory=1000
    )
    
    print("\n🎉 Diagnostic terminé !")

    """