import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Supprimer les warnings GARCH répétitifs
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='The optimizer returned code 4')

sys.path.append(os.path.abspath(".."))
from src.environment.execution_env import OptimalExecutionEnv
from src.models.ppo_agent import PPOAgent


def calculate_twap_performance(env: OptimalExecutionEnv, horizon_steps: int, initial_inventory: float) -> float:
    """
    Calculer la performance d'une stratégie TWAP (Time-Weighted Average Price)
    en simulant l'exécution avec les impacts de marché
    
    Returns:
        float: Revenu total de la stratégie TWAP
    """
    # Quantité vendue à chaque pas (répartition égale)
    twap_quantity = initial_inventory / horizon_steps
    
    total_revenue = 0.0
    inventory = initial_inventory
    
    # Sauvegarder l'état initial de l'environnement
    initial_prices = env.prices_history.copy()
    initial_vols = env.realized_vols_history.copy()
    initial_volumes = env.volumes_history.copy()
    
    for step in range(horizon_steps):
        current_price = env.prices_history[-1]
        realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
        
        # Normalisateurs
        rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.vol_window)
        rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.vol_window)
        
        # Vendre la quantité TWAP
        quantity = min(twap_quantity, inventory)
        
        # Calculer l'impact temporaire
        temp_impact_relative = env._calculate_temporary_impact(
            quantity, realized_vol, rolling_sigma, rolling_volume
        )
        
        execution_price = current_price * (1 - temp_impact_relative)
        revenue = quantity * execution_price
        total_revenue += revenue
        inventory -= quantity
        
        # Impact permanent
        perm_impact_relative = env._calculate_permanent_impact(temp_impact_relative)
        perm_impact_relative = np.clip(perm_impact_relative, 0, 0.005)
        
        # Simuler le prochain prix
        next_price, next_vol, next_volume = env.garch_simulator.step()
        
        # Appliquer l'impact permanent
        impact_decay = 0.5
        next_price = next_price * (1 - perm_impact_relative * impact_decay)
        
        # Vérifier que le prix reste dans une plage raisonnable
        if next_price > env.initial_price * 2 or next_price < env.initial_price * 0.5:
            next_price = current_price * np.random.uniform(0.99, 1.01)
        
        # Mettre à jour l'historique
        env.prices_history.append(next_price)
        env.volumes_history.append(next_volume)
        
        new_realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
        env.realized_vols_history.append(new_realized_vol)
    
    # Restaurer l'état initial de l'environnement
    env.prices_history = initial_prices
    env.realized_vols_history = initial_vols
    env.volumes_history = initial_volumes
    
    return total_revenue


def run_validation(agent: PPOAgent, env: OptimalExecutionEnv, n_episodes: int, 
                   horizon_steps: int, initial_inventory: float) -> dict:
    """
    Exécute une boucle de validation propre sans entraînement
    
    Args:
        agent: Agent PPO à évaluer
        env: Environnement de validation (random_start_prob=0.0)
        n_episodes: Nombre d'épisodes de validation
        horizon_steps: Nombre de pas par épisode
        initial_inventory: Inventaire initial
        
    Returns:
        dict: Métriques moyennes sur tous les épisodes de validation
    """
    ep_revenues = []
    ep_lengths = []  # Nombre de pas avec inventaire > 0
    ep_impacts = []
    ep_inv_remaining = []
    ep_prices = []
    ep_twap_revenues = []
    ep_twap_comparisons = []
    
    for _ in tqdm(range(n_episodes), desc="Validation", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0  # NOUVEAU: Compteur de pas avec vente effective
        episode_impacts_local = []
        episode_prices_local = []
        
        # Calculer TWAP pour cet épisode
        twap_revenue = calculate_twap_performance(env, horizon_steps, initial_inventory)
        ep_twap_revenues.append(twap_revenue)
        
        # Exécuter l'épisode avec l'agent (mode déterministe)
        while not done:
            # IMPORTANT: deterministic=True pour l'évaluation
            action, _, _ = agent.select_action(state, deterministic=True)
            
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # NOUVEAU: Compter seulement les pas où on a vendu quelque chose
            if info['quantity_sold'] > 1e-6:  # Si on a effectivement vendu
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
                episode_prices_local.append(info['execution_price'])
                
            state = next_state
            step += 1
        
        # Stocker les métriques de cet épisode de validation
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)  # MODIFIÉ: Stocker effective_steps au lieu de step
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0)
        ep_inv_remaining.append(info['inventory_remaining'])
        ep_prices.append(np.mean(episode_prices_local) if episode_prices_local else env.initial_price)
        
        # Performance relative vs TWAP
        relative_performance = ((agent_revenue - twap_revenue) / twap_revenue) * 100
        ep_twap_comparisons.append(relative_performance)
    
    # Calculer les moyennes
    avg_revenue = np.mean(ep_revenues)
    avg_length = np.mean(ep_lengths)  # Maintenant c'est la moyenne des pas effectifs
    avg_impact = np.mean(ep_impacts)
    avg_inv_remaining = np.mean(ep_inv_remaining)
    avg_price = np.mean(ep_prices)
    avg_twap_revenue = np.mean(ep_twap_revenues)
    avg_twap_comparison = np.mean(ep_twap_comparisons)
    
    # Calcul du taux de complétion
    completion_rate = (1 - avg_inv_remaining / initial_inventory) * 100
    
    # Calcul du slippage moyen
    avg_slippage_bps = (avg_price / env.initial_price - 1) * 10000
    
    return {
        'avg_revenue': avg_revenue,
        'avg_length': avg_length,
        'avg_impact': avg_impact,
        'avg_inv_remaining': avg_inv_remaining,
        'avg_price': avg_price,
        'completion_rate': completion_rate,
        'avg_slippage_bps': avg_slippage_bps,
        'avg_twap_revenue': avg_twap_revenue,
        'avg_twap_comparison': avg_twap_comparison
    }


def print_validation_stats(episode: int, n_episodes: int, metrics: dict):
    """Afficher les statistiques de validation"""
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION @ ÉPISODE {episode}/{n_episodes}")
    print(f"{'='*80}")
    print(f"  💰 Revenu Agent:              {metrics['avg_revenue']:>15,.2f} USDT")
    print(f"  📊 Revenu TWAP:               {metrics['avg_twap_revenue']:>15,.2f} USDT")
    print(f"  🎯 Performance vs TWAP:       {metrics['avg_twap_comparison']:>15.2f} %")
    print(f"  📏 Longueur moyenne:          {metrics['avg_length']:>15.1f} pas")
    print(f"  📦 Inventaire restant:        {metrics['avg_inv_remaining']:>15.2f} BTC")
    print(f"  📈 Taux de complétion:        {metrics['completion_rate']:>15.1f} %")
    print(f"  💥 Impact moyen:              {metrics['avg_impact']:>15.2f} bps")
    print(f"  💵 Prix d'exécution moyen:    {metrics['avg_price']:>15,.2f} USDT")
    print(f"  📉 Slippage moyen:            {metrics['avg_slippage_bps']:>15.2f} bps")
    print(f"{'='*80}")


def train_ppo(
    data_path: str,
    n_episodes: int = 5000,
    horizon_steps: int = 60,
    initial_inventory: float = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    lambda_gae: float = 0.95,
    update_interval: int = 20,
    validation_interval: int = 100,  # NOUVEAU: Valider tous les 100 épisodes
    n_validation_episodes: int = 50,  # NOUVEAU: 50 épisodes de validation
    random_start_prob: float = 0.9,  # NOUVEAU: 90% de départs aléatoires en train
    save_interval: int = 100,
    model_save_path: str = '../models/ppo_execution.pth'
):
    """Entraîner l'agent PPO avec validation périodique"""
    
    # ═══════════════════════════════════════════════════════════════
    # 1. CRÉER DEUX ENVIRONNEMENTS (TRAIN ET VALIDATION)
    # ═══════════════════════════════════════════════════════════════
    
    print("Initialisation de l'environnement d'ENTRAÎNEMENT (90% départs aléatoires)...")
    env_train = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.0005,
        alpha=0.5,
        delta=0.01,
        random_start_prob=random_start_prob  # 90% de départs aléatoires
    )
    
    print("Initialisation de l'environnement de VALIDATION (0% départs aléatoires)...")
    env_val = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.0005,
        alpha=0.5,
        delta=0.05,
        random_start_prob=0.0  # 0% de départs aléatoires (tâche complète)
    )
    
    # Créer l'agent
    print("Initialisation de l'agent PPO...")
    agent = PPOAgent(
        state_dim=env_train.observation_space.shape[0],
        action_dim=env_train.action_space.n,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        lambda_gae=lambda_gae,
        hidden_dims=[128, 128],
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 2. MÉTRIQUES DE VALIDATION UNIQUEMENT (pour les graphiques)
    # ═══════════════════════════════════════════════════════════════
    
    validation_episodes = []  # Numéros d'épisodes de validation
    validation_revenues = []
    validation_lengths = []
    validation_avg_impacts = []
    validation_inventory_remaining = []
    validation_avg_execution_price = []
    validation_twap_revenues = []
    validation_twap_comparisons = []
    validation_completion_rates = []
    
    # ═══════════════════════════════════════════════════════════════
    # 3. BOUCLE D'ENTRAÎNEMENT PRINCIPALE
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"DÉBUT DE L'ENTRAÎNEMENT - {n_episodes} épisodes")
    print(f"Validation tous les {validation_interval} épisodes ({n_validation_episodes} épisodes/validation)")
    print(f"{'='*80}\n")
    
    main_pbar = tqdm(range(n_episodes), desc="Entraînement PPO")
    
    for episode in main_pbar:
        
        # ─────────────────────────────────────────────────────────────
        # PHASE D'ENTRAÎNEMENT (1 épisode)
        # ─────────────────────────────────────────────────────────────
        
        state, _ = env_train.reset()
        done = False
        step = 0
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            step += 1
        
        # Calculer next_value pour GAE
        if terminated:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state, deterministic=True)
        
        # Mise à jour PPO
        if (episode + 1) % update_interval == 0:
            agent.update(next_value=next_value, epochs=4, batch_size=64)
        
        # ─────────────────────────────────────────────────────────────
        # PHASE DE VALIDATION (Périodiquement)
        # ─────────────────────────────────────────────────────────────
        
        if (episode + 1) % validation_interval == 0:
            
            # Exécuter la validation
            val_metrics = run_validation(
                agent, 
                env_val, 
                n_validation_episodes,
                horizon_steps,
                initial_inventory
            )
            
            # Stocker les métriques pour les graphiques
            validation_episodes.append(episode + 1)
            validation_revenues.append(val_metrics['avg_revenue'])
            validation_lengths.append(val_metrics['avg_length'])
            validation_avg_impacts.append(val_metrics['avg_impact'])
            validation_inventory_remaining.append(val_metrics['avg_inv_remaining'])
            validation_avg_execution_price.append(val_metrics['avg_price'])
            validation_twap_revenues.append(val_metrics['avg_twap_revenue'])
            validation_twap_comparisons.append(val_metrics['avg_twap_comparison'])
            validation_completion_rates.append(val_metrics['completion_rate'])
            
            # Afficher les stats de validation
            print_validation_stats(episode + 1, n_episodes, val_metrics)
            
            # Mettre à jour la barre de progression avec les métriques clés
            main_pbar.set_postfix({
                'Val_Revenue': f"{val_metrics['avg_revenue']:.0f}",
                'Val_vs_TWAP': f"{val_metrics['avg_twap_comparison']:.2f}%",
                'Val_Length': f"{val_metrics['avg_length']:.1f}"
            })
        
        # Sauvegarder le modèle
        if (episode + 1) % save_interval == 0:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            agent.save(model_save_path)
            print(f"\n💾 Modèle sauvegardé: {model_save_path}")
    
    # Sauvegarder le modèle final
    agent.save(model_save_path)
    print(f"\n{'='*80}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"💾 Modèle final sauvegardé: {model_save_path}")
    print(f"{'='*80}\n")
    
    # Visualiser les résultats de VALIDATION
    plot_validation_results(
        validation_episodes,
        validation_revenues,
        validation_lengths,
        validation_avg_impacts,
        validation_inventory_remaining,
        validation_avg_execution_price,
        validation_twap_revenues,
        validation_twap_comparisons,
        validation_completion_rates,
        initial_inventory
    )
    
    return agent, env_val


def plot_validation_results(episodes, revenues, lengths, avg_impacts, inv_remaining, 
                           exec_prices, twap_revenues, twap_comparisons, completion_rates,
                           initial_inventory):
    """Visualiser UNIQUEMENT les résultats de validation (propres et non bruités)"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Résultats de VALIDATION - Agent PPO vs TWAP', fontsize=16, fontweight='bold')
    
    # 1. Revenus Agent vs TWAP
    ax = axes[0, 0]
    ax.plot(episodes, revenues, 'o-', label='Agent PPO', color='steelblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues, 's-', label='TWAP', color='orange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux: Agent vs TWAP')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 2. Performance relative vs TWAP
    ax = axes[0, 1]
    colors = ['green' if x >= 0 else 'red' for x in twap_comparisons]
    # CORRECTION: Calculer la largeur des barres automatiquement
    if len(episodes) > 1:
        bar_width = (episodes[1] - episodes[0]) * 0.8
    else:
        bar_width = 80  # Largeur par défaut
    ax.bar(episodes, twap_comparisons, color=colors, alpha=0.7, width=bar_width)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Performance Agent vs TWAP (%)')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Amélioration (%)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Longueurs d'épisode
    ax = axes[1, 0]
    ax.plot(episodes, lengths, 'o-', label='Longueur Agent', color='coral', linewidth=2, markersize=4)
    ax.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='TWAP (60 pas)', linewidth=2)
    ax.set_title('Longueur des Épisodes (Validation)')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Nombre de pas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 65])
    
    # 4. Impact de marché
    ax = axes[1, 1]
    ax.plot(episodes, avg_impacts, 'o-', label='Impact moyen (bps)', color='purple', linewidth=2, markersize=4)
    ax.set_title('Impact de Marché Moyen (Validation)')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Impact (basis points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Taux de complétion
    ax = axes[2, 0]
    ax.plot(episodes, completion_rates, 'o-', label='Taux de complétion', color='teal', linewidth=2, markersize=4)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Cible (100%)', linewidth=2)
    ax.set_title('Taux de Complétion (Validation)')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Complétion (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([95, 105])
    
    # 6. Statistiques finales (texte)
    ax = axes[2, 1]
    ax.axis('off')
    
    final_revenue = revenues[-1]
    final_twap = twap_revenues[-1]
    final_improvement = twap_comparisons[-1]
    final_length = lengths[-1]
    final_impact = avg_impacts[-1]
    final_completion = completion_rates[-1]
    
    stats_text = f"""
    📊 STATISTIQUES FINALES (Validation)
    {'─'*40}
    
    💰 Revenu Agent:          {final_revenue:,.2f} USDT
    📊 Revenu TWAP:           {final_twap:,.2f} USDT
    🎯 Performance vs TWAP:   {final_improvement:+.2f} %
    
    📏 Longueur moyenne:      {final_length:.1f} pas
    💥 Impact moyen:          {final_impact:.2f} bps
    📈 Taux complétion:       {final_completion:.1f} %
    
    {'─'*40}
    Nombre de validations: {len(episodes)}
    Épisodes d'entraînement: {episodes[-1]}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('../sample/validation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphiques de validation sauvegardés: ../sample/validation_results.png")
    plt.show()


if __name__ == "__main__":
    data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    
    # Entraîner l'agent avec validation périodique
    agent, env = train_ppo(
        data_path=data_path,
        n_episodes=1000,
        horizon_steps=60,
        initial_inventory=1000,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        lambda_gae=0.95,
        update_interval=30,
        validation_interval=100,      # Valider tous les 100 épisodes
        n_validation_episodes=50,     # 50 épisodes de validation
        random_start_prob=0.9,        # 90% départs aléatoires en train
        save_interval=100
    )