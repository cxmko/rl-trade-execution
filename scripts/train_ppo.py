import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import seaborn as sns

# Supprimer les warnings GARCH répétitifs
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='The optimizer returned code 4')

sys.path.append(os.path.abspath(".."))
from src.environment.execution_env import OptimalExecutionEnv
from src.models.ppo_agent import PPOAgent


def calculate_twap_performance(env: OptimalExecutionEnv, horizon_steps: int, initial_inventory: float) -> dict:
    """
    Calculer la performance d'une stratégie TWAP (Time-Weighted Average Price)
    en simulant l'exécution avec les impacts de marché
    
    Returns:
        dict: Revenu total et impact moyen de la stratégie TWAP
    """
    # Quantité vendue à chaque pas (répartition égale)
    twap_quantity = initial_inventory / horizon_steps
    
    total_revenue = 0.0
    inventory = initial_inventory
    impacts = []
    
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
        impacts.append(temp_impact_relative * 10000)  # En bps
        
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
    
    return {
        'revenue': total_revenue,
        'avg_impact': np.mean(impacts)
    }


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
        dict: Métriques moyennes et médianes sur tous les épisodes de validation
    """
    ep_revenues = []
    ep_lengths = []
    ep_total_steps = []
    ep_impacts = []
    ep_inv_remaining = []
    ep_prices = []
    ep_twap_revenues = []
    ep_twap_impacts = []
    ep_twap_comparisons = []
    
    for _ in tqdm(range(n_episodes), desc="Validation", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0
        episode_impacts_local = []
        episode_prices_local = []
        
        # Calculer TWAP pour cet épisode
        twap_metrics = calculate_twap_performance(env, horizon_steps, initial_inventory)
        ep_twap_revenues.append(twap_metrics['revenue'])
        ep_twap_impacts.append(twap_metrics['avg_impact'])
        
        # Exécuter l'épisode avec l'agent (mode déterministe)
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info['quantity_sold'] > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
                episode_prices_local.append(info['execution_price'])
                
            state = next_state
            step += 1
        
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)
        ep_total_steps.append(step)
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0)
        ep_inv_remaining.append(info['inventory_remaining'])
        ep_prices.append(np.mean(episode_prices_local) if episode_prices_local else env.initial_price)
        
        relative_performance = ((agent_revenue - twap_metrics['revenue']) / twap_metrics['revenue']) * 100
        ep_twap_comparisons.append(relative_performance)
    
    # Calculer les métriques
    avg_revenue = np.mean(ep_revenues)
    median_revenue = np.median(ep_revenues)
    avg_length = np.mean(ep_lengths)
    median_length = np.median(ep_lengths)
    avg_total_steps = np.mean(ep_total_steps)
    median_total_steps = np.median(ep_total_steps)
    avg_impact = np.mean(ep_impacts)
    median_impact = np.median(ep_impacts)
    avg_inv_remaining = np.mean(ep_inv_remaining)
    avg_price = np.mean(ep_prices)
    avg_twap_revenue = np.mean(ep_twap_revenues)
    median_twap_revenue = np.median(ep_twap_revenues)
    avg_twap_impact = np.mean(ep_twap_impacts)
    avg_twap_comparison = np.mean(ep_twap_comparisons)
    median_twap_comparison = np.median(ep_twap_comparisons)
    
    completion_rate = (1 - avg_inv_remaining / initial_inventory) * 100
    avg_slippage_bps = (avg_price / env.initial_price - 1) * 10000
    
    return {
        'avg_revenue': avg_revenue,
        'median_revenue': median_revenue,
        'avg_length': avg_length,
        'median_length': median_length,
        'avg_total_steps': avg_total_steps,
        'median_total_steps': median_total_steps,
        'avg_impact': avg_impact,
        'median_impact': median_impact,
        'avg_inv_remaining': avg_inv_remaining,
        'avg_price': avg_price,
        'completion_rate': completion_rate,
        'avg_slippage_bps': avg_slippage_bps,
        'avg_twap_revenue': avg_twap_revenue,
        'median_twap_revenue': median_twap_revenue,
        'avg_twap_impact': avg_twap_impact,
        'avg_twap_comparison': avg_twap_comparison,
        'median_twap_comparison': median_twap_comparison
    }


def run_final_validation(agent: PPOAgent, env: OptimalExecutionEnv, n_episodes: int,
                        horizon_steps: int, initial_inventory: float, agent_name: str) -> dict:
    """
    Validation finale détaillée avec tracking de l'inventaire
    
    Returns:
        dict: Statistiques + trajectoires d'inventaire moyennes
    """
    print(f"\n🔍 Validation finale: {agent_name} ({n_episodes} épisodes)")
    
    ep_revenues = []
    ep_lengths = []
    ep_impacts = []
    ep_twap_comparisons = []
    
    # Trajectoires d'inventaire (matrice: n_episodes × horizon_steps)
    inventory_trajectories = np.zeros((n_episodes, horizon_steps + 1))
    
    for ep_idx in tqdm(range(n_episodes), desc=f"  {agent_name}", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0
        episode_impacts_local = []
        
        # Stocker inventaire initial
        inventory_trajectories[ep_idx, 0] = initial_inventory
        
        # Calculer TWAP
        twap_metrics = calculate_twap_performance(env, horizon_steps, initial_inventory)
        
        # Exécuter l'épisode
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info['quantity_sold'] > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
            
            # Stocker l'inventaire à ce pas
            inventory_trajectories[ep_idx, step + 1] = info['inventory_remaining']
            
            state = next_state
            step += 1
        
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0)
        
        relative_performance = ((agent_revenue - twap_metrics['revenue']) / twap_metrics['revenue']) * 100
        ep_twap_comparisons.append(relative_performance)
    
    # Calculer trajectoire moyenne
    avg_inventory_trajectory = np.mean(inventory_trajectories, axis=0)
    
    return {
        'agent_name': agent_name,
        'avg_revenue': np.mean(ep_revenues),
        'median_revenue': np.median(ep_revenues),
        'avg_length': np.mean(ep_lengths),
        'median_length': np.median(ep_lengths),
        'avg_impact': np.mean(ep_impacts),
        'median_impact': np.median(ep_impacts),
        'avg_twap_comparison': np.mean(ep_twap_comparisons),
        'median_twap_comparison': np.median(ep_twap_comparisons),
        'avg_inventory_trajectory': avg_inventory_trajectory
    }


def print_validation_stats(episode: int, n_episodes: int, metrics: dict):
    """Afficher les statistiques de validation"""
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION @ ÉPISODE {episode}/{n_episodes}")
    print(f"{'='*80}")
    print(f"  💰 Revenu Agent (moy):        {metrics['avg_revenue']:>15,.2f} USDT")
    print(f"  💰 Revenu Agent (med):        {metrics['median_revenue']:>15,.2f} USDT")
    print(f"  📊 Revenu TWAP (moy):         {metrics['avg_twap_revenue']:>15,.2f} USDT")
    print(f"  📊 Revenu TWAP (med):         {metrics['median_twap_revenue']:>15,.2f} USDT")
    print(f"  🎯 Perf vs TWAP (moy):        {metrics['avg_twap_comparison']:>15.2f} %")
    print(f"  🎯 Perf vs TWAP (med):        {metrics['median_twap_comparison']:>15.2f} %")
    print(f"  📏 Pas effectifs (moy):       {metrics['avg_length']:>15.1f} pas")
    print(f"  📏 Pas effectifs (med):       {metrics['median_length']:>15.1f} pas")
    print(f"  📏 Pas totaux (moy):          {metrics['avg_total_steps']:>15.1f} pas")
    print(f"  📦 Inventaire restant:        {metrics['avg_inv_remaining']:>15.2f} BTC")
    print(f"  📈 Taux de complétion:        {metrics['completion_rate']:>15.1f} %")
    print(f"  💥 Impact moyen:              {metrics['avg_impact']:>15.2f} bps")
    print(f"  💥 Impact TWAP:               {metrics['avg_twap_impact']:>15.2f} bps")
    print(f"{'='*80}")


def print_final_validation_stats(results: list):
    """Afficher les statistiques de la validation finale"""
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION FINALE - COMPARAISON DES MODÈLES")
    print(f"{'='*80}\n")
    
    for res in results:
        print(f"🤖 {res['agent_name']}")
        print(f"  {'─'*76}")
        print(f"  💰 Revenu moyen:         {res['avg_revenue']:>15,.2f} USDT")
        print(f"  💰 Revenu médian:        {res['median_revenue']:>15,.2f} USDT")
        print(f"  🎯 Perf vs TWAP (moy):   {res['avg_twap_comparison']:>15.2f} %")
        print(f"  🎯 Perf vs TWAP (med):   {res['median_twap_comparison']:>15.2f} %")
        print(f"  📏 Pas effectifs (moy):  {res['avg_length']:>15.1f}")
        print(f"  📏 Pas effectifs (med):  {res['median_length']:>15.1f}")
        print(f"  💥 Impact moyen:         {res['avg_impact']:>15.2f} bps")
        print(f"  💥 Impact médian:        {res['median_impact']:>15.2f} bps")
        print()
    
    print(f"{'='*80}\n")


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
    validation_interval: int = 100,
    n_validation_episodes: int = 50,
    random_start_prob: float = 0.9,
    save_interval: int = 100,
    model_save_path: str = '../models/ppo_execution.pth',
    pretrained_model_path: str = None,
    override_epsilon: float = None
):
    """Entraîner l'agent PPO avec validation périodique et sauvegarde du meilleur modèle"""
    
    # ═══════════════════════════════════════════════════════════════
    # 1. CRÉER DEUX ENVIRONNEMENTS (TRAIN ET VALIDATION)
    # ═══════════════════════════════════════════════════════════════
    
    print("Initialisation de l'environnement d'ENTRAÎNEMENT...")
    env_train = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.0005,
        alpha=0.5,
        delta=0.1,
        random_start_prob=random_start_prob
    )
    
    print("Initialisation de l'environnement de VALIDATION...")
    env_val = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.0005,
        alpha=0.5,
        delta=0.05,
        random_start_prob=0.0
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 2. CRÉER L'AGENT
    # ═══════════════════════════════════════════════════════════════
    
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
    
    # Charger un modèle pré-entraîné si spécifié
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n{'='*80}")
        print(f"📦 CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ")
        print(f"{'='*80}")
        print(f"Chemin: {pretrained_model_path}")
        
        agent.load(pretrained_model_path)
        print(f"✅ Modèle chargé avec succès !")
        
        if override_epsilon is not None:
            old_epsilon = agent.epsilon
            agent.epsilon = override_epsilon
            print(f"🔧 Epsilon modifié: {old_epsilon:.3f} → {override_epsilon:.3f}")
        
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr
        print(f"🔧 Learning rate configuré: {lr:.6f}")
        print(f"{'='*80}\n")
    
    elif pretrained_model_path:
        print(f"\n⚠️  ATTENTION: Modèle introuvable: {pretrained_model_path}")
        print(f"   Démarrage FROM SCRATCH\n")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. MÉTRIQUES DE VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    validation_episodes = []
    validation_revenues_mean = []
    validation_revenues_median = []
    validation_lengths_mean = []
    validation_lengths_median = []
    validation_total_steps_mean = []
    validation_total_steps_median = []
    validation_avg_impacts = []
    validation_twap_impacts = []
    validation_inventory_remaining = []
    validation_avg_execution_price = []
    validation_twap_revenues_mean = []
    validation_twap_revenues_median = []
    validation_twap_comparisons_mean = []
    validation_twap_comparisons_median = []
    validation_completion_rates = []
    
    best_mean_performance = -np.inf
    best_median_performance = -np.inf
    best_mean_model_path = model_save_path.replace('.pth', '_best_mean.pth')
    best_median_model_path = model_save_path.replace('.pth', '_best_median.pth')
    
    # ═══════════════════════════════════════════════════════════════
    # 4. BOUCLE D'ENTRAÎNEMENT
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"DÉBUT DE L'ENTRAÎNEMENT - {n_episodes} épisodes")
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Mode: FINE-TUNING")
    else:
        print(f"Mode: FROM SCRATCH")
    print(f"Validation tous les {validation_interval} épisodes ({n_validation_episodes} épisodes/validation)")
    print(f"Epsilon actuel: {agent.epsilon:.3f}")
    print(f"{'='*80}\n")
    
    main_pbar = tqdm(range(n_episodes), desc="Entraînement PPO")
    
    for episode in main_pbar:
        state, _ = env_train.reset()
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
        
        if terminated:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state, deterministic=True)
        
        if (episode + 1) % update_interval == 0:
            agent.update(next_value=next_value, epochs=4, batch_size=64)
        
        if (episode + 1) % validation_interval == 0:
            val_metrics = run_validation(
                agent, env_val, n_validation_episodes,
                horizon_steps, initial_inventory
            )
            
            validation_episodes.append(episode + 1)
            validation_revenues_mean.append(val_metrics['avg_revenue'])
            validation_revenues_median.append(val_metrics['median_revenue'])
            validation_lengths_mean.append(val_metrics['avg_length'])
            validation_lengths_median.append(val_metrics['median_length'])
            validation_total_steps_mean.append(val_metrics['avg_total_steps'])
            validation_total_steps_median.append(val_metrics['median_total_steps'])
            validation_avg_impacts.append(val_metrics['avg_impact'])
            validation_twap_impacts.append(val_metrics['avg_twap_impact'])
            validation_inventory_remaining.append(val_metrics['avg_inv_remaining'])
            validation_avg_execution_price.append(val_metrics['avg_price'])
            validation_twap_revenues_mean.append(val_metrics['avg_twap_revenue'])
            validation_twap_revenues_median.append(val_metrics['median_twap_revenue'])
            validation_twap_comparisons_mean.append(val_metrics['avg_twap_comparison'])
            validation_twap_comparisons_median.append(val_metrics['median_twap_comparison'])
            validation_completion_rates.append(val_metrics['completion_rate'])
            
            print_validation_stats(episode + 1, n_episodes, val_metrics)
            
            if val_metrics['avg_twap_comparison'] > best_mean_performance:
                best_mean_performance = val_metrics['avg_twap_comparison']
                agent.save(best_mean_model_path)
                print(f"💎 Nouveau meilleur modèle (MOYENNE) ! Performance: {best_mean_performance:.2f}% vs TWAP")
            
            if val_metrics['median_twap_comparison'] > best_median_performance:
                best_median_performance = val_metrics['median_twap_comparison']
                agent.save(best_median_model_path)
                print(f"💎 Nouveau meilleur modèle (MÉDIANE) ! Performance: {best_median_performance:.2f}% vs TWAP")
            
            main_pbar.set_postfix({
                'Val_Mean': f"{val_metrics['avg_twap_comparison']:.2f}%",
                'Val_Median': f"{val_metrics['median_twap_comparison']:.2f}%",
                'Best_Mean': f"{best_mean_performance:.2f}%",
                'Best_Median': f"{best_median_performance:.2f}%"
            })
        
        if (episode + 1) % save_interval == 0:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            agent.save(model_save_path)
            print(f"\n💾 Modèle sauvegardé: {model_save_path}")
    
    agent.save(model_save_path)
    print(f"\n{'='*80}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"💾 Modèle final: {model_save_path}")
    print(f"💎 Meilleur (moyenne): {best_mean_model_path} ({best_mean_performance:.2f}%)")
    print(f"💎 Meilleur (médiane): {best_median_model_path} ({best_median_performance:.2f}%)")
    print(f"{'='*80}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. VALIDATION FINALE (100 épisodes) + ÉVOLUTION INVENTAIRE
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"🔬 VALIDATION FINALE - 100 ÉPISODES")
    print(f"{'='*80}\n")
    
    final_results = []
    
    # Charger et valider les 3 modèles
    agents_to_test = []
    
    # Modèle final
    agents_to_test.append(('Modèle Final', agent))
    
    # Meilleur (moyenne)
    if os.path.exists(best_mean_model_path):
        best_mean_agent = PPOAgent(
            state_dim=env_val.observation_space.shape[0],
            action_dim=env_val.action_space.n,
            lr=lr, gamma=gamma, epsilon=epsilon, lambda_gae=lambda_gae,
            hidden_dims=[128, 128]
        )
        best_mean_agent.load(best_mean_model_path)
        agents_to_test.append(('Meilleur (Moyenne)', best_mean_agent))
    
    # Meilleur (médiane)
    if os.path.exists(best_median_model_path):
        best_median_agent = PPOAgent(
            state_dim=env_val.observation_space.shape[0],
            action_dim=env_val.action_space.n,
            lr=lr, gamma=gamma, epsilon=epsilon, lambda_gae=lambda_gae,
            hidden_dims=[128, 128]
        )
        best_median_agent.load(best_median_model_path)
        agents_to_test.append(('Meilleur (Médiane)', best_median_agent))
    
    # Valider chaque agent
    for agent_name, test_agent in agents_to_test:
        result = run_final_validation(
            test_agent, env_val, 100,
            horizon_steps, initial_inventory, agent_name
        )
        final_results.append(result)
    
    # Afficher les résultats
    print_final_validation_stats(final_results)
    
    # ═══════════════════════════════════════════════════════════════
    # 6. GRAPHIQUES
    # ═══════════════════════════════════════════════════════════════
    
    plot_validation_results(
        validation_episodes,
        validation_revenues_mean,
        validation_revenues_median,
        validation_lengths_mean,
        validation_lengths_median,
        validation_total_steps_mean,
        validation_total_steps_median,
        validation_avg_impacts,
        validation_twap_impacts,
        validation_inventory_remaining,
        validation_avg_execution_price,
        validation_twap_revenues_mean,
        validation_twap_revenues_median,
        validation_twap_comparisons_mean,
        validation_twap_comparisons_median,
        validation_completion_rates,
        initial_inventory
    )
    
    plot_final_inventory_evolution(final_results, horizon_steps, initial_inventory)
    
    return agent, env_val


def plot_final_inventory_evolution(results: list, horizon_steps: int, initial_inventory: float):
    """Visualiser l'évolution moyenne de l'inventaire pour chaque modèle"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_steps = np.arange(0, horizon_steps + 1)
    colors = ['steelblue', 'darkgreen', 'coral']
    
    for idx, res in enumerate(results):
        ax.plot(time_steps, res['avg_inventory_trajectory'], 
                label=res['agent_name'], color=colors[idx % len(colors)],
                linewidth=2.5, marker='o', markersize=4, markevery=5)
    
    # TWAP de référence (linéaire)
    twap_trajectory = np.linspace(initial_inventory, 0, horizon_steps + 1)
    ax.plot(time_steps, twap_trajectory, '--', label='TWAP (référence)',
            color='gray', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Pas de temps', fontsize=13)
    ax.set_ylabel('Inventaire (BTC)', fontsize=13)
    ax.set_title('Évolution Moyenne de l\'Inventaire\n(100 épisodes de validation)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, horizon_steps])
    ax.set_ylim([0, initial_inventory * 1.05])
    
    plt.tight_layout()
    plt.savefig('../sample/inventory_evolution.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Évolution de l'inventaire sauvegardée: ../sample/inventory_evolution.png")
    plt.show()


def plot_validation_results(episodes, revenues_mean, revenues_median, lengths_mean, lengths_median,
                           total_steps_mean, total_steps_median, avg_impacts, twap_impacts,
                           inv_remaining, exec_prices, twap_revenues_mean, twap_revenues_median, 
                           twap_comparisons_mean, twap_comparisons_median, completion_rates,
                           initial_inventory):
    """Visualiser les résultats de validation"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Résultats de VALIDATION - Agent PPO vs TWAP', 
                 fontsize=16, fontweight='bold')
    
    # 1. Revenus (Moyenne)
    ax = axes[0, 0]
    ax.plot(episodes, revenues_mean, 'o-', label='Agent PPO (moy)', color='steelblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues_mean, 's-', label='TWAP (moy)', color='orange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux (Moyenne)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 2. Revenus (Médiane)
    ax = axes[0, 1]
    ax.plot(episodes, revenues_median, 'o-', label='Agent PPO (med)', color='darkblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues_median, 's-', label='TWAP (med)', color='darkorange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux (Médiane)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 3. Performance vs TWAP
    ax = axes[0, 2]
    ax.plot(episodes, twap_comparisons_mean, 'o-', label='Moyenne', color='green', linewidth=2, markersize=4)
    ax.plot(episodes, twap_comparisons_median, 's-', label='Médiane', color='darkgreen', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Performance vs TWAP (%)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Amélioration (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Pas effectifs
    ax = axes[1, 0]
    ax.plot(episodes, lengths_mean, 'o-', label='Pas effectifs (moy)', color='coral', linewidth=2, markersize=4)
    ax.plot(episodes, lengths_median, 's-', label='Pas effectifs (med)', color='darkred', linewidth=2, markersize=4)
    ax.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='TWAP (60 pas)', linewidth=2)
    ax.set_title('Nombre de Pas avec Vente')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Nombre de pas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 65])
    
    # 5. Pas totaux
    ax = axes[1, 1]
    ax.plot(episodes, total_steps_mean, 'o-', label='Total (moy)', color='purple', linewidth=2, markersize=4)
    ax.plot(episodes, total_steps_median, 's-', label='Total (med)', color='indigo', linewidth=2, markersize=4)
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Horizon (60 pas)', linewidth=2)
    ax.set_title('Nombre Total de Pas')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Nombre de pas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([55, 65])
    
    # 6. Impact de marché (AVEC TWAP)
    ax = axes[1, 2]
    ax.plot(episodes, avg_impacts, 'o-', label='Agent PPO', color='purple', linewidth=2, markersize=4)
    ax.plot(episodes, twap_impacts, 's-', label='TWAP', color='orange', linewidth=2, markersize=4)
    ax.set_title('Impact de Marché Moyen')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Impact (basis points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Taux de complétion
    ax = axes[2, 0]
    ax.plot(episodes, completion_rates, 'o-', label='Taux de complétion', color='teal', linewidth=2, markersize=4)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Cible (100%)', linewidth=2)
    ax.set_title('Taux de Complétion')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Complétion (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([95, 105])
    
    # 8. Distribution Performance
    ax = axes[2, 1]
    if len(episodes) > 1:
        bar_width = (episodes[1] - episodes[0]) * 0.4
    else:
        bar_width = 40
    ax.bar([e - bar_width/2 for e in episodes], twap_comparisons_mean, 
           width=bar_width, label='Moyenne', color='green', alpha=0.7)
    ax.bar([e + bar_width/2 for e in episodes], twap_comparisons_median, 
           width=bar_width, label='Médiane', color='darkgreen', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Performance vs TWAP')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Amélioration (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 9. Statistiques finales
    ax = axes[2, 2]
    ax.axis('off')
    
    stats_text = f"""
    📊 STATISTIQUES FINALES
    {'─'*45}
    
    💰 Revenu Agent (moy):    {revenues_mean[-1]:,.0f}
    💰 Revenu Agent (med):    {revenues_median[-1]:,.0f}
    📊 Revenu TWAP (moy):     {twap_revenues_mean[-1]:,.0f}
    📊 Revenu TWAP (med):     {twap_revenues_median[-1]:,.0f}
    
    🎯 Perf vs TWAP (moy):    {twap_comparisons_mean[-1]:+.2f} %
    🎯 Perf vs TWAP (med):    {twap_comparisons_median[-1]:+.2f} %
    
    📏 Pas effectifs (moy):   {lengths_mean[-1]:.1f}
    📏 Pas effectifs (med):   {lengths_median[-1]:.1f}
    
    {'─'*45}
    Validations: {len(episodes)}
    Épisodes: {episodes[-1]}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('../sample/validation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphiques de validation sauvegardés: ../sample/validation_results.png")
    plt.show()


if __name__ == "__main__":
    data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    

    # ═══════════════════════════════════════════════════════════════
    # EXEMPLE 2: FINE-TUNING (changer directement lr et epsilon)
    # ═══════════════════════════════════════════════════════════════
    agent, env = train_ppo(
        data_path=data_path,
        n_episodes=5000,
        horizon_steps=60,
        initial_inventory=1000,
        lr=3e-4/2,                           # ✅ Changer directement ici (10x plus petit)
        gamma=0.99,
        epsilon=0.3,                       # Valeur de base
        lambda_gae=0.95,
        update_interval=80,
        validation_interval=100,
        n_validation_episodes=200,
        random_start_prob=0.9,
        save_interval=100,
        
        pretrained_model_path='../models/ppo_execution_best_median_004.pth',
        override_epsilon=0.15               # ✅ Override seulement epsilon
    )