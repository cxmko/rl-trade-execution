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



def run_validation(agent: PPOAgent, env: OptimalExecutionEnv, n_episodes: int, 
                   horizon_steps: int, initial_inventory: float,
                   debug_mode: bool = False) -> dict:
    """
    Exécute une boucle de validation propre sans entraînement
    
    ✅ CORRECTION FINALE : Agent et TWAP calculés dans la MÊME boucle
    """
    ep_revenues = []
    ep_lengths = []  # ✅ Liste complète pour min/max
    ep_impacts = []
    ep_inv_remaining = []
    ep_prices = []
    ep_twap_revenues = []
    ep_twap_impacts = []
    ep_twap_comparisons = []
    ep_rewards = []
    
    for ep_idx in tqdm(range(n_episodes), desc="Validation", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0
        episode_impacts_local = []
        episode_prices_local = []
        episode_reward_total = 0.0
        
        twap_inventory = initial_inventory
        twap_total_revenue = 0.0
        twap_impacts_local = []
        
        if debug_mode:
            episode_step_rewards = []
            episode_agent_revenues = []
            episode_twap_revenues_steps = []
            episode_normalizers = []
            episode_relative_gains = []
        
        while not done:
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.vol_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.vol_window)
            time_remaining = env.horizon_steps - env.current_step
            
            if time_remaining > 0 and twap_inventory > 1e-6:
                twap_quantity = twap_inventory / time_remaining
                twap_quantity = min(twap_quantity, twap_inventory)
                
                twap_impact = env._calculate_temporary_impact(
                    twap_quantity, realized_vol, rolling_sigma, rolling_volume
                )
                twap_execution_price = current_price * (1 - twap_impact)
                twap_revenue_step = twap_quantity * twap_execution_price
                
                twap_total_revenue += twap_revenue_step
                twap_inventory -= twap_quantity
                twap_impacts_local.append(twap_impact * 10000)
            else:
                twap_revenue_step = 0.0
            
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward_total += reward
            
            if debug_mode:
                episode_step_rewards.append(reward)
                episode_agent_revenues.append(info.get('agent_revenue', 0.0))
                episode_twap_revenues_steps.append(twap_revenue_step)
                episode_normalizers.append(info.get('normalizer', 1.0))
                episode_relative_gains.append(info.get('relative_gain', 0.0))
            
            if info['quantity_sold'] > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
                episode_prices_local.append(info['execution_price'])
            
            state = next_state
            step += 1
        
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)  # ✅ Stocker pour chaque épisode
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0)
        ep_inv_remaining.append(info['inventory_remaining'])
        ep_prices.append(np.mean(episode_prices_local) if episode_prices_local else env.initial_price)
        ep_rewards.append(episode_reward_total)
        
        ep_twap_revenues.append(twap_total_revenue)
        ep_twap_impacts.append(np.mean(twap_impacts_local) if twap_impacts_local else 0)
        
        if twap_total_revenue > 1e-6:
            relative_performance = ((agent_revenue - twap_total_revenue) / twap_total_revenue) * 100
        else:
            relative_performance = 0.0
        ep_twap_comparisons.append(relative_performance)
        
        if debug_mode:
            sign_reward = np.sign(episode_reward_total)
            sign_perf = np.sign(relative_performance)
            
            if sign_reward != sign_perf and abs(episode_reward_total) > 0.1:
                total_agent_revenue = sum(episode_agent_revenues)
                total_twap_revenue = sum(episode_twap_revenues_steps)
                
                print(f"\n{'🚨'*50}")
                print(f"INCOHÉRENCE DÉTECTÉE EN VALIDATION - ÉPISODE {ep_idx + 1}/{n_episodes}")
                print(f"{'🚨'*50}\n")
                
                print(f"📊 MÉTRIQUES GLOBALES:")
                print(f"  Récompense totale:           {episode_reward_total:>15.4f} → signe: {'+' if sign_reward > 0 else '-' if sign_reward < 0 else '0'}")
                print(f"  Revenu Agent (total):        {agent_revenue:>15,.2f} USDT")
                print(f"  Revenu TWAP (total):         {twap_total_revenue:>15,.2f} USDT")
                print(f"  Différence:                  {agent_revenue - twap_total_revenue:>15,.2f} USDT")
                print(f"  Performance vs TWAP:         {relative_performance:>15.4f} % → signe: {'+' if sign_perf > 0 else '-' if sign_perf < 0 else '0'}")
                
                print(f"\n  📌 DÉTAIL DES REVENUS STEP-BY-STEP:")
                print(f"     Agent revenue (sum steps):  {total_agent_revenue:>15,.2f} USDT")
                print(f"     TWAP revenue (sum steps):   {total_twap_revenue:>15,.2f} USDT")
                print(f"     Diff step-by-step:          {total_agent_revenue - total_twap_revenue:>15,.2f} USDT")
                
                print(f"\n❌ INCOHÉRENCE : Récompense ({'+' if sign_reward > 0 else '-' if sign_reward < 0 else '0'}) ≠ Performance ({'+' if sign_perf > 0 else '-' if sign_perf < 0 else '0'})")
                
                # Afficher détails (premiers/derniers pas)
                print(f"\n{'─'*100}")
                print(f"DÉTAILS DES TRANSACTIONS (premiers 5 et derniers 5 pas)")
                print(f"{'─'*100}\n")
                
                n_steps = len(episode_step_rewards)
                indices_to_show = list(range(min(5, n_steps))) + list(range(max(n_steps - 5, 5), n_steps))
                
                for idx in indices_to_show:
                    if idx < 0 or idx >= n_steps:
                        continue
                    
                    print(f"  Pas {idx + 1}/{n_steps}:")
                    print(f"    Agent revenue:       {episode_agent_revenues[idx]:>15,.2f} USDT")
                    print(f"    TWAP revenue:        {episode_twap_revenues_steps[idx]:>15,.2f} USDT")
                    print(f"    Relative gain:       {episode_relative_gains[idx]:>15,.2f} USDT")
                    print(f"    Normalizer:          {episode_normalizers[idx]:>15,.2f}")
                    print(f"    Reward (final):      {episode_step_rewards[idx]:>15.6f}")
                    print()
                    
                    if idx == 4 and n_steps > 10:
                        print(f"  [...] (pas 6 à {n_steps - 5} omis)\n")
                
                print(f"\n{'🚨'*50}")
                print(f"VALIDATION INTERROMPUE")
                print(f"{'🚨'*50}\n")
                
                return {
                    'avg_revenue': np.mean(ep_revenues) if ep_revenues else 0,
                    'median_revenue': np.median(ep_revenues) if ep_revenues else 0,
                    'avg_length': np.mean(ep_lengths) if ep_lengths else 0,
                    'median_length': np.median(ep_lengths) if ep_lengths else 0,
                    'min_length': np.min(ep_lengths) if ep_lengths else 0,  # ✅ NOUVEAU
                    'max_length': np.max(ep_lengths) if ep_lengths else 0,  # ✅ NOUVEAU
                    'all_lengths': ep_lengths,  # ✅ NOUVEAU : Pour le graphique
                    'avg_impact': np.mean(ep_impacts) if ep_impacts else 0,
                    'median_impact': np.median(ep_impacts) if ep_impacts else 0,
                    'avg_inv_remaining': np.mean(ep_inv_remaining) if ep_inv_remaining else 0,
                    'avg_price': np.mean(ep_prices) if ep_prices else 0,
                    'completion_rate': 0,
                    'avg_slippage_bps': 0,
                    'avg_twap_revenue': np.mean(ep_twap_revenues) if ep_twap_revenues else 0,
                    'median_twap_revenue': np.median(ep_twap_revenues) if ep_twap_revenues else 0,
                    'avg_twap_impact': np.mean(ep_twap_impacts) if ep_twap_impacts else 0,
                    'avg_twap_comparison': np.mean(ep_twap_comparisons) if ep_twap_comparisons else 0,
                    'median_twap_comparison': np.median(ep_twap_comparisons) if ep_twap_comparisons else 0,
                    'avg_reward': np.mean(ep_rewards) if ep_rewards else 0,
                    'median_reward': np.median(ep_rewards) if ep_rewards else 0,
                    'inconsistency_detected': True
                }
    
    # Calculer les métriques finales
    avg_revenue = np.mean(ep_revenues)
    median_revenue = np.median(ep_revenues)
    avg_length = np.mean(ep_lengths)
    median_length = np.median(ep_lengths)
    min_length = np.min(ep_lengths)  # ✅ NOUVEAU
    max_length = np.max(ep_lengths)  # ✅ NOUVEAU
    avg_impact = np.mean(ep_impacts)
    median_impact = np.median(ep_impacts)
    avg_inv_remaining = np.mean(ep_inv_remaining)
    avg_price = np.mean(ep_prices)
    avg_twap_revenue = np.mean(ep_twap_revenues)
    median_twap_revenue = np.median(ep_twap_revenues)
    avg_twap_impact = np.mean(ep_twap_impacts)
    avg_twap_comparison = np.mean(ep_twap_comparisons)
    median_twap_comparison = np.median(ep_twap_comparisons)
    avg_reward = np.mean(ep_rewards)
    median_reward = np.median(ep_rewards)
    
    completion_rate = (1 - avg_inv_remaining / initial_inventory) * 100
    avg_slippage_bps = (avg_price / env.initial_price - 1) * 10000
    
    return {
        'avg_revenue': avg_revenue,
        'median_revenue': median_revenue,
        'avg_length': avg_length,
        'median_length': median_length,
        'min_length': min_length,  # ✅ NOUVEAU
        'max_length': max_length,  # ✅ NOUVEAU
        'all_lengths': ep_lengths,  # ✅ NOUVEAU : Liste complète pour graphique
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
        'median_twap_comparison': median_twap_comparison,
        'avg_reward': avg_reward,
        'median_reward': median_reward,
        'inconsistency_detected': False
    }


def run_final_validation(agent: PPOAgent, env: OptimalExecutionEnv, n_episodes: int,
                        horizon_steps: int, initial_inventory: float, agent_name: str) -> dict:
    """
    Validation finale détaillée avec tracking de l'inventaire
    
    ✅ CORRECTION : Agent et TWAP calculés dans la MÊME boucle (comme run_validation)
    
    Returns:
        dict: Statistiques + trajectoires d'inventaire moyennes
    """
    print(f"\n🔍 Validation finale: {agent_name} ({n_episodes} épisodes)")
    
    ep_revenues = []
    ep_lengths = []
    ep_impacts = []
    ep_twap_comparisons = []
    ep_twap_revenues = []  # ✅ NOUVEAU
    
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
        
        # ✅ NOUVEAU : Variables pour TWAP calculé en parallèle
        twap_inventory = initial_inventory
        twap_total_revenue = 0.0
        
        # ═══════════════════════════════════════════════════════════════
        # BOUCLE PRINCIPALE : Agent ET TWAP dans le MÊME ENVIRONNEMENT
        # ═══════════════════════════════════════════════════════════════
        while not done:
            # Sauvegarder l'état du marché AVANT l'action
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.vol_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.vol_window)
            time_remaining = env.horizon_steps - env.current_step
            
            # ──────────────────────────────────────────────────────────
            # 1. CALCULER L'ACTION TWAP (AVANT que l'agent n'agisse)
            # ──────────────────────────────────────────────────────────
            if time_remaining > 0 and twap_inventory > 1e-6:
                twap_quantity = twap_inventory / time_remaining
                twap_quantity = min(twap_quantity, twap_inventory)
                
                twap_impact = env._calculate_temporary_impact(
                    twap_quantity, realized_vol, rolling_sigma, rolling_volume
                )
                twap_execution_price = current_price * (1 - twap_impact)
                twap_revenue_step = twap_quantity * twap_execution_price
                
                twap_total_revenue += twap_revenue_step
                twap_inventory -= twap_quantity
            
            # ──────────────────────────────────────────────────────────
            # 2. EXÉCUTER L'AGENT (qui modifie l'environnement)
            # ──────────────────────────────────────────────────────────
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
        
        # ═══════════════════════════════════════════════════════════════
        # MÉTRIQUES FINALES DE L'ÉPISODE
        # ═══════════════════════════════════════════════════════════════
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0)
        ep_twap_revenues.append(twap_total_revenue)  # ✅ NOUVEAU
        
        # Performance relative (avec le VRAI TWAP calculé en parallèle)
        if twap_total_revenue > 1e-6:
            relative_performance = ((agent_revenue - twap_total_revenue) / twap_total_revenue) * 100
        else:
            relative_performance = 0.0
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
        'avg_twap_revenue': np.mean(ep_twap_revenues),  # ✅ NOUVEAU (optionnel, pour debug)
        'avg_inventory_trajectory': avg_inventory_trajectory
    }


def print_validation_stats(episode: int, n_episodes: int, metrics: dict):
    """Afficher les statistiques de validation"""
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION @ ÉPISODE {episode}/{n_episodes}")
    print(f"{'='*80}")
    print(f"  🎁 Récompense Agent (moy):    {metrics['avg_reward']:>15.2f}")
    print(f"  🎁 Récompense Agent (med):    {metrics['median_reward']:>15.2f}")
    print(f"  🎯 Perf vs TWAP (moy):        {metrics['avg_twap_comparison']:>15.2f} %")
    print(f"  🎯 Perf vs TWAP (med):        {metrics['median_twap_comparison']:>15.2f} %")
    print(f"  📏 Pas effectifs (moy):       {metrics['avg_length']:>15.1f} pas")
    print(f"  📏 Pas effectifs (med):       {metrics['median_length']:>15.1f} pas")
    print(f"  📏 Pas effectifs (min):       {metrics['min_length']:>15.0f} pas")  # ✅ NOUVEAU
    print(f"  📏 Pas effectifs (max):       {metrics['max_length']:>15.0f} pas")  # ✅ NOUVEAU
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
    n_episodes: int = 1000,
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
    override_epsilon: float = None,
    debug_reward_inconsistency: bool = False
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
        delta=0.1,
        random_start_prob=0.0
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 2. CRÉER L'AGENT
    # ═══════════════════════════════════════════════════════════════
    
    print("Initialisation de l'agent PPO...")
    hidden_dims = [256, 256, 128]  # ✅ DÉFINIR UNE SEULE FOIS
    
    agent = PPOAgent(
        state_dim=env_train.observation_space.shape[0],
        action_dim=env_train.action_space.n,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        lambda_gae=lambda_gae,
        hidden_dims=hidden_dims,  # ✅ Utiliser la variable
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
    validation_avg_impacts = []
    validation_twap_impacts = []
    validation_inventory_remaining = []
    validation_avg_execution_price = []
    validation_twap_revenues_mean = []
    validation_twap_revenues_median = []
    validation_twap_comparisons_mean = []
    validation_twap_comparisons_median = []
    validation_completion_rates = []
    validation_rewards_mean = []
    validation_rewards_median = []
    all_lengths_history = []  # ✅ NOUVEAU : Liste de toutes les listes de longueurs
    
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
    
    if debug_reward_inconsistency:
        print(f"⚠️  MODE DÉBOGAGE ACTIVÉ : Vérification cohérence en validation uniquement")
    
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
                horizon_steps, initial_inventory,
                debug_mode=debug_reward_inconsistency
            )
            
            # Vérifier si incohérence détectée
            if debug_reward_inconsistency and val_metrics.get('inconsistency_detected', False):
                print(f"\n{'⚠️ '*50}")
                print(f"ENTRAÎNEMENT INTERROMPU À L'ÉPISODE {episode + 1}")
                print(f"{'⚠️ '*50}\n")
                return agent, env_val
            
            validation_episodes.append(episode + 1)
            validation_revenues_mean.append(val_metrics['avg_revenue'])
            validation_revenues_median.append(val_metrics['median_revenue'])
            validation_lengths_mean.append(val_metrics['avg_length'])
            validation_lengths_median.append(val_metrics['median_length'])
            validation_avg_impacts.append(val_metrics['avg_impact'])
            validation_twap_impacts.append(val_metrics['avg_twap_impact'])
            validation_inventory_remaining.append(val_metrics['avg_inv_remaining'])
            validation_avg_execution_price.append(val_metrics['avg_price'])
            validation_twap_revenues_mean.append(val_metrics['avg_twap_revenue'])
            validation_twap_revenues_median.append(val_metrics['median_twap_revenue'])
            validation_twap_comparisons_mean.append(val_metrics['avg_twap_comparison'])
            validation_twap_comparisons_median.append(val_metrics['median_twap_comparison'])
            validation_completion_rates.append(val_metrics['completion_rate'])
            validation_rewards_mean.append(val_metrics['avg_reward'])
            validation_rewards_median.append(val_metrics['median_reward'])
            all_lengths_history.append(val_metrics['all_lengths'])  # ✅ NOUVEAU
            
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
            lr=lr, 
            gamma=gamma, 
            epsilon=epsilon, 
            lambda_gae=lambda_gae,
            hidden_dims=hidden_dims,  # ✅ CORRECTION
            device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
        )
        best_mean_agent.load(best_mean_model_path)
        agents_to_test.append(('Meilleur (Moyenne)', best_mean_agent))
    
    # Meilleur (médiane)
    if os.path.exists(best_median_model_path):
        best_median_agent = PPOAgent(
            state_dim=env_val.observation_space.shape[0],
            action_dim=env_val.action_space.n,
            lr=lr, 
            gamma=gamma, 
            epsilon=epsilon, 
            lambda_gae=lambda_gae,
            hidden_dims=hidden_dims,  # ✅ CORRECTION
            device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
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
        validation_avg_impacts,
        validation_twap_impacts,
        validation_inventory_remaining,
        validation_avg_execution_price,
        validation_twap_revenues_mean,
        validation_twap_revenues_median,
        validation_twap_comparisons_mean,
        validation_twap_comparisons_median,
        validation_completion_rates,
        validation_rewards_mean,
        validation_rewards_median,
        initial_inventory,
        all_lengths_history  # ✅ NOUVEAU
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
                           avg_impacts, twap_impacts, inv_remaining, exec_prices, 
                           twap_revenues_mean, twap_revenues_median, 
                           twap_comparisons_mean, twap_comparisons_median, completion_rates,
                           rewards_mean, rewards_median,
                           initial_inventory,
                           all_lengths_history):  # ✅ NOUVEAU PARAMÈTRE
    """Visualiser les résultats de validation"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Résultats de VALIDATION - Agent PPO vs TWAP', 
                 fontsize=16, fontweight='bold')
    
    # 1. Récompenses (Moyenne et Médiane)
    ax = axes[0, 0]
    ax.plot(episodes, rewards_mean, 'o-', label='Moyenne', color='darkviolet', linewidth=2, markersize=4)
    ax.plot(episodes, rewards_median, 's-', label='Médiane', color='purple', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Baseline (TWAP=0)')
    ax.set_title('Récompenses Totales de l\'Agent')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Revenus (Moyenne)
    ax = axes[0, 1]
    ax.plot(episodes, revenues_mean, 'o-', label='Agent PPO (moy)', color='steelblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues_mean, 's-', label='TWAP (moy)', color='orange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux (Moyenne)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 3. Revenus (Médiane)
    ax = axes[0, 2]
    ax.plot(episodes, revenues_median, 'o-', label='Agent PPO (med)', color='darkblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues_median, 's-', label='TWAP (med)', color='darkorange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux (Médiane)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 4. Performance vs TWAP
    ax = axes[1, 0]
    ax.plot(episodes, twap_comparisons_mean, 'o-', label='Moyenne', color='green', linewidth=2, markersize=4)
    ax.plot(episodes, twap_comparisons_median, 's-', label='Médiane', color='darkgreen', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Performance vs TWAP (%)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Amélioration (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Pas effectifs (SANS pas totaux)
    ax = axes[1, 1]
    ax.plot(episodes, lengths_mean, 'o-', label='Pas effectifs (moy)', color='coral', linewidth=2, markersize=4)
    ax.plot(episodes, lengths_median, 's-', label='Pas effectifs (med)', color='darkred', linewidth=2, markersize=4)
    ax.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='TWAP (60 pas)', linewidth=2)
    ax.set_title('Nombre de Pas avec Vente')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Nombre de pas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 65])
    
    # 6. Impact de marché (AVEC TWAP)
    ax = axes[1, 2]
    ax.plot(episodes, avg_impacts, 'o-', label='Agent PPO', color='purple', linewidth=2, markersize=4)
    ax.plot(episodes, twap_impacts, 's-', label='TWAP', color='orange', linewidth=2, markersize=4)
    ax.set_title('Impact de Marché Moyen')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Impact (basis points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. ✅ NOUVEAU : Pas effectifs par épisode + Moyenne mobile
    ax = axes[2, 0]
    
    # Préparer les données : tous les épisodes de validation
    all_validation_episodes = []
    all_validation_lengths = []
    for i, val_ep in enumerate(episodes):
        # Chaque validation contient N épisodes
        n_eps_in_validation = len(all_lengths_history[i])
        # Créer des indices d'épisodes relatifs pour cette validation
        episode_indices = np.arange(n_eps_in_validation) + val_ep - n_eps_in_validation + 1
        all_validation_episodes.extend(episode_indices)
        all_validation_lengths.extend(all_lengths_history[i])
    
    # Calculer la moyenne mobile sur 10 épisodes
    rolling_mean = []
    window_size = 10
    for i in range(len(all_validation_lengths)):
        start_idx = max(0, i - window_size + 1)
        rolling_mean.append(np.mean(all_validation_lengths[start_idx:i+1]))
    
    # Tracer les points individuels (plus petits et transparents)
    ax.scatter(all_validation_episodes, all_validation_lengths, 
               alpha=0.3, s=10, color='steelblue', label='Pas effectifs')
    
    # Tracer la moyenne mobile
    ax.plot(all_validation_episodes, rolling_mean, 
            color='darkred', linewidth=2.5, label='Moyenne mobile (10 épisodes)')
    
    # Ligne de référence TWAP
    ax.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='TWAP (60 pas)', linewidth=2)
    
    ax.set_title('Pas Effectifs par Épisode de Validation')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Nombre de pas')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 65])
    
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
    
    🎁 Récompense (moy):      {rewards_mean[-1]:,.1f}
    🎁 Récompense (med):      {rewards_median[-1]:,.1f}
    
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
        n_episodes=1000,
        horizon_steps=60,
        initial_inventory=1000,
        lr=3e-4,                          
        gamma=0.99,
        epsilon=0.2,                       # Valeur de base
        lambda_gae=0.95,
        update_interval=80,
        validation_interval=100,
        n_validation_episodes=200,
        random_start_prob=0.9,
        save_interval=100,
        
        pretrained_model_path=None#'../models/ppo_execution_best_median.pth',
        #override_epsilon=0.10               
    )
