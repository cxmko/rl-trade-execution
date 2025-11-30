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


def evaluate_targeted_robustness(agent_revenues, twap_revenues, initial_inventory_value):
    """
    CVaR and Win Rate evaluated ONLY on episodes where TWAP <= Initial Price.
    (Excludes 'Easy' Bull Markets).
    """
    agent_revenues = np.array(agent_revenues)
    twap_revenues = np.array(twap_revenues)
    
    # Filter: Keep only 'Bear/Flat' markets
    relevant_mask = twap_revenues <= initial_inventory_value
    
    # Calculate Alpha (Agent - TWAP) on filtered data
    filtered_excess = (agent_revenues - twap_revenues)[relevant_mask]
    
    if len(filtered_excess) == 0:
        return 0.0, 0.0 # No downside scenarios found
        
    # CVaR @ 5%
    sorted_alpha = np.sort(filtered_excess)
    cutoff_idx = int(len(sorted_alpha) * 0.05)
    cvar_5 = sorted_alpha[0] if cutoff_idx == 0 else np.mean(sorted_alpha[:cutoff_idx])
    
    # Win Rate
    win_rate = np.mean(filtered_excess > 0) * 100
    
    return cvar_5, win_rate

def run_validation(agent: PPOAgent, env: OptimalExecutionEnv, n_episodes: int, 
                   horizon_steps: int, initial_inventory: float) -> dict:
    """
    Exécute une boucle de validation propre sans entraînement
    """
    ep_revenues = []
    ep_lengths = []
    ep_impacts = []
    ep_inv_remaining = []
    ep_prices = []
    ep_twap_revenues = []
    ep_twap_impacts = []
    ep_twap_comparisons = []
    ep_rewards = []
    
    # ✅ NEW METRICS LISTS
    ep_vwap_agent = []      # Volume Weighted Average Price (Agent)
    ep_vwap_twap = []       # Volume Weighted Average Price (TWAP)
    ep_impact_std = []      # Standard deviation of impact (Volatility of execution)
    ep_twap_impact_std = [] # Standard deviation of TWAP impact
    ep_timing_bias = []     # Center of mass of execution time (0=Start, 0.5=Middle, 1=End)
    ep_twap_timing_bias = [] # Center of mass for TWAP
    
    for ep_idx in tqdm(range(n_episodes), desc="Validation", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0
        episode_impacts_local = []
        episode_prices_local = []
        episode_volumes_local = [] # Track volume per step for VWAP
        episode_reward_total = 0.0
        
        twap_inventory = initial_inventory
        twap_total_revenue = 0.0
        twap_impacts_local = []
        
        # Track execution times for Center of Mass
        execution_times = []
        execution_quantities = []
        
        # Track TWAP execution for Center of Mass
        twap_execution_times = []
        twap_execution_quantities = []
        
        while not done:
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.vol_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.vol_window)
            time_remaining = env.horizon_steps - env.current_step
            
            # TWAP Logic
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
                
                twap_execution_times.append(step)
                twap_execution_quantities.append(twap_quantity)
            else:
                twap_revenue_step = 0.0
            
            # Agent Logic
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward_total += reward
            
            # ✅ CAPTURE EXECUTION DATA
            qty_sold = info['quantity_sold']
            if qty_sold > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
                episode_prices_local.append(info['execution_price'])
                episode_volumes_local.append(qty_sold)
                
                # For Timing Bias
                execution_times.append(step)
                execution_quantities.append(qty_sold)
            
            state = next_state
            step += 1
        
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)
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
        
        # ✅ CALCULATE ADVANCED METRICS
        
        # 1. VWAP (Volume Weighted Average Price)
        total_vol_agent = sum(episode_volumes_local)
        if total_vol_agent > 0:
            vwap_agent = sum(p * v for p, v in zip(episode_prices_local, episode_volumes_local)) / total_vol_agent
        else:
            vwap_agent = 0
        ep_vwap_agent.append(vwap_agent)
        
        # 2. TWAP VWAP (Approximate)
        if initial_inventory > 0:
            vwap_twap = twap_total_revenue / initial_inventory
        else:
            vwap_twap = 0
        ep_vwap_twap.append(vwap_twap)
        
        # 3. Impact Volatility (Square Root Fallacy Detector)
        if episode_impacts_local:
            ep_impact_std.append(np.std(episode_impacts_local))
        else:
            ep_impact_std.append(0)
            
        if twap_impacts_local:
            ep_twap_impact_std.append(np.std(twap_impacts_local))
        else:
            ep_twap_impact_std.append(0)
            
        # 4. Timing Bias (Center of Mass)
        if sum(execution_quantities) > 0:
            center_of_mass = np.average(execution_times, weights=execution_quantities)
            normalized_bias = center_of_mass / horizon_steps
            ep_timing_bias.append(normalized_bias)
        else:
            ep_timing_bias.append(0.5)
            
        if sum(twap_execution_quantities) > 0:
            twap_center_of_mass = np.average(twap_execution_times, weights=twap_execution_quantities)
            twap_normalized_bias = twap_center_of_mass / horizon_steps
            ep_twap_timing_bias.append(twap_normalized_bias)
        else:
            ep_twap_timing_bias.append(0.5)
    
    # ✅ NEW: Robustness Metrics
    initial_value = initial_inventory * env.initial_price
    robust_cvar, robust_win_rate = evaluate_targeted_robustness(
        ep_revenues, ep_twap_revenues, initial_value
    )
    
    # Calculate final metrics
    avg_revenue = np.mean(ep_revenues)
    median_revenue = np.median(ep_revenues)
    avg_length = np.mean(ep_lengths)
    median_length = np.median(ep_lengths)
    min_length = np.min(ep_lengths)
    max_length = np.max(ep_lengths)
    
    avg_impact = np.mean(ep_impacts)
    median_impact = np.median(ep_impacts) # ✅ Added
    
    avg_inv_remaining = np.mean(ep_inv_remaining)
    avg_price = np.mean(ep_prices)
    
    avg_twap_revenue = np.mean(ep_twap_revenues)
    median_twap_revenue = np.median(ep_twap_revenues)
    
    avg_twap_impact = np.mean(ep_twap_impacts)
    median_twap_impact = np.median(ep_twap_impacts) # ✅ Added
    
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
        'min_length': min_length,
        'max_length': max_length,
        'all_lengths': ep_lengths,
        
        'avg_impact': avg_impact,
        'median_impact': median_impact, # ✅ Added
        
        'avg_twap_impact': avg_twap_impact,
        'median_twap_impact': median_twap_impact, # ✅ Added
        
        'impact_std': np.mean(ep_impact_std),
        'twap_impact_std': np.mean(ep_twap_impact_std), # ✅ Added
        
        'avg_vwap_agent': np.mean(ep_vwap_agent),
        'avg_vwap_twap': np.mean(ep_vwap_twap),
        'vwap_diff_bps': (np.mean(ep_vwap_agent) - np.mean(ep_vwap_twap)) / np.mean(ep_vwap_twap) * 10000,
        
        'timing_bias': np.mean(ep_timing_bias),
        'twap_timing_bias': np.mean(ep_twap_timing_bias), # ✅ Added
        
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
        'robust_cvar': robust_cvar,
        'robust_win_rate': robust_win_rate,
        'inconsistency_detected': False
    }


def run_final_validation(agent: PPOAgent, env: OptimalExecutionEnv, n_episodes: int,
                        horizon_steps: int, initial_inventory: float, agent_name: str) -> dict:
    """
    Validation finale détaillée avec tracking de l'inventaire
    """
    print(f"\n🔍 Validation finale: {agent_name} ({n_episodes} épisodes)")
    
    ep_revenues = []
    ep_lengths = []
    ep_impacts = []
    ep_twap_comparisons = []
    ep_twap_revenues = []
    
    # ✅ NEW: Detailed Metrics Tracking
    ep_vwap_diffs = []
    ep_timing_bias = []
    ep_twap_timing_bias = []
    ep_twap_impacts = []
    
    # Trajectoires d'inventaire (matrice: n_episodes × horizon_steps)
    inventory_trajectories = np.zeros((n_episodes, horizon_steps + 1))
    
    for ep_idx in tqdm(range(n_episodes), desc=f"  {agent_name}", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0
        episode_impacts_local = []
        episode_prices_local = []
        episode_volumes_local = []
        
        # Track execution times for Center of Mass
        execution_times = []
        execution_quantities = []
        
        # Stocker inventaire initial
        inventory_trajectories[ep_idx, 0] = initial_inventory
        
        twap_inventory = initial_inventory
        twap_total_revenue = 0.0
        twap_impacts_local = []
        twap_execution_times = []
        twap_execution_quantities = []
        
        while not done:
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))
            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.vol_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.vol_window)
            time_remaining = env.horizon_steps - env.current_step
            
            # 1. CALCULER L'ACTION TWAP
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
                
                # ✅ FIX: Track TWAP metrics inside the loop
                twap_impacts_local.append(twap_impact * 10000)
                twap_execution_times.append(step)
                twap_execution_quantities.append(twap_quantity)
            
            # 2. EXÉCUTER L'AGENT
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info['quantity_sold'] > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
                episode_prices_local.append(info['execution_price'])
                episode_volumes_local.append(info['quantity_sold'])
                
                execution_times.append(step)
                execution_quantities.append(info['quantity_sold'])
            
            inventory_trajectories[ep_idx, step + 1] = info['inventory_remaining']
            
            state = next_state
            step += 1
        
        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0)
        ep_twap_revenues.append(twap_total_revenue)
        ep_twap_impacts.append(np.mean(twap_impacts_local) if twap_impacts_local else 0)
        
        if twap_total_revenue > 1e-6:
            relative_performance = ((agent_revenue - twap_total_revenue) / twap_total_revenue) * 100
        else:
            relative_performance = 0.0
        ep_twap_comparisons.append(relative_performance)
        
        # ✅ FIX: Calculate Per-Episode Metrics INSIDE the loop
        
        # VWAP Diff Calculation
        total_vol_agent = sum(episode_volumes_local)
        if total_vol_agent > 0:
            vwap_agent = sum(p * v for p, v in zip(episode_prices_local, episode_volumes_local)) / total_vol_agent
        else:
            vwap_agent = 0
        
        if initial_inventory > 0:
            vwap_twap = twap_total_revenue / initial_inventory
        else:
            vwap_twap = 0
        
        if vwap_twap > 0:
            ep_vwap_diffs.append((vwap_agent - vwap_twap) / vwap_twap * 10000)
        else:
            ep_vwap_diffs.append(0.0)
        
        # Timing Bias Calculation
        if sum(execution_quantities) > 0:
            center_of_mass = np.average(execution_times, weights=execution_quantities)
            ep_timing_bias.append(center_of_mass / horizon_steps)
        else:
            ep_timing_bias.append(0.5)
        
        if sum(twap_execution_quantities) > 0:
            twap_center_of_mass = np.average(twap_execution_times, weights=twap_execution_quantities)
            ep_twap_timing_bias.append(twap_center_of_mass / horizon_steps)
        else:
            ep_twap_timing_bias.append(0.5)
    
    avg_inventory_trajectory = np.mean(inventory_trajectories, axis=0)
    std_inventory_trajectory = np.std(inventory_trajectories, axis=0) if 'inventory_trajectory' in locals() else np.std(inventory_trajectories, axis=0)
    min_inventory_trajectory = np.min(inventory_trajectories, axis=0)
    max_inventory_trajectory = np.max(inventory_trajectories, axis=0)
    
    # ✅ NEW: Calculate Robustness for Final Validation
    initial_value = initial_inventory * env.initial_price
    robust_cvar, robust_win_rate = evaluate_targeted_robustness(
        ep_revenues, ep_twap_revenues, initial_value
    )
    
    return {
        'agent_name': agent_name,
        'avg_revenue': np.mean(ep_revenues),
        'median_revenue': np.median(ep_revenues),
        'avg_twap_revenue': np.mean(ep_twap_revenues),
        'median_twap_revenue': np.median(ep_twap_revenues),
        
        'avg_length': np.mean(ep_lengths),
        'median_length': np.median(ep_lengths),
        'min_length': np.min(ep_lengths),
        'max_length': np.max(ep_lengths),
        
        'avg_impact': np.mean(ep_impacts),
        'median_impact': np.median(ep_impacts),
        'std_impact': np.std(ep_impacts),
        
        'avg_twap_impact': np.mean(ep_twap_impacts),
        'median_twap_impact': np.median(ep_twap_impacts),
        'std_twap_impact': np.std(ep_twap_impacts),
        
        'avg_twap_comparison': np.mean(ep_twap_comparisons),
        'median_twap_comparison': np.median(ep_twap_comparisons),
        
        'avg_vwap_diff': np.mean(ep_vwap_diffs),
        'avg_timing_bias': np.mean(ep_timing_bias),
        'avg_twap_timing_bias': np.mean(ep_twap_timing_bias),
        
        'avg_inventory_trajectory': avg_inventory_trajectory,
        'std_inventory_trajectory': std_inventory_trajectory,
        'min_inventory_trajectory': min_inventory_trajectory,
        'max_inventory_trajectory': max_inventory_trajectory,
        'robust_cvar': robust_cvar,
        'robust_win_rate': robust_win_rate
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
    print(f"{'-'*80}")
    print(f"  📉 ANALYSE DÉTAILLÉE (Agent vs TWAP)")
    print(f"  💥 Impact Moyen:              {metrics['avg_impact']:>10.2f} bps  vs {metrics['avg_twap_impact']:>10.2f} bps (TWAP)")
    print(f"  💥 Impact Médian:             {metrics['median_impact']:>10.2f} bps  vs {metrics['median_twap_impact']:>10.2f} bps (TWAP)")
    print(f"  📊 Impact Std Dev:            {metrics['impact_std']:>10.2f} bps  vs {metrics['twap_impact_std']:>10.2f} bps (TWAP)")
    print(f"  ⚖️  VWAP Différence:          {metrics['vwap_diff_bps']:>15.2f} bps (Negative = Sold Cheaper)")
    print(f"  ⏳ Timing Bias (0-1):         {metrics['timing_bias']:>10.2f}       vs {metrics['twap_timing_bias']:>10.2f}       (TWAP)")
    print(f"{'-'*80}")
    print(f"  🛡️ ROBUSTNESS (Bear Mkts):    CVaR {metrics['robust_cvar']:.2f} USDT | WinRate {metrics['robust_win_rate']:.1f}%")
    print(f"{'-'*80}")
    print(f"  📏 Pas effectifs (moy):       {metrics['avg_length']:>15.1f} pas")
    print(f"  📏 Pas effectifs (med):       {metrics['median_length']:>15.1f} pas")
    print(f"  📏 Pas effectifs (min):       {metrics['min_length']:>15.0f} pas")
    print(f"  📏 Pas effectifs (max):       {metrics['max_length']:>15.0f} pas")
    print(f"  📈 Taux de complétion:        {metrics['completion_rate']:>15.1f} %")
    print(f"{'='*80}")


def print_final_validation_stats(results: list):
    """Afficher les statistiques de la validation finale"""
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION FINALE - COMPARAISON DES MODÈLES")
    print(f"{'='*80}\n")
    
    for res in results:
        print(f"🤖 {res['agent_name']}")
        print(f"  {'─'*76}")
        print(f"  💰 Revenu Agent (moy):   {res['avg_revenue']:>15,.2f} USDT")
        print(f"  💰 Revenu TWAP (moy):    {res['avg_twap_revenue']:>15,.2f} USDT")
        print(f"  🎯 Perf vs TWAP (moy):   {res['avg_twap_comparison']:>15.2f} %")
        print(f"  🎯 Perf vs TWAP (med):   {res['median_twap_comparison']:>15.2f} %")
        print(f"  🛡️ CVaR (Bear Mkts):     {res['robust_cvar']:>15.2f} USDT")
        print(f"  🏆 Win Rate (Bear):      {res['robust_win_rate']:>15.1f} %")
        print(f"  ⚖️ VWAP Différence:      {res['avg_vwap_diff']:>15.2f} bps")
        print(f"  ⏳ Timing Bias:          {res['avg_timing_bias']:>15.2f} vs {res['avg_twap_timing_bias']:.2f} (TWAP)")
        print(f"  📏 Pas effectifs (moy):  {res['avg_length']:>15.1f}")
        print(f"  📏 Pas effectifs (min/max): {res['min_length']:.0f} / {res['max_length']:.0f}")
        print(f"  💥 Impact moyen Agent :         {res['avg_impact']:>15.2f} bps")
        print(f"  💥 Impact moyen TWAP :       {res['avg_twap_impact']:>15.2f} bps")
        print(f"  💥 Impact Agent (med/std):  {res['median_impact']:.2f} ± {res['std_impact']:.2f} bps")
        print(f"  💥 Impact TWAP (med/std):   {res['median_twap_impact']:.2f} ± {res['std_twap_impact']:.2f} bps")
   
        print()
    
    print(f"{'='*80}\n")


def train_ppo(
    data_path: str,
    n_episodes: int = 1000,
    horizon_steps: int = 240,
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
        lambda_0=0.003,
        alpha=0.5,
        delta=0,
        random_start_prob=random_start_prob
    )
    
    print("Initialisation de l'environnement de VALIDATION...")
    env_val = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.003,
        alpha=0.5,
        delta=0,
        random_start_prob=0.0
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 2. CRÉER L'AGENT
    # ═══════════════════════════════════════════════════════════════
    
    print("Initialisation de l'agent PPO...")
    hidden_dims = [256, 128, 64]  
    
    agent = PPOAgent(
        state_dim=env_train.observation_space.shape[0],
        action_dim=env_train.action_space.n,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        lambda_gae=lambda_gae,
        hidden_dims=hidden_dims,
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
    
    # ✅ NEW: Tracking Robustness Metrics History
    validation_cvar = []
    validation_win_rate = []
    validation_vwap_diff = []
    
    all_lengths_history = []
    
    best_mean_performance = -np.inf
    best_median_performance = -np.inf
    best_cvar_performance = -np.inf
    best_win_rate_performance = -np.inf
    
    best_mean_model_path = model_save_path.replace('.pth', '_best_mean.pth')
    best_median_model_path = model_save_path.replace('.pth', '_best_median.pth')
    best_cvar_model_path = model_save_path.replace('.pth', '_best_cvar.pth')
    best_win_rate_model_path = model_save_path.replace('.pth', '_best_win_rate.pth')
    
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
            
            # ✅ NEW: Append Robustness Metrics
            validation_cvar.append(val_metrics['robust_cvar'])
            validation_win_rate.append(val_metrics['robust_win_rate'])
            validation_vwap_diff.append(val_metrics['vwap_diff_bps'])
            
            all_lengths_history.append(val_metrics['all_lengths'])
            
            print_validation_stats(episode + 1, n_episodes, val_metrics)
            
            # Save Best Models
            if val_metrics['avg_twap_comparison'] > best_mean_performance:
                best_mean_performance = val_metrics['avg_twap_comparison']
                agent.save(best_mean_model_path)
                print(f"💎 Nouveau meilleur modèle (MOYENNE) ! Performance: {best_mean_performance:.2f}% vs TWAP")
            
            if val_metrics['median_twap_comparison'] > best_median_performance:
                best_median_performance = val_metrics['median_twap_comparison']
                agent.save(best_median_model_path)
                print(f"💎 Nouveau meilleur modèle (MÉDIANE) ! Performance: {best_median_performance:.2f}% vs TWAP")
                
            # ✅ NEW: Save Best Robust Models
            if val_metrics['robust_cvar'] > best_cvar_performance:
                best_cvar_performance = val_metrics['robust_cvar']
                agent.save(best_cvar_model_path)
                print(f"🛡️ Nouveau meilleur modèle (CVaR) ! CVaR: {best_cvar_performance:.2f} USDT")
                
            if val_metrics['robust_win_rate'] > best_win_rate_performance:
                best_win_rate_performance = val_metrics['robust_win_rate']
                agent.save(best_win_rate_model_path)
                print(f"🏆 Nouveau meilleur modèle (WIN RATE) ! Win Rate: {best_win_rate_performance:.1f}%")
            
            main_pbar.set_postfix({
                'Val_Mean': f"{val_metrics['avg_twap_comparison']:.2f}%",
                'CVaR': f"{val_metrics['robust_cvar']:.0f}",
                'WinRate': f"{val_metrics['robust_win_rate']:.0f}%"
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
    print(f"🛡️ Meilleur (CVaR):    {best_cvar_model_path} ({best_cvar_performance:.2f})")
    print(f"🏆 Meilleur (WinRate): {best_win_rate_model_path} ({best_win_rate_performance:.1f}%)")
    print(f"{'='*80}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. VALIDATION FINALE (100 épisodes) + ÉVOLUTION INVENTAIRE
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*80}")
    print(f"🔬 VALIDATION FINALE - 100 ÉPISODES")
    print(f"{'='*80}\n")
    
    final_results = []
    
    # Charger et valider les 4 modèles (Mean, Median, CVaR, WinRate)
    agents_to_test = []
    
    # Modèle final
    agents_to_test.append(('Modèle Final', agent))
    
    # Meilleur (moyenne)
    if os.path.exists(best_mean_model_path):
        best_mean_agent = PPOAgent(state_dim=env_val.observation_space.shape[0], action_dim=env_val.action_space.n, lr=lr, gamma=gamma, epsilon=epsilon, lambda_gae=lambda_gae, hidden_dims=hidden_dims, device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu')
        best_mean_agent.load(best_mean_model_path)
        agents_to_test.append(('Meilleur (Moyenne)', best_mean_agent))
    
    # Meilleur (médiane)
    if os.path.exists(best_median_model_path):
        best_median_agent = PPOAgent(state_dim=env_val.observation_space.shape[0], action_dim=env_val.action_space.n, lr=lr, gamma=gamma, epsilon=epsilon, lambda_gae=lambda_gae, hidden_dims=hidden_dims, device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu')
        best_median_agent.load(best_median_model_path)
        agents_to_test.append(('Meilleur (Médiane)', best_median_agent))
        
    # ✅ NEW: Load Robust Models
    if os.path.exists(best_cvar_model_path):
        best_cvar_agent = PPOAgent(state_dim=env_val.observation_space.shape[0], action_dim=env_val.action_space.n, lr=lr, gamma=gamma, epsilon=epsilon, lambda_gae=lambda_gae, hidden_dims=hidden_dims, device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu')
        best_cvar_agent.load(best_cvar_model_path)
        agents_to_test.append(('Meilleur (CVaR)', best_cvar_agent))
        
    if os.path.exists(best_win_rate_model_path):
        best_win_rate_agent = PPOAgent(state_dim=env_val.observation_space.shape[0], action_dim=env_val.action_space.n, lr=lr, gamma=gamma, epsilon=epsilon, lambda_gae=lambda_gae, hidden_dims=hidden_dims, device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu')
        best_win_rate_agent.load(best_win_rate_model_path)
        agents_to_test.append(('Meilleur (WinRate)', best_win_rate_agent))
    
    # Valider chaque agent
    for agent_name, test_agent in agents_to_test:
        result = run_final_validation(
            test_agent, env_val, 1000,
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
        # ✅ NEW ARGS
        validation_cvar,
        validation_win_rate,
        validation_vwap_diff,
        
        initial_inventory,
        all_lengths_history,
        horizon_steps=horizon_steps
    )
    
    plot_final_inventory_evolution(final_results, horizon_steps, initial_inventory)
    
    return agent, env_val


def plot_final_inventory_evolution(results: list, horizon_steps: int, initial_inventory: float):
    """Visualiser l'évolution moyenne de l'inventaire pour chaque modèle avec bandes de variance"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_steps = np.arange(0, horizon_steps + 1)
    colors = ['steelblue', 'darkgreen', 'coral']
    
    for idx, res in enumerate(results):
        avg_traj = res['avg_inventory_trajectory']
        std_traj = res.get('std_inventory_trajectory', np.zeros_like(avg_traj))
        
        color = colors[idx % len(colors)]
        
        # Tracer la moyenne
        ax.plot(time_steps, avg_traj, 
                label=f"{res['agent_name']} (Moyenne)", color=color,
                linewidth=2.5, marker='o', markersize=4, markevery=5)
        
        # Tracer la bande d'écart-type (Variance)
        ax.fill_between(time_steps, 
                        np.maximum(0, avg_traj - std_traj), 
                        np.minimum(initial_inventory, avg_traj + std_traj),
                        color=color, alpha=0.2, label=f"{res['agent_name']} (±1 std)")
    
    # TWAP de référence (linéaire)
    twap_trajectory = np.linspace(initial_inventory, 0, horizon_steps + 1)
    ax.plot(time_steps, twap_trajectory, '--', label='TWAP (référence)',
            color='gray', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Pas de temps', fontsize=13)
    ax.set_ylabel('Inventaire (BTC)', fontsize=13)
    ax.set_title('Évolution de l\'Inventaire (Moyenne ± Écart-type)\n(100 épisodes de validation)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
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
                           # ✅ NEW ARGS
                           cvar_history, win_rate_history, vwap_diff_history,
                           
                           initial_inventory,
                           all_lengths_history,
                           horizon_steps=240):
    """Visualiser les résultats de validation"""
    
    # ✅ CHANGED: Increased to 4 rows to fit new metrics
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Résultats de VALIDATION - Agent PPO vs TWAP', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Rewards, Revenues Mean, Revenues Median
    ax = axes[0, 0]
    ax.plot(episodes, rewards_mean, 'o-', label='Moyenne', color='darkviolet')
    ax.plot(episodes, rewards_median, 's-', label='Médiane', color='purple')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Récompenses Totales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(episodes, revenues_mean, 'o-', label='Agent', color='steelblue')
    ax.plot(episodes, twap_revenues_mean, 's-', label='TWAP', color='orange')
    ax.set_title('Revenus (Moyenne)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(episodes, revenues_median, 'o-', label='Agent', color='darkblue')
    ax.plot(episodes, twap_revenues_median, 's-', label='TWAP', color='darkorange')
    ax.set_title('Revenus (Médiane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: Perf vs TWAP, Lengths, Impacts
    ax = axes[1, 0]
    ax.plot(episodes, twap_comparisons_mean, 'o-', label='Moyenne', color='green')
    ax.plot(episodes, twap_comparisons_median, 's-', label='Médiane', color='darkgreen')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Performance vs TWAP (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(episodes, lengths_mean, 'o-', label='Moyenne', color='coral')
    ax.axhline(y=horizon_steps, color='blue', linestyle='--')
    ax.set_title('Pas Effectifs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(episodes, avg_impacts, 'o-', label='Agent', color='purple')
    ax.plot(episodes, twap_impacts, 's-', label='TWAP', color='orange')
    ax.set_title('Impact de Marché (bps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 3: Length Distribution, Perf Distribution, VWAP Diff (Moved here)
    ax = axes[2, 0]
    # ... [Length scatter plot code] ...
    all_validation_episodes = []
    all_validation_lengths = []
    for i, val_ep in enumerate(episodes):
        n_eps_in_validation = len(all_lengths_history[i])
        episode_indices = np.arange(n_eps_in_validation) + val_ep - n_eps_in_validation + 1
        all_validation_episodes.extend(episode_indices)
        all_validation_lengths.extend(all_lengths_history[i])
    ax.scatter(all_validation_episodes, all_validation_lengths, alpha=0.3, s=10, color='steelblue')
    ax.axhline(y=horizon_steps, color='blue', linestyle='--')
    ax.set_title('Distribution des Pas')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    # ... [Bar chart code] ...
    if len(episodes) > 1:
        bar_width = (episodes[1] - episodes[0]) * 0.4
    else:
        bar_width = 40
    ax.bar([e - bar_width/2 for e in episodes], twap_comparisons_mean, width=bar_width, color='green', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_title('Distribution Performance')
    ax.grid(True, alpha=0.3)
    
    # ✅ NEW PLOT: VWAP Difference
    ax = axes[2, 2]
    ax.plot(episodes, vwap_diff_history, 'o-', color='teal', label='VWAP Diff (bps)')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('VWAP Différence (bps)')
    ax.set_ylabel('Agent - TWAP (bps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ✅ ROW 4: ROBUSTNESS METRICS
    
    # CVaR
    ax = axes[3, 0]
    ax.plot(episodes, cvar_history, 'o-', color='crimson', label='CVaR (5%)')
    ax.set_title('Robustesse: CVaR (Bear Markets)')
    ax.set_ylabel('Perte vs TWAP (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Win Rate
    ax = axes[3, 1]
    ax.plot(episodes, win_rate_history, 'o-', color='forestgreen', label='Win Rate')
    ax.axhline(y=50, color='gray', linestyle='--')
    ax.set_title('Robustesse: Win Rate (Bear Markets)')
    ax.set_ylabel('% Victoires')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final Stats Text
    ax = axes[3, 2]
    ax.axis('off')
    stats_text = f"""
    📊 STATISTIQUES FINALES
    {'─'*45}
    🎁 Récompense (moy):      {rewards_mean[-1]:,.1f}
    🎯 Perf vs TWAP (moy):    {twap_comparisons_mean[-1]:+.2f} %
    🎯 Perf vs TWAP (med):    {twap_comparisons_median[-1]:+.2f} %
    
    🛡️ CVaR (Bear):           {cvar_history[-1]:.2f} USDT
    🏆 Win Rate (Bear):       {win_rate_history[-1]:.1f} %
    ⚖️ VWAP Diff:             {vwap_diff_history[-1]:.2f} bps
    
    📏 Pas effectifs (moy):   {lengths_mean[-1]:.1f}
    📏 Pas effectifs (med):   {lengths_median[-1]:.1f}
    💥 Impact Agent:          {avg_impacts[-1]:.2f} bps
    💥 Impact TWAP:           {twap_impacts[-1]:.2f} bps
    {'─'*45}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('../sample/validation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphiques de validation sauvegardés: ../sample/validation_results.png")
    plt.show()


if __name__ == "__main__":
    data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    
    agent, env = train_ppo(
        data_path=data_path,
        n_episodes=5000,
        horizon_steps=240,
        initial_inventory=1000,
        
        # ✅ CHANGED: Hyperparameters for Robust Training
        lr=1e-4/2,                            # Slower, more stable learning
        update_interval=20,                 # Faster feedback (approx every 4800 steps)
        
        gamma=1.0,
        epsilon=0.3,
        lambda_gae=0.95,
        validation_interval=100,
        n_validation_episodes=200,
        random_start_prob=0.9,
        save_interval=100,
        
        pretrained_model_path='../models/ppo_execution_best_median_nn.pth',
        override_epsilon=0.2        
    )
