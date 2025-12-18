import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='The optimizer returned code 4')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.environment.execution_envdqn import OptimalExecutionEnv
from src.models.dqn_agent import DQNAgent


def evaluate_targeted_robustness(agent_revenues, twap_revenues, initial_values):
    """
    CVaR + Win Rate évalués UNIQUEMENT sur les épisodes où TWAP <= valeur initiale (arrivée).
    initial_values peut être un scalaire ou un array (1 par épisode).
    """
    agent_revenues = np.asarray(agent_revenues, dtype=np.float64)
    twap_revenues = np.asarray(twap_revenues, dtype=np.float64)
    initial_values = np.asarray(initial_values, dtype=np.float64)

    if initial_values.size == 1:
        relevant_mask = twap_revenues <= float(initial_values)
    else:
        relevant_mask = twap_revenues <= initial_values

    filtered_excess = (agent_revenues - twap_revenues)[relevant_mask]

    if filtered_excess.size == 0:
        return 0.0, 0.0

    sorted_alpha = np.sort(filtered_excess)
    cutoff_idx = int(sorted_alpha.size * 0.05)
    cvar_5 = sorted_alpha[0] if cutoff_idx == 0 else float(np.mean(sorted_alpha[:cutoff_idx]))

    win_rate = float(np.mean(filtered_excess > 0) * 100.0)
    return cvar_5, win_rate


def run_validation(agent: DQNAgent, env: OptimalExecutionEnv, n_episodes: int,
                   horizon_steps: int, initial_inventory: float, debug_mode: bool = False) -> dict:
    """
    Validation greedy (sans exploration), TWAP en parallèle.
    """
    ep_revenues, ep_lengths, ep_impacts, ep_inv_remaining, ep_prices = [], [], [], [], []
    ep_twap_revenues, ep_twap_impacts, ep_twap_comparisons, ep_rewards = [], [], [], []

    ep_vwap_agent, ep_vwap_twap = [], []
    ep_impact_std, ep_twap_impact_std = [], []
    ep_timing_bias, ep_twap_timing_bias = [], []

    ep_initial_values = []  # ✅ NEW

    for ep_idx in tqdm(range(n_episodes), desc="Validation", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0

        episode_impacts_local, episode_prices_local, episode_volumes_local = [], [], []
        episode_reward_total = 0.0

        # TWAP parallèle
        twap_inventory = initial_inventory
        twap_total_revenue = 0.0
        twap_impacts_local = []

        execution_times, execution_quantities = [], []
        twap_execution_times, twap_execution_quantities = [], []


        ep_initial_values.append(initial_inventory * env.initial_price)

        while not done:
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))


            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.avg_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.avg_window)

            time_remaining = env.horizon_steps - env.current_step

            # TWAP
            if time_remaining > 0 and twap_inventory > 1e-6:
                twap_quantity = min(twap_inventory / time_remaining, twap_inventory)

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

            # Agent greedy
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward_total += reward

            qty_sold = info['quantity_sold']
            if qty_sold > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(info['temp_impact_relative'] * 10000)
                episode_prices_local.append(info['execution_price'])
                episode_volumes_local.append(qty_sold)

                execution_times.append(step)
                execution_quantities.append(qty_sold)

            state = next_state
            step += 1

        agent_revenue = info['total_revenue']
        ep_revenues.append(agent_revenue)
        ep_lengths.append(effective_steps)
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0.0)
        ep_inv_remaining.append(info['inventory_remaining'])
        ep_prices.append(np.mean(episode_prices_local) if episode_prices_local else env.initial_price)
        ep_rewards.append(episode_reward_total)

        ep_twap_revenues.append(twap_total_revenue)
        ep_twap_impacts.append(np.mean(twap_impacts_local) if twap_impacts_local else 0.0)

        if twap_total_revenue > 1e-6:
            relative_performance = ((agent_revenue - twap_total_revenue) / twap_total_revenue) * 100.0
        else:
            relative_performance = 0.0
        ep_twap_comparisons.append(relative_performance)

        # VWAP agent
        total_vol_agent = float(np.sum(episode_volumes_local))
        if total_vol_agent > 0:
            vwap_agent = float(np.sum(np.array(episode_prices_local) * np.array(episode_volumes_local)) / total_vol_agent)
        else:
            vwap_agent = 0.0
        ep_vwap_agent.append(vwap_agent)

        # VWAP TWAP
        vwap_twap = float(twap_total_revenue / initial_inventory) if initial_inventory > 0 else 0.0
        ep_vwap_twap.append(vwap_twap)

        # Volatilité impact
        ep_impact_std.append(float(np.std(episode_impacts_local)) if episode_impacts_local else 0.0)
        ep_twap_impact_std.append(float(np.std(twap_impacts_local)) if twap_impacts_local else 0.0)

        # Timing bias
        if np.sum(execution_quantities) > 0:
            center_of_mass = float(np.average(execution_times, weights=execution_quantities))
            ep_timing_bias.append(center_of_mass / horizon_steps)
        else:
            ep_timing_bias.append(0.5)

        if np.sum(twap_execution_quantities) > 0:
            twap_center_of_mass = float(np.average(twap_execution_times, weights=twap_execution_quantities))
            ep_twap_timing_bias.append(twap_center_of_mass / horizon_steps)
        else:
            ep_twap_timing_bias.append(0.5)

        if debug_mode and ep_idx < 3:
            print(f"    [DEBUG] Ep {ep_idx+1}: Revenue={agent_revenue:.2f}, "
                  f"TWAP={twap_total_revenue:.2f}, Perf={relative_performance:+.2f}%, Steps={effective_steps}")

    robust_cvar, robust_win_rate = evaluate_targeted_robustness(
        ep_revenues, ep_twap_revenues, ep_initial_values
    )

    completion_rate = (1 - np.mean(ep_inv_remaining) / initial_inventory) * 100.0

    avg_vwap_agent = float(np.mean(ep_vwap_agent))
    avg_vwap_twap = float(np.mean(ep_vwap_twap))
    vwap_diff_bps = ((avg_vwap_agent - avg_vwap_twap) / avg_vwap_twap * 10000.0) if avg_vwap_twap > 0 else 0.0

    return {
        'avg_revenue': float(np.mean(ep_revenues)),
        'median_revenue': float(np.median(ep_revenues)),
        'std_revenue': float(np.std(ep_revenues)),
        'avg_length': float(np.mean(ep_lengths)),
        'median_length': float(np.median(ep_lengths)),
        'min_length': float(np.min(ep_lengths)),
        'max_length': float(np.max(ep_lengths)),
        'all_lengths': ep_lengths,

        'avg_impact': float(np.mean(ep_impacts)),
        'median_impact': float(np.median(ep_impacts)),

        'avg_twap_impact': float(np.mean(ep_twap_impacts)),
        'median_twap_impact': float(np.median(ep_twap_impacts)),

        'impact_std': float(np.mean(ep_impact_std)),
        'twap_impact_std': float(np.mean(ep_twap_impact_std)),

        'avg_vwap_agent': avg_vwap_agent,
        'avg_vwap_twap': avg_vwap_twap,
        'vwap_diff_bps': float(vwap_diff_bps),

        'timing_bias': float(np.mean(ep_timing_bias)),
        'twap_timing_bias': float(np.mean(ep_twap_timing_bias)),

        'avg_inv_remaining': float(np.mean(ep_inv_remaining)),
        'avg_price': float(np.mean(ep_prices)),
        'completion_rate': float(completion_rate),

        'avg_twap_revenue': float(np.mean(ep_twap_revenues)),
        'median_twap_revenue': float(np.median(ep_twap_revenues)),
        'avg_twap_comparison': float(np.mean(ep_twap_comparisons)),
        'median_twap_comparison': float(np.median(ep_twap_comparisons)),

        'avg_reward': float(np.mean(ep_rewards)),
        'median_reward': float(np.median(ep_rewards)),

        'robust_cvar': float(robust_cvar),
        'robust_win_rate': float(robust_win_rate)
    }


def run_final_validation(agent: DQNAgent, env: OptimalExecutionEnv, n_episodes: int,
                        horizon_steps: int, initial_inventory: float, agent_name: str) -> dict:
    """
    Validation finale détaillée.
    """
    print(f"\n🔍 Validation finale: {agent_name} ({n_episodes} épisodes)")

    ep_revenues, ep_lengths, ep_impacts = [], [], []
    ep_twap_comparisons, ep_twap_revenues, ep_twap_impacts = [], [], []
    all_actions = []

    ep_vwap_diffs, ep_timing_bias, ep_twap_timing_bias = [], [], []
    ep_initial_values = []

    inventory_trajectories = np.zeros((n_episodes, horizon_steps + 1), dtype=np.float64)

    for ep_idx in tqdm(range(n_episodes), desc=f"  {agent_name}", leave=False):
        state, _ = env.reset()
        done = False
        step = 0
        effective_steps = 0

        episode_impacts_local, episode_prices_local, episode_volumes_local = [], [], []
        episode_actions = []

        execution_times, execution_quantities = [], []
        twap_execution_times, twap_execution_quantities = [], []

        inventory_trajectories[ep_idx, 0] = initial_inventory

        twap_inventory = initial_inventory
        twap_total_revenue = 0.0
        twap_impacts_local = []

        ep_initial_values.append(initial_inventory * env.initial_price)

        while not done:
            current_price = env.prices_history[-1]
            realized_vol = env._calculate_realized_volatility(np.array(env.prices_history))

            rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.avg_window)
            rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.avg_window)
            time_remaining = env.horizon_steps - env.current_step

            # TWAP
            if time_remaining > 0 and twap_inventory > 1e-6:
                twap_quantity = min(twap_inventory / time_remaining, twap_inventory)

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

            # Agent greedy)
            old_eps = agent.epsilon
            agent.epsilon = 0.02
            action, _, _ = agent.select_action(state, deterministic=False)
            agent.epsilon = old_eps
            episode_actions.append(action)

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
        ep_impacts.append(np.mean(episode_impacts_local) if episode_impacts_local else 0.0)

        ep_twap_revenues.append(twap_total_revenue)
        ep_twap_impacts.append(np.mean(twap_impacts_local) if twap_impacts_local else 0.0)

        all_actions.extend(episode_actions)

        relative_performance = ((agent_revenue - twap_total_revenue) / twap_total_revenue) * 100.0 if twap_total_revenue > 1e-6 else 0.0
        ep_twap_comparisons.append(relative_performance)

        # VWAP diff bps
        total_vol_agent = float(np.sum(episode_volumes_local))
        vwap_agent = float(np.sum(np.array(episode_prices_local) * np.array(episode_volumes_local)) / total_vol_agent) if total_vol_agent > 0 else 0.0
        vwap_twap = float(twap_total_revenue / initial_inventory) if initial_inventory > 0 else 0.0
        ep_vwap_diffs.append(((vwap_agent - vwap_twap) / vwap_twap * 10000.0) if vwap_twap > 0 else 0.0)

        # Timing bias
        if np.sum(execution_quantities) > 0:
            center_of_mass = float(np.average(execution_times, weights=execution_quantities))
            ep_timing_bias.append(center_of_mass / horizon_steps)
        else:
            ep_timing_bias.append(0.5)

        if np.sum(twap_execution_quantities) > 0:
            twap_center_of_mass = float(np.average(twap_execution_times, weights=twap_execution_quantities))
            ep_twap_timing_bias.append(twap_center_of_mass / horizon_steps)
        else:
            ep_twap_timing_bias.append(0.5)

    action_counts = np.bincount(all_actions, minlength=env.action_space.n)
    action_percentages = action_counts / max(1, len(all_actions)) * 100.0

    avg_inventory_trajectory = np.mean(inventory_trajectories, axis=0)
    std_inventory_trajectory = np.std(inventory_trajectories, axis=0)
    min_inventory_trajectory = np.min(inventory_trajectories, axis=0)
    max_inventory_trajectory = np.max(inventory_trajectories, axis=0)

    robust_cvar, robust_win_rate = evaluate_targeted_robustness(
        ep_revenues, ep_twap_revenues, ep_initial_values
    )

    return {
        'agent_name': agent_name,
        'avg_revenue': float(np.mean(ep_revenues)),
        'median_revenue': float(np.median(ep_revenues)),
        'std_revenue': float(np.std(ep_revenues)),
        'avg_twap_revenue': float(np.mean(ep_twap_revenues)),
        'median_twap_revenue': float(np.median(ep_twap_revenues)),

        'avg_length': float(np.mean(ep_lengths)),
        'median_length': float(np.median(ep_lengths)),
        'min_length': float(np.min(ep_lengths)),
        'max_length': float(np.max(ep_lengths)),

        'avg_impact': float(np.mean(ep_impacts)),
        'median_impact': float(np.median(ep_impacts)),
        'std_impact': float(np.std(ep_impacts)),

        'avg_twap_impact': float(np.mean(ep_twap_impacts)),
        'median_twap_impact': float(np.median(ep_twap_impacts)),
        'std_twap_impact': float(np.std(ep_twap_impacts)),

        'avg_twap_comparison': float(np.mean(ep_twap_comparisons)),
        'median_twap_comparison': float(np.median(ep_twap_comparisons)),

        'avg_vwap_diff': float(np.mean(ep_vwap_diffs)),
        'avg_timing_bias': float(np.mean(ep_timing_bias)),
        'avg_twap_timing_bias': float(np.mean(ep_twap_timing_bias)),

        'action_distribution': action_percentages,

        'avg_inventory_trajectory': avg_inventory_trajectory,
        'std_inventory_trajectory': std_inventory_trajectory,
        'min_inventory_trajectory': min_inventory_trajectory,
        'max_inventory_trajectory': max_inventory_trajectory,

        'robust_cvar': float(robust_cvar),
        'robust_win_rate': float(robust_win_rate)
    }






def print_validation_stats(episode: int, n_episodes: int, metrics: dict, epsilon: float):
    """Affiche les statistiques de validation (format aligné avec PPO)"""
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
    print(f"  🎲 Epsilon:                   {epsilon:>15.4f}")
    print(f"{'='*80}")





def print_final_validation_stats(results: list):
    """Afficher les statistiques de la validation finale (aligné avec PPO)"""
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
        print(f"  💥 Impact moyen Agent:   {res['avg_impact']:>15.2f} bps")
        print(f"  💥 Impact moyen TWAP:    {res['avg_twap_impact']:>15.2f} bps")
        print(f"  💥 Impact Agent (med/std):  {res['median_impact']:.2f} ± {res['std_impact']:.2f} bps")
        print(f"  💥 Impact TWAP (med/std):   {res['median_twap_impact']:.2f} ± {res['std_twap_impact']:.2f} bps")
        
        # Distribution des actions
        print(f"\n  📊 Distribution des actions:")
        action_names = ['0%', '0.25%', '0.5%', '1%', '2%', '5%', '10%', '25%', '50%', '75%', '100%']
        for i, (name, pct) in enumerate(zip(action_names, res['action_distribution'])):
            bar = '█' * int(pct / 2)
            print(f"      Action {i:2d} ({name:>5s}): {pct:5.1f}% {bar}")
        print()
    
    print(f"{'='*80}\n")




def plot_final_inventory_evolution(results: list, horizon_steps: int, initial_inventory: float, save_path: str):
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
                linewidth=2.5, marker='o', markersize=4, markevery=20)
        
        # Tracer la bande d'écart-type
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
    ax.set_title('Évolution de l\'Inventaire - DQN (Moyenne ± Écart-type)\n(100 épisodes de validation)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, horizon_steps])
    ax.set_ylim([0, initial_inventory * 1.05])
    
    plt.tight_layout()
    inventory_plot_path = save_path.replace('.pth', '_inventory_evolution.png')
    plt.savefig(inventory_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Évolution de l'inventaire sauvegardée: {inventory_plot_path}")
    plt.show()


def plot_validation_results(
    episodes, 
    validation_history,
    initial_inventory,
    horizon_steps,
    save_path
):
    """Visualiser les résultats de validation (format 4x3 aligné avec PPO)"""
    
    # Extraire les métriques
    rewards_mean = [v['avg_reward'] for v in validation_history]
    rewards_median = [v['median_reward'] for v in validation_history]
    revenues_mean = [v['avg_revenue'] for v in validation_history]
    revenues_median = [v['median_revenue'] for v in validation_history]
    twap_revenues_mean = [v['avg_twap_revenue'] for v in validation_history]
    twap_revenues_median = [v['median_twap_revenue'] for v in validation_history]
    twap_comparisons_mean = [v['avg_twap_comparison'] for v in validation_history]
    twap_comparisons_median = [v['median_twap_comparison'] for v in validation_history]
    lengths_mean = [v['avg_length'] for v in validation_history]
    lengths_median = [v['median_length'] for v in validation_history]
    avg_impacts = [v['avg_impact'] for v in validation_history]
    twap_impacts = [v['avg_twap_impact'] for v in validation_history]
    all_lengths_history = [v['all_lengths'] for v in validation_history]
    
 
    cvar_history = [v['robust_cvar'] for v in validation_history]
    win_rate_history = [v['robust_win_rate'] for v in validation_history]
    vwap_diff_history = [v['vwap_diff_bps'] for v in validation_history]
    
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('Résultats de VALIDATION - Agent DQN vs TWAP', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Rewards, Revenues Mean, Revenues Median
    ax = axes[0, 0]
    ax.plot(episodes, rewards_mean, 'o-', label='Moyenne', color='darkviolet', linewidth=2, markersize=4)
    ax.plot(episodes, rewards_median, 's-', label='Médiane', color='purple', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Récompenses Totales de l\'Agent')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(episodes, revenues_mean, 'o-', label='Agent DQN (moy)', color='steelblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues_mean, 's-', label='TWAP (moy)', color='orange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux (Moyenne)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    ax = axes[0, 2]
    ax.plot(episodes, revenues_median, 'o-', label='Agent DQN (med)', color='darkblue', linewidth=2, markersize=4)
    ax.plot(episodes, twap_revenues_median, 's-', label='TWAP (med)', color='darkorange', linewidth=2, markersize=4)
    ax.set_title('Revenus Totaux (Médiane)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Revenu (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # Row 2: Perf vs TWAP, Lengths, Impacts
    ax = axes[1, 0]
    ax.plot(episodes, twap_comparisons_mean, 'o-', label='Moyenne', color='green', linewidth=2, markersize=4)
    ax.plot(episodes, twap_comparisons_median, 's-', label='Médiane', color='darkgreen', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Performance vs TWAP (%)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Amélioration (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(episodes, lengths_mean, 'o-', label='Pas effectifs (moy)', color='coral', linewidth=2, markersize=4)
    ax.plot(episodes, lengths_median, 's-', label='Pas effectifs (med)', color='darkred', linewidth=2, markersize=4)
    ax.axhline(y=horizon_steps, color='blue', linestyle='--', alpha=0.5, label=f'TWAP ({horizon_steps} pas)', linewidth=2)
    ax.set_title('Nombre de Pas avec Vente')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Nombre de pas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, horizon_steps * 1.1])
    
    ax = axes[1, 2]
    ax.plot(episodes, avg_impacts, 'o-', label='Agent DQN', color='purple', linewidth=2, markersize=4)
    ax.plot(episodes, twap_impacts, 's-', label='TWAP', color='orange', linewidth=2, markersize=4)
    ax.set_title('Impact de Marché Moyen')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Impact (basis points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 3: Length Distribution, Perf Distribution, VWAP Diff
    ax = axes[2, 0]
    all_validation_episodes = []
    all_validation_lengths = []
    for i, val_ep in enumerate(episodes):
        n_eps_in_validation = len(all_lengths_history[i])
        episode_indices = np.arange(n_eps_in_validation) + val_ep - n_eps_in_validation + 1
        all_validation_episodes.extend(episode_indices)
        all_validation_lengths.extend(all_lengths_history[i])
    
    rolling_mean = []
    window_size = 10
    for i in range(len(all_validation_lengths)):
        start_idx = max(0, i - window_size + 1)
        rolling_mean.append(np.mean(all_validation_lengths[start_idx:i+1]))
    
    ax.scatter(all_validation_episodes, all_validation_lengths, 
               alpha=0.3, s=10, color='steelblue', label='Pas effectifs')
    ax.plot(all_validation_episodes, rolling_mean, 
            color='darkred', linewidth=2.5, label='Moyenne mobile (10 épisodes)')
    ax.axhline(y=horizon_steps, color='blue', linestyle='--', alpha=0.5, label=f'TWAP ({horizon_steps} pas)', linewidth=2)
    ax.set_title('Pas Effectifs par Épisode de Validation')
    ax.set_xlabel('Épisode d\'entraînement')
    ax.set_ylabel('Nombre de pas')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, horizon_steps * 1.1])
    
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
    
    
    ax = axes[2, 2]
    ax.plot(episodes, vwap_diff_history, 'o-', color='teal', label='VWAP Diff (bps)', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('VWAP Différence (bps)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Agent - TWAP (bps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    
    # CVaR
    ax = axes[3, 0]
    ax.plot(episodes, cvar_history, 'o-', color='crimson', label='CVaR (5%)', linewidth=2, markersize=4)
    ax.set_title('Robustesse: CVaR (Bear Markets)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Perte vs TWAP (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Win Rate
    ax = axes[3, 1]
    ax.plot(episodes, win_rate_history, 'o-', color='forestgreen', label='Win Rate', linewidth=2, markersize=4)
    ax.axhline(y=50, color='gray', linestyle='--')
    ax.set_title('Robustesse: Win Rate (Bear Markets)')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('% Victoires')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final Stats Text
    ax = axes[3, 2]
    ax.axis('off')
    
    stats_text = f"""
    📊 STATISTIQUES FINALES (DQN)
    {'─'*45}
    
    🎁 Récompense (moy):      {rewards_mean[-1]:,.2f}
    🎁 Récompense (med):      {rewards_median[-1]:,.2f}
    
    💰 Revenu Agent (moy):    {revenues_mean[-1]:,.0f}
    💰 Revenu Agent (med):    {revenues_median[-1]:,.0f}
    📊 Revenu TWAP (moy):     {twap_revenues_mean[-1]:,.0f}
    📊 Revenu TWAP (med):     {twap_revenues_median[-1]:,.0f}
    
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
    Validations: {len(episodes)}
    Épisodes: {episodes[-1]}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    validation_plot_path = save_path.replace('.pth', '_validation_results.png')
    plt.savefig(validation_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphiques de validation sauvegardés: {validation_plot_path}")
    plt.show()





def train_dqn(
    train_data_path: str,
    test_data_path: str,
    n_episodes: int = 10000,
    initial_inventory: float = 1000,
    horizon_steps: int = 240,
    update_freq: int = 4,
    validation_interval: int = 200,
    validation_episodes: int = 20,
    final_validation_episodes: int = 200,
    save_path: str = '../models/dqn_execution.pth',
    save_freq: int = 500,
    debug_validation: bool = True,
    learning_starts: int = 2000,        # warmup
    gradient_steps: int = 1             # nb updates quand on update
):
    if not os.path.exists(train_data_path):
        print(f"ERREUR: Données d'entraînement introuvables : {train_data_path}")
        return None
    if not os.path.exists(test_data_path):
        print(f"ERREUR: Données de test introuvables : {test_data_path}")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Initialisation des environnements...")
    env_train = OptimalExecutionEnv(
        data_path=train_data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        random_start_prob=0.5
    )
    env_val = OptimalExecutionEnv(
    data_path=train_data_path,
    initial_inventory=initial_inventory,
    horizon_steps=horizon_steps,
    )
    env_test = OptimalExecutionEnv(
        data_path=test_data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
    )

    state_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Horizon: {horizon_steps} steps, Inventaire initial: {initial_inventory}")

    total_steps = n_episodes * horizon_steps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon_decay_steps=total_steps,
        epsilon_end=0.1,
        buffer_size=100000,
        batch_size=64,
        device=device
    )

    # Warmup/gradient steps
    agent.learning_starts = getattr(agent, "learning_starts", learning_starts)
    agent.gradient_steps = getattr(agent, "gradient_steps", gradient_steps)

    validation_episodes_list = []
    validation_history = []


    best_mean_performance = -np.inf
    best_median_performance = -np.inf
    best_vwap_performance = -np.inf          
    best_win_rate_performance = -np.inf

    best_mean_model_path = save_path.replace('.pth', '_best_mean.pth')
    best_median_model_path = save_path.replace('.pth', '_best_median.pth')
    best_vwap_model_path = save_path.replace('.pth', '_best_vwap.pth')  
    best_win_rate_model_path = save_path.replace('.pth', '_best_win_rate.pth')

    print(f"\n{'='*80}")
    print(f"DÉBUT DE L'ENTRAÎNEMENT DQN - {n_episodes} épisodes")
    print(f"Validation tous les {validation_interval} épisodes ({validation_episodes} épisodes/validation)")
    print(f"{'='*80}\n")

    pbar = tqdm(range(n_episodes), desc="Training DQN")

    for episode in pbar:
        state, _ = env_train.reset()
        done = False
        step = 0
        episode_loss = []

        while not done:
            action, _, _ = agent.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            done = terminated or truncated
            step += 1

            if episode < 3 and step <= 5:
                print(f"  [TRAIN] Ep {episode} Step {step}: action={action}, "
                      f"reward={reward:.2f}, qty={info['quantity_sold']:.1f}, "
                      f"inv={info['inventory_remaining']:.0f}")

            agent.store_transition(state, action, reward, next_state, done)

            # updates après warmup seulement
            if agent.steps_done >= agent.learning_starts and (step % update_freq == 0):
                for _ in range(agent.gradient_steps):
                    loss = agent.update()
                    if loss > 0:
                        episode_loss.append(loss)

            state = next_state

        # Validation périodique
        if (episode + 1) % validation_interval == 0:
            metrics = run_validation(
                agent, env_val, validation_episodes,
                horizon_steps, initial_inventory,
                debug_mode=debug_validation
            )

            validation_episodes_list.append(episode + 1)
            validation_history.append(metrics)

            print_validation_stats(episode + 1, n_episodes, metrics, agent.epsilon)

            if metrics['avg_twap_comparison'] > best_mean_performance:
                best_mean_performance = metrics['avg_twap_comparison']
                agent.save(best_mean_model_path)
                print(f"💎 Nouveau meilleur modèle (MOYENNE) ! Performance: {best_mean_performance:.2f}% vs TWAP")

            if metrics['median_twap_comparison'] > best_median_performance:
                best_median_performance = metrics['median_twap_comparison']
                agent.save(best_median_model_path)
                print(f"💎 Nouveau meilleur modèle (MÉDIANE) ! Performance: {best_median_performance:.2f}% vs TWAP")

            if metrics['vwap_diff_bps'] > best_vwap_performance:
                best_vwap_performance = metrics['vwap_diff_bps']
                agent.save(best_vwap_model_path)
                print(f"🛡️ Nouveau meilleur modèle (VWAP) ! VWAP Diff: {best_vwap_performance:.2f} bps")

            if metrics['robust_win_rate'] > best_win_rate_performance:
                best_win_rate_performance = metrics['robust_win_rate']
                agent.save(best_win_rate_model_path)
                print(f"🏆 Nouveau meilleur modèle (WIN RATE) ! Win Rate: {best_win_rate_performance:.1f}%")

            # (optionnel) petit postfix comme PPO
            pbar.set_postfix({
                "ValMean%": f"{metrics['avg_twap_comparison']:.2f}",
                "VWAP(bps)": f"{metrics['vwap_diff_bps']:.0f}",
                "WinRate%": f"{metrics['robust_win_rate']:.0f}"
            })

        # Sauvegarde périodique
        if (episode + 1) % save_freq == 0:
            agent.save(save_path)
            print(f"\n💾 Modèle sauvegardé: {save_path}")

    # Sauvegarde finale
    agent.save(save_path)
    print(f"\n{'='*80}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"💾 Modèle final: {save_path}")
    print(f"💎 Meilleur (moyenne): {best_mean_model_path} ({best_mean_performance:.2f}%)")
    print(f"💎 Meilleur (médiane): {best_median_model_path} ({best_median_performance:.2f}%)")
    print(f"🛡️ Meilleur (VWAP):    {best_vwap_model_path} ({best_vwap_performance:.2f} bps)")
    print(f"🏆 Meilleur (WinRate): {best_win_rate_model_path} ({best_win_rate_performance:.1f}%)")
    print(f"{'='*80}\n")

    # Validation finale
    print(f"\n{'='*80}")
    print(f"🔬 VALIDATION FINALE - {final_validation_episodes} ÉPISODES")
    print(f"{'='*80}\n")

    final_results = []
    final_results.append(run_final_validation(
        agent, env_test, final_validation_episodes,
        horizon_steps, initial_inventory, "DQN - Modèle Final"
    ))

    def _load_agent(path):
        if not os.path.exists(path):
            return None
        a = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            epsilon_decay_steps=max(1, total_steps // 2),
            buffer_size=100000,
            batch_size=64,
            device=device
        )
        a.load(path)
        return a

    best_mean_agent = _load_agent(best_mean_model_path)
    if best_mean_agent is not None:
        final_results.append(run_final_validation(
            best_mean_agent, env_test, final_validation_episodes,
            horizon_steps, initial_inventory, "DQN - Meilleur (Moyenne)"
        ))

    best_median_agent = _load_agent(best_median_model_path)
    if best_median_agent is not None:
        final_results.append(run_final_validation(
            best_median_agent, env_test, final_validation_episodes,
            horizon_steps, initial_inventory, "DQN - Meilleur (Médiane)"
        ))


    best_vwap_agent = _load_agent(best_vwap_model_path)
    if best_vwap_agent is not None:
        final_results.append(run_final_validation(
            best_vwap_agent, env_test, final_validation_episodes,
            horizon_steps, initial_inventory, "DQN - Meilleur (VWAP)"
        ))

    best_win_rate_agent = _load_agent(best_win_rate_model_path)
    if best_win_rate_agent is not None:
        final_results.append(run_final_validation(
            best_win_rate_agent, env_test, final_validation_episodes,
            horizon_steps, initial_inventory, "DQN - Meilleur (WinRate)"
        ))

    print_final_validation_stats(final_results)

    plot_validation_results(
        validation_episodes_list,
        validation_history,
        initial_inventory,
        horizon_steps,
        save_path
    )
    plot_final_inventory_evolution(final_results, horizon_steps, initial_inventory, save_path)

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraîner un agent DQN pour l'exécution optimale.")
    parser.add_argument("--name", type=str, default='dqn_execution_v2',
                        help="Nom du modèle (sans .pth).")
    parser.add_argument("--episodes", type=int, default=10000, help="Nombre d'épisodes")
    parser.add_argument("--validation-interval", type=int, default=200, help="Intervalle de validation")
    parser.add_argument("--validation-episodes", type=int, default=20, help="Épisodes par validation")
    parser.add_argument("--final-episodes", type=int, default=100, help="Épisodes pour validation finale")
    args = parser.parse_args()

    TRAIN_DATA = os.path.join(project_root, 'data', 'raw', 'BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv')
    TEST_DATA = os.path.join(project_root, 'data', 'raw', 'BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv')
    SAVE_PATH = os.path.join(project_root, 'models', f'{args.name}.pth')

    print(f"\n{'='*80}")
    print(f"🚀 ENTRAÎNEMENT DQN - {args.name}")
    print(f"   Les modèles seront sauvegardés sous: {args.name}_best_*.pth")
    print(f"{'='*80}\n")

    agent = train_dqn(
        train_data_path=TRAIN_DATA,
        test_data_path=TEST_DATA,
        n_episodes=args.episodes,
        initial_inventory=1000,
        horizon_steps=240,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        final_validation_episodes=args.final_episodes,
        save_path=SAVE_PATH,
        debug_validation=True
    )
    