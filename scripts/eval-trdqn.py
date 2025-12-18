import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import torch
from typing import Tuple, Dict, Any, Optional

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.environment.execution_envdqn import OptimalExecutionEnv
from src.models.dqn_agent import DQNAgent


class Logger(object):
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()



def evaluate_targeted_robustness(agent_revenues, twap_revenues, initial_values):
    """
    CVaR + WinRate UNIQUEMENT sur épisodes où TWAP <= valeur initiale (arrival).
    (Conserve ta définition existante pour continuité des stats “robust_*”.)
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
    cutoff_idx = max(1, cutoff_idx)
    cvar_5 = float(np.mean(sorted_alpha[:cutoff_idx]))

    win_rate = float(np.mean(filtered_excess > 0) * 100.0)
    return cvar_5, win_rate



def print_extended_stats(result: dict):
    revenues = np.array(result.get("all_revenues", []), dtype=np.float64)
    twap_revenues = np.array(result.get("all_twap_revenues", []), dtype=np.float64)
    indices = np.array(result.get("all_indices", []), dtype=np.int64)
    initial_values = np.array(result.get("all_initial_values", []), dtype=np.float64)
    market_returns = np.array(result.get("all_market_returns", []), dtype=np.float64)

    if revenues.size == 0 or twap_revenues.size == 0:
        print("\n⚠️ Extended stats: données manquantes (all_revenues / all_twap_revenues).")
        return

    excess_dollar = revenues - twap_revenues
    safe_twap = np.where(twap_revenues == 0, 1e-6, twap_revenues)
    excess_bps = (excess_dollar / safe_twap) * 10000.0

    general_win_rate = float(np.mean(excess_dollar > 0) * 100.0)
    ir_general = float(np.mean(excess_dollar) / np.std(excess_dollar)) if np.std(excess_dollar) > 0 else 0.0

    n_episodes = len(revenues)
    if market_returns.size != n_episodes:
        market_returns = np.zeros(n_episodes, dtype=np.float64)

    sorted_idx = np.argsort(market_returns)
    cutoff = int(n_episodes * 0.20)
    cutoff = max(1, cutoff)
    bear_local_idx = sorted_idx[:cutoff]

    bear_mask = np.zeros(n_episodes, dtype=bool)
    bear_mask[bear_local_idx] = True

    bear_revenues = revenues[bear_mask]
    bear_twap = twap_revenues[bear_mask]
    bear_excess_dollar = excess_dollar[bear_mask]
    bear_excess_bps = excess_bps[bear_mask]
    bear_real_indices = indices[bear_mask] if indices.size == n_episodes else np.full(len(bear_revenues), -1)
    bear_market_rets = market_returns[bear_mask]

    n_bear = len(bear_revenues)

    print(f"\n{'='*80}")
    print(f"📈 EXTENDED METRICS & BEAR MARKET ANALYSIS (Worst 20% Market Returns)")
    print(f"{'='*80}")
    print(f"📊 General Win Rate:          {general_win_rate:.2f}%")
    print(f"📊 Information Ratio (All):   {ir_general:.4f}")
    print(f"{'-'*80}")
    print(f"🐻 BEAR MARKETS (Worst 20%):  {n_bear} episodes")
    if n_bear > 0:
        print(f"   Avg Market Return (Bear):  {np.mean(bear_market_rets)*100:.2f}%")
        print(f"   Max Market Return (Bear):  {np.max(bear_market_rets)*100:.2f}%")
        print(f"   Min Market Return (Bear):  {np.min(bear_market_rets)*100:.2f}%")

    if n_bear <= 0:
        print("⚠️ No Bear Markets detected in this sample.")
        print(f"{'='*80}\n")
        return

    bear_win_rate = float(np.mean(bear_excess_dollar > 0) * 100.0)
    ir_bear = float(np.mean(bear_excess_dollar) / np.std(bear_excess_dollar)) if np.std(bear_excess_dollar) > 0 else 0.0

    sorted_bps_agent = np.sort(bear_excess_bps)
    cvar_cut = int(len(sorted_bps_agent) * 0.05)
    cvar_cut = max(1, cvar_cut)
    cvar_agent_bps = float(np.mean(sorted_bps_agent[:cvar_cut]))

    twap_excess_bps = -bear_excess_bps
    sorted_bps_twap = np.sort(twap_excess_bps)
    cvar_twap_bps = float(np.mean(sorted_bps_twap[:cvar_cut]))

    print(f"🐻 Win Rate (Bear):           {bear_win_rate:.2f}%")
    print(f"🐻 Information Ratio (Bear):  {ir_bear:.4f}")
    print(f"🐻 CVaR 5% (Agent - TWAP):    {cvar_agent_bps:.2f} bps (Agent Downside Risk)")
    print(f"🐻 CVaR 5% (TWAP - Agent):    {cvar_twap_bps:.2f} bps (TWAP Downside Risk)")
    print(f"{'-'*80}")

    best_i = int(np.argmax(bear_excess_dollar))
    worst_i = int(np.argmin(bear_excess_dollar))

    b_idx = int(bear_real_indices[best_i]) if len(bear_real_indices) else -1
    w_idx = int(bear_real_indices[worst_i]) if len(bear_real_indices) else -1

    b_rev, b_twap, b_diff, b_bps, b_mkt = (
        float(bear_revenues[best_i]),
        float(bear_twap[best_i]),
        float(bear_excess_dollar[best_i]),
        float(bear_excess_bps[best_i]),
        float(bear_market_rets[best_i]),
    )
    w_rev, w_twap, w_diff, w_bps, w_mkt = (
        float(bear_revenues[worst_i]),
        float(bear_twap[worst_i]),
        float(bear_excess_dollar[worst_i]),
        float(bear_excess_bps[worst_i]),
        float(bear_market_rets[worst_i]),
    )

    print(f"🏆 BEST Bear Performance (Start Index: {b_idx})")
    print(f"   Mkt Return: {b_mkt*100:.2f}%")
    print(f"   Agent: {b_rev:,.2f} | TWAP: {b_twap:,.2f}")
    print(f"   Diff:  {b_diff:,.2f} (+{b_bps:.2f} bps)")

    print(f"\n💀 WORST Bear Performance (Start Index: {w_idx})")
    print(f"   Mkt Return: {w_mkt*100:.2f}%")
    print(f"   Agent: {w_rev:,.2f} | TWAP: {w_twap:,.2f}")
    print(f"   Diff:  {w_diff:,.2f} ({w_bps:.2f} bps)")
    print(f"{'='*80}\n")



def preview_twap_step(env: OptimalExecutionEnv) -> Tuple[float, float, float]:
    """
    Reproduit à l'identique la logique TWAP utilisée dans env.step(),
    SANS modifier l'état de l'env (juste pour logger impacts/quantités).
    Retourne: (twap_quantity, twap_impact_relative, twap_revenue_step)
    """
    current_price = float(env.prices_history[-1])
    realized_vol = env._calculate_realized_volatility(np.array(env.prices_history, dtype=np.float64))

    rolling_sigma = env._calculate_rolling_mean(env.realized_vols_history, env.avg_window)
    rolling_volume = env._calculate_rolling_mean(env.volumes_history, env.avg_window)

    time_remaining = env.horizon_steps - env.current_step

    twap_revenue = 0.0
    twap_quantity = 0.0
    twap_impact = 0.0

    if time_remaining > 0 and env.twap_inventory > 1e-6:
        twap_quantity = env.twap_inventory / time_remaining
        twap_quantity = min(twap_quantity, env.twap_inventory)

        twap_impact = env._calculate_temporary_impact(
            twap_quantity, realized_vol, rolling_sigma, rolling_volume
        )
        twap_execution_price = current_price * (1.0 - twap_impact)
        twap_revenue = twap_quantity * twap_execution_price

    return float(twap_quantity), float(twap_impact), float(twap_revenue)


def run_final_validation_extended(
    agent: DQNAgent,
    env: OptimalExecutionEnv,
    n_episodes: int,
    horizon_steps: int,
    initial_inventory: float,
    agent_name: str,
    seed: Optional[int] = None
) -> dict:
    """
    Fixes:
    - TWAP revenue = somme des info['twap_revenue'] renvoyés par env.step()
      (donc même logique que l'env, sans TWAP "offline" désynchronisé)
    - Impacts/quantités TWAP loggés via preview_twap_step() (même formules que l'env)
    - Avertit si TWAP n'est pas liquidé à la fin (devrait être ~0)
    """
    print(f"\n🔍 Validation finale: {agent_name} ({n_episodes} épisodes)")
    if seed is not None:
        print(f"🎲 Seed éval: {seed}")

    ep_revenues, ep_lengths, ep_impacts = [], [], []
    ep_twap_comparisons, ep_twap_revenues, ep_twap_impacts = [], [], []
    all_actions = []

    ep_vwap_diffs, ep_timing_bias, ep_twap_timing_bias = [], [], []
    ep_initial_values = []

    all_revenues, all_twap_revenues = [], []
    all_indices, all_market_returns = [], []

    inventory_trajectories = np.zeros((n_episodes, horizon_steps + 1), dtype=np.float64)

    warned_twap_mismatch = False
    warned_twap_not_flat = False

    for ep_idx in range(n_episodes):
        ep_seed = None if seed is None else (seed + ep_idx)
        state, _ = env.reset(seed=ep_seed)

        done = False
        step = 0
        effective_steps = 0

        episode_impacts_local, episode_prices_local, episode_volumes_local = [], [], []
        episode_actions = []

        execution_times, execution_quantities = [], []
        twap_execution_times, twap_execution_quantities = [], []
        twap_impacts_local = []

        inventory_trajectories[ep_idx, 0] = initial_inventory

        # initial (arrival) value
        ep_initial_values.append(float(initial_inventory * env.initial_price))

        # start index
        start_idx = getattr(env, "random_start_idx", -1)
        all_indices.append(int(start_idx))

        twap_total_revenue = 0.0

        while not done:
            # ── preview TWAP (pour logs d'impact/timing uniquement)
            tw_q, tw_imp, tw_rev_preview = preview_twap_step(env)
            if tw_q > 1e-6:
                twap_impacts_local.append(tw_imp * 10000.0)
                twap_execution_times.append(step)
                twap_execution_quantities.append(tw_q)

            # ── Agent greedy
            action, _, _ = agent.select_action(state, deterministic=True)
            episode_actions.append(int(action))

            next_state, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            # ── Accumule TWAP EXACT env.step()
            tw_step = float(info.get("twap_revenue", 0.0))
            twap_total_revenue += tw_step

            # Sanity check (une fois)
            if (not warned_twap_mismatch) and (abs(tw_step - tw_rev_preview) > 1e-6):
                warned_twap_mismatch = True
                print("⚠️ Warning: twap_revenue (info) != twap_revenue (preview). "
                      "On fait confiance à info['twap_revenue'].")

            # Logs agent
            qty = float(info.get('quantity_sold', 0.0))
            if qty > 1e-6:
                effective_steps += 1
                episode_impacts_local.append(float(info.get('temp_impact_relative', 0.0)) * 10000.0)
                episode_prices_local.append(float(info.get('execution_price', 0.0)))
                episode_volumes_local.append(qty)

                execution_times.append(step)
                execution_quantities.append(qty)

            inventory_trajectories[ep_idx, step + 1] = float(info.get('inventory_remaining', 0.0))

            state = next_state
            step += 1

        agent_revenue = float(info.get('total_revenue', 0.0))

        # check TWAP inventory end
        twap_inv_end = float(getattr(env, "twap_inventory", 0.0))
        if (not warned_twap_not_flat) and (abs(twap_inv_end) > 1e-4):
            warned_twap_not_flat = True
            print(f"⚠️ Warning: TWAP inventory end non nul ({twap_inv_end:.6f}). "
                  f"Vérifie la logique TWAP interne / horizon.")

        ep_revenues.append(agent_revenue)
        ep_lengths.append(float(effective_steps))
        ep_impacts.append(float(np.mean(episode_impacts_local)) if episode_impacts_local else 0.0)

        ep_twap_revenues.append(float(twap_total_revenue))
        ep_twap_impacts.append(float(np.mean(twap_impacts_local)) if twap_impacts_local else 0.0)

        all_actions.extend(episode_actions)

        relative_performance = ((agent_revenue - twap_total_revenue) / twap_total_revenue) * 100.0 if twap_total_revenue > 1e-6 else 0.0
        ep_twap_comparisons.append(float(relative_performance))

        # VWAP diff bps
        total_vol_agent = float(np.sum(episode_volumes_local))
        vwap_agent = float(np.sum(np.array(episode_prices_local) * np.array(episode_volumes_local)) / total_vol_agent) if total_vol_agent > 0 else 0.0
        vwap_twap = float(twap_total_revenue / initial_inventory) if initial_inventory > 0 else 0.0
        ep_vwap_diffs.append(float(((vwap_agent - vwap_twap) / vwap_twap) * 10000.0) if vwap_twap > 0 else 0.0)

        # Timing bias
        if np.sum(execution_quantities) > 0:
            center_of_mass = float(np.average(execution_times, weights=execution_quantities))
            ep_timing_bias.append(float(center_of_mass / horizon_steps))
        else:
            ep_timing_bias.append(0.5)

        if np.sum(twap_execution_quantities) > 0:
            twap_center_of_mass = float(np.average(twap_execution_times, weights=twap_execution_quantities))
            ep_twap_timing_bias.append(float(twap_center_of_mass / horizon_steps))
        else:
            ep_twap_timing_bias.append(0.5)

        # Market return (end mid / initial - 1)
        final_mid = float(env.prices_history[-1])
        mkt_ret = (final_mid / float(env.initial_price)) - 1.0 if float(env.initial_price) > 0 else 0.0
        all_market_returns.append(float(mkt_ret))

        all_revenues.append(agent_revenue)
        all_twap_revenues.append(float(twap_total_revenue))

    action_counts = np.bincount(np.array(all_actions, dtype=np.int64), minlength=env.action_space.n)
    action_percentages = (action_counts / max(1, len(all_actions))) * 100.0

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
        'robust_win_rate': float(robust_win_rate),

        'all_revenues': list(map(float, all_revenues)),
        'all_twap_revenues': list(map(float, all_twap_revenues)),
        'all_indices': list(map(int, all_indices)),
        'all_initial_values': list(map(float, ep_initial_values)),
        'all_market_returns': list(map(float, all_market_returns)),
    }



def print_final_validation_stats(results: list):
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION FINALE - COMPARAISON DES MODÈLES")
    print(f"{'='*80}\n")

    action_names = ['0%', '0.25%', '0.5%', '1%', '2%', '5%', '10%', '25%', '50%', '75%', '100%']

    for res in results:
        print(f"🤖 {res['agent_name']}")
        print(f"  {'─'*76}")
        print(f"  💰 Revenu Agent (moy):   {res['avg_revenue']:>15,.2f} USDT")
        print(f"  💰 Revenu TWAP (moy):    {res['avg_twap_revenue']:>15,.2f} USDT")
        print(f"  🎯 Perf vs TWAP (moy):   {res['avg_twap_comparison']:>15.2f} %")
        print(f"  🎯 Perf vs TWAP (med):   {res['median_twap_comparison']:>15.2f} %")
        print(f"  🛡️ CVaR (old robust):    {res['robust_cvar']:>15.2f} USDT")
        print(f"  🏆 Win Rate (old robust): {res['robust_win_rate']:>14.1f} %")
        print(f"  ⚖️ VWAP Différence:      {res['avg_vwap_diff']:>15.2f} bps")
        print(f"  ⏳ Timing Bias:          {res['avg_timing_bias']:>15.2f} vs {res['avg_twap_timing_bias']:.2f} (TWAP)")
        print(f"  📏 Pas effectifs (moy):  {res['avg_length']:>15.1f}")
        print(f"  📏 Pas effectifs (min/max): {res['min_length']:.0f} / {res['max_length']:.0f}")
        print(f"  💥 Impact moyen Agent:   {res['avg_impact']:>15.2f} bps")
        print(f"  💥 Impact moyen TWAP:    {res['avg_twap_impact']:>15.2f} bps")
        print(f"  💥 Impact Agent (med/std):  {res['median_impact']:.2f} ± {res['std_impact']:.2f} bps")
        print(f"  💥 Impact TWAP (med/std):   {res['median_twap_impact']:.2f} ± {res['std_twap_impact']:.2f} bps")

        print(f"\n  📊 Distribution des actions:")
        dist = res.get('action_distribution', np.zeros(len(action_names)))
        for i, (name, pct) in enumerate(zip(action_names, dist)):
            bar = '█' * int(pct / 2)
            print(f"      Action {i:2d} ({name:>5s}): {pct:5.1f}% {bar}")
        print()

    print(f"{'='*80}\n")


def plot_final_inventory_evolution(results: list, horizon_steps: int, initial_inventory: float, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    time_steps = np.arange(0, horizon_steps + 1)

    for res in results:
        avg_traj = res['avg_inventory_trajectory']
        std_traj = res.get('std_inventory_trajectory', np.zeros_like(avg_traj))

        ax.plot(time_steps, avg_traj, linewidth=2.5, marker='o', markersize=4, markevery=20,
                label=f"{res['agent_name']} (moy)")
        ax.fill_between(time_steps,
                        np.maximum(0, avg_traj - std_traj),
                        np.minimum(initial_inventory, avg_traj + std_traj),
                        alpha=0.2)

    twap_trajectory = np.linspace(initial_inventory, 0, horizon_steps + 1)
    ax.plot(time_steps, twap_trajectory, '--', linewidth=2, alpha=0.7, label='TWAP (référence)')

    ax.set_xlabel('Pas de temps', fontsize=13)
    ax.set_ylabel('Inventaire', fontsize=13)
    ax.set_title('Évolution Inventaire (moy ± std)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, horizon_steps])
    ax.set_ylim([0, initial_inventory * 1.05])

    plt.tight_layout()
    if save_path:
        out = save_path.replace('.pth', '_inventory_evolution.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n📊 Inventory plot saved: {out}")
    plt.show()


def evaluate_model(
    model_path: str,
    data_path: str,
    n_episodes: int = 1000,
    horizon_steps: int = 240,
    initial_inventory: float = 1000,
    use_real_data: bool = False,
    full_test: bool = False,
    delta_eval: float = 0.0,
    seed: Optional[int] = None
):
    log_filename = f"eval_dqn_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = Logger(log_filename)

    print(f"\n{'='*80}")
    print(f"🧪 ÉVALUATION DQN (FIXED): {os.path.basename(model_path)}")
    if full_test:
        print(f"📅 MODE: FULL BACKTEST SÉQUENTIEL (Dataset Complet) [REAL DATA FORCÉ]")
        use_real_data = True
    elif use_real_data:
        print(f"📊 MODE: DONNÉES RÉELLES (Échantillonnage Aléatoire)")
    else:
        print(f"🎲 MODE: SIMULATION GARCH")
    print(f"{'='*80}")


    print(f"⚙️ Paramètres éval: delta={delta_eval:.4f} (default recommandé: 0.0)")
    if delta_eval > 0:
        print("⚠️ WARNING: delta>0 peut biaiser la comparaison vs TWAP dans cet env "
              "(perm impact piloté par l'agent).")

    calib_window = 200 if use_real_data else 5000

    env = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.003,
        alpha=0.5,
        delta=delta_eval,              # ✅ FIX: delta contrôlé en éval
        calibration_window=calib_window,
        random_start_prob=0.0,
        use_real_data=use_real_data
    )

    if full_test:
        min_start = env.calibration_window + 1000
        max_start = len(env.historical_data) - horizon_steps - 100
        available_steps = max_start - min_start
        n_episodes = max(1, available_steps // horizon_steps)

        print(f"ℹ️  Configuration Backtest:")
        print(f"   - Index Début: {min_start}")
        print(f"   - Index Fin:   {max_start}")
        print(f"   - Horizon:     {horizon_steps} min")
        print(f"   - Épisodes:    {n_episodes} (Couverture complète approx.)")

        env.set_sequential_backtest(min_start, horizon_steps)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        epsilon_decay_steps=max(1, n_episodes * horizon_steps),
        buffer_size=100000,
        batch_size=64,
        device=device
    )

    if not os.path.exists(model_path):
        print(f"❌ ERREUR: modèle introuvable: {model_path}")
        return

    agent.load(model_path)
    print(f"✅ Modèle chargé avec succès.")
    print(f"💾 Log saved to: {log_filename}")

    result = run_final_validation_extended(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        horizon_steps=horizon_steps,
        initial_inventory=initial_inventory,
        agent_name="DQN - Agent Évalué (FIXED)",
        seed=seed
    )

    print_final_validation_stats([result])
    print_extended_stats(result)
    plot_final_inventory_evolution([result], horizon_steps, initial_inventory, save_path=model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle DQN (éval corrigée / TWAP fiable).")
    parser.add_argument("--model", type=str, default=os.path.join(project_root, "models", "dqn_execution_v2.pth"),
                        help="Chemin vers le fichier .pth du modèle DQN")
    parser.add_argument("--data", type=str, default=os.path.join(project_root, "data", "raw", "BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv"),
                        help="Chemin vers les données (csv)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Nombre d'épisodes (ignoré si --full)")

    parser.add_argument("--garch", action="store_true", help="Utiliser GARCH (sinon real data)")
    parser.add_argument("--full", action="store_true", help="Backtest séquentiel complet")

    parser.add_argument("--horizon", type=int, default=240, help="Horizon en pas")
    parser.add_argument("--inventory", type=float, default=1000, help="Inventaire initial")

    # ✅ NEW: delta contrôlé (par défaut 0.0 pour éviter le biais perm impact vs TWAP)
    parser.add_argument("--delta", type=float, default=0.0, help="Permanent impact delta en évaluation (recommandé: 0.0)")
    parser.add_argument("--seed", type=int, default=None, help="Seed pour reproductibilité (optionnel)")

    args = parser.parse_args()

    use_real_data = not args.garch

    if not os.path.exists(args.data):
        print(f"⚠️ Données introuvables: {args.data}")
        alt = os.path.join(project_root, "data", "raw", "BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv")
        if os.path.exists(alt):
            print(f"🔄 Utilisation fallback train: {alt}")
            args.data = alt
        else:
            print("❌ Aucune donnée trouvée.")
            sys.exit(1)

    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        n_episodes=args.episodes,
        horizon_steps=args.horizon,
        initial_inventory=args.inventory,
        use_real_data=use_real_data,
        full_test=args.full,
        delta_eval=args.delta,
        seed=args.seed
    )
