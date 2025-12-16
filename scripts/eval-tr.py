import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(".."))
from src.environment.execution_env import OptimalExecutionEnv
from src.models.ppo_agent import PPOAgent
from scripts.train_ppo import run_final_validation, print_final_validation_stats, plot_final_inventory_evolution

# ✅ NEW: Logger Class to save output
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def print_extended_stats(result):
    """
    Calculates and prints additional metrics:
    - Information Ratio (General & Bear)
    - CVaR (Bear) - Agent vs TWAP & TWAP vs Agent (in bps)
    - Win Rate (General & Bear)
    - Best/Worst Bear Market Episodes
    """
    revenues = np.array(result['all_revenues'])
    twap_revenues = np.array(result['all_twap_revenues'])
    indices = np.array(result['all_indices'])
    initial_values = np.array(result['all_initial_values'])
    market_returns = np.array(result['all_market_returns']) 
    
    # Calculate differences
    excess_dollar = revenues - twap_revenues
    # Avoid division by zero
    safe_twap = np.where(twap_revenues == 0, 1e-6, twap_revenues)
    excess_bps = (excess_dollar / safe_twap) * 10000
    
    # 1. General Metrics
    general_win_rate = np.mean(excess_dollar > 0) * 100
    
    # Information Ratio (General)
    if np.std(excess_dollar) > 0:
        ir_general = np.mean(excess_dollar) / np.std(excess_dollar)
    else:
        ir_general = 0.0
        
    # 2. Bear Market Filter (Worst 20% of Market Returns)
    # Sort by market return
    sorted_indices = np.argsort(market_returns)
    n_episodes = len(market_returns)
    cutoff_index = int(n_episodes * 0.20) # Bottom 20%
    
    # Get indices of the worst 20% episodes
    bear_indices_local = sorted_indices[:cutoff_index]
    
    # Create mask
    bear_mask = np.zeros(n_episodes, dtype=bool)
    bear_mask[bear_indices_local] = True
    
    # Filter Data
    bear_revenues = revenues[bear_mask]
    bear_twap = twap_revenues[bear_mask]
    bear_excess_dollar = excess_dollar[bear_mask]
    bear_excess_bps = excess_bps[bear_mask]
    bear_real_indices = indices[bear_mask]
    bear_market_rets = market_returns[bear_mask]
    
    n_bear = len(bear_revenues)
    
    print(f"\n{'='*80}")
    print(f"📈 EXTENDED METRICS & BEAR MARKET ANALYSIS")
    print(f"{'='*80}")
    print(f"📊 General Win Rate:          {general_win_rate:.2f}%")
    print(f"📊 Information Ratio (All):   {ir_general:.4f}")
    print(f"{'-'*80}")
    print(f"🐻 BEAR MARKETS (Worst 20% Returns): {n_bear} episodes")
    if n_bear > 0:
        print(f"   Avg Market Return in Bear: {np.mean(bear_market_rets)*100:.2f}%")
        print(f"   Max Market Return in Bear: {np.max(bear_market_rets)*100:.2f}%")
    
    if n_bear > 0:
        # Bear Metrics
        bear_win_rate = np.mean(bear_excess_dollar > 0) * 100 # <--- NEW: Win Rate for Worst 20%
        
        if np.std(bear_excess_dollar) > 0:
            ir_bear = np.mean(bear_excess_dollar) / np.std(bear_excess_dollar)
        else:
            ir_bear = 0.0
            
        # CVaR 1: Agent Risk (Worst 5% of Agent - TWAP) in bps
        sorted_bps_agent = np.sort(bear_excess_bps)
        cutoff_idx_cvar = int(len(sorted_bps_agent) * 0.05)
        cutoff_idx_cvar = max(1, cutoff_idx_cvar)
        cvar_agent_bps = np.mean(sorted_bps_agent[:cutoff_idx_cvar])
        
        # CVaR 2: TWAP Risk (Worst 5% of TWAP - Agent) in bps
        twap_excess_bps = -bear_excess_bps
        sorted_bps_twap = np.sort(twap_excess_bps)
        cvar_twap_bps = np.mean(sorted_bps_twap[:cutoff_idx_cvar])
            
        print(f"🐻 Win Rate (Bear):          {bear_win_rate:.2f}%") # <--- PRINTED HERE
        print(f"🐻 Information Ratio (Bear):  {ir_bear:.4f}")
        print(f"🐻 CVaR 5% (Agent - TWAP):    {cvar_agent_bps:.2f} bps (Agent Downside Risk)")
        print(f"🐻 CVaR 5% (TWAP - Agent):    {cvar_twap_bps:.2f} bps (TWAP Downside Risk)")
        print(f"{'-'*80}")
        
        # Best & Worst Bear Episodes (Based on raw dollar difference)
        best_idx_local = np.argmax(bear_excess_dollar)
        worst_idx_local = np.argmin(bear_excess_dollar)
        
        best_real_idx = bear_real_indices[best_idx_local]
        worst_real_idx = bear_real_indices[worst_idx_local]
        
        # Best Stats
        b_rev = bear_revenues[best_idx_local]
        b_twap = bear_twap[best_idx_local]
        b_diff = bear_excess_dollar[best_idx_local]
        b_bps = bear_excess_bps[best_idx_local]
        b_mkt = bear_market_rets[best_idx_local]
        
        # Worst Stats
        w_rev = bear_revenues[worst_idx_local]
        w_twap = bear_twap[worst_idx_local]
        w_diff = bear_excess_dollar[worst_idx_local]
        w_bps = bear_excess_bps[worst_idx_local]
        w_mkt = bear_market_rets[worst_idx_local]
        
        print(f"🏆 BEST Bear Performance (Start Index: {best_real_idx})")
        print(f"   Mkt Return: {b_mkt*100:.2f}%")
        print(f"   Agent: ${b_rev:,.2f} | TWAP: ${b_twap:,.2f}")
        print(f"   Diff:  ${b_diff:,.2f} (+{b_bps:.2f} bps)")
        
        print(f"\n💀 WORST Bear Performance (Start Index: {worst_real_idx})")
        print(f"   Mkt Return: {w_mkt*100:.2f}%")
        print(f"   Agent: ${w_rev:,.2f} | TWAP: ${w_twap:,.2f}")
        print(f"   Diff:  ${w_diff:,.2f} ({w_bps:.2f} bps)")
    else:
        print("⚠️ No Bear Markets detected in this sample.")
    print(f"{'='*80}\n")


def evaluate_model(model_path, data_path, n_episodes=1000, horizon_steps=240, initial_inventory=1000, use_real_data=False, full_test=False):
    """
    Load a trained model and run a comprehensive evaluation.
    """
    # ✅ NEW: Setup Logging
    log_filename = f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = Logger(log_filename)
    
    print(f"\n{'='*80}")
    print(f"🧪 ÉVALUATION DU MODÈLE: {os.path.basename(model_path)}")
    
    if full_test:
        print(f"📅 MODE: FULL BACKTEST SÉQUENTIEL (Dataset Complet)")
        use_real_data = True # Force real data for full test
    elif use_real_data:
        print(f"📊 MODE: DONNÉES RÉELLES (Échantillonnage Aléatoire)")
    else:
        print(f"🎲 MODE: SIMULATION GARCH")
    print(f"{'='*80}")
    
    # ✅ FIX: Optimize window for Real Data
    # GARCH needs 5000 steps to calibrate parameters.
    # Real Data only needs ~60 steps for rolling features.
    calib_window = 200 if use_real_data else 5000
    
    # 1. Initialize Environment
    env = OptimalExecutionEnv(
        data_path=data_path,
        initial_inventory=initial_inventory,
        horizon_steps=horizon_steps,
        lambda_0=0.003,
        alpha=0.5,
        delta=0,
        calibration_window=calib_window, # ✅ Apply optimized window
        random_start_prob=0.0, 
        use_real_data=use_real_data 
    )
    
    # ✅ NEW: Configure Sequential Mode if requested
    if full_test:
        # Calculate valid start range
        # min_start will now be ~1200 instead of 6000 for real data
        min_start = env.calibration_window + 60 
        max_start = len(env.historical_data) - horizon_steps - 100
        
        # Calculate exact number of episodes fitting in the data
        available_steps = max_start - min_start
        n_episodes = available_steps // horizon_steps
        
        print(f"ℹ️  Configuration Backtest:")
        print(f"   - Index Début: {min_start}")
        print(f"   - Index Fin:   {max_start}")
        print(f"   - Horizon:     {horizon_steps} min")
        print(f"   - Épisodes:    {n_episodes} (Couverture complète)")
        
        env.set_sequential_backtest(min_start, horizon_steps)
    
    # 2. Initialize Agent
    hidden_dims = [256, 128, 64]
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=3e-4, # LR doesn't matter for evaluation
        gamma=1.0,
        epsilon=0.2,
        lambda_gae=0.95,
        hidden_dims=hidden_dims,
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # 3. Load Weights
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Modèle chargé avec succès.")
    else:
        print(f"❌ ERREUR: Le fichier modèle n'existe pas: {model_path}")
        return

    # 4. Run Validation
    results = []
    result = run_final_validation(
        agent, env, n_episodes,
        horizon_steps, initial_inventory, "Agent Évalué"
    )
    results.append(result)
    
    # 5. Print Stats
    print(f"\nℹ️  NOTE: The 'Robustness (Bear Mkts)' stats below use the OLD definition: TWAP Revenue <= Initial Portfolio Value.")
    print_final_validation_stats(results)
    
    # ✅ NEW: Print Extended Stats
    print_extended_stats(result)
    
    print(f"\n💾 Log saved to: {log_filename}")
    
    # 6. Plot Inventory Evolution
    plot_final_inventory_evolution(results, horizon_steps, initial_inventory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle PPO entraîné.")
    parser.add_argument("--model", type=str,default='../models/ppo_execution_best_median_cap0.pth', help="Chemin vers le fichier .pth du modèle")
    parser.add_argument("--data", type=str, default='../data/raw/BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv', help="Chemin vers les données de test")
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes (ignoré si --full)")

    parser.add_argument("--real", action="store_false", help="Utiliser les données réelles (aléatoire)") 
    parser.add_argument("--full", action="store_false", help="Backtest complet séquentiel sur tout le fichier") 
    
    args = parser.parse_args()
    
    # Check if data file exists, fallback to train data if test not found
    if not os.path.exists(args.data):
        print(f"⚠️ Données de test introuvables: {args.data}")
        train_data = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
        if os.path.exists(train_data):
            print(f"🔄 Utilisation des données d'entraînement: {train_data}")
            args.data = train_data
        else:
            print("❌ Aucune donnée trouvée.")
            sys.exit(1)
            
    evaluate_model(args.model, args.data, args.episodes, use_real_data=args.real, full_test=args.full, horizon_steps=240, initial_inventory=1000)