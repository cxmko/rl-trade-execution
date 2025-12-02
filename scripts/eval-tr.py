import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(".."))
from src.environment.execution_env import OptimalExecutionEnv
from src.models.ppo_agent import PPOAgent
from scripts.train_ppo import run_final_validation, print_final_validation_stats, plot_final_inventory_evolution

def evaluate_model(model_path, data_path, n_episodes=1000, horizon_steps=240, initial_inventory=1000, use_real_data=False, full_test=False):
    """
    Load a trained model and run a comprehensive evaluation.
    """
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
    print_final_validation_stats(results)
    
    # 6. Plot Inventory Evolution
    plot_final_inventory_evolution(results, horizon_steps, initial_inventory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle PPO entraîné.")
    parser.add_argument("--model", type=str,default='../models/ppo_execution_best_cvar.pth', help="Chemin vers le fichier .pth du modèle")
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
            
    evaluate_model(args.model, args.data, args.episodes, use_real_data=args.real, full_test=args.full)