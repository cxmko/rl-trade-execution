import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from src.environment.execution_env import OptimalExecutionEnv
from src.models.ppo_agent import PPOAgent


def evaluate_agent(
    agent: PPOAgent,
    env: OptimalExecutionEnv,
    n_episodes: int = 100
):
    """Évaluer l'agent entraîné"""
    
    episode_revenues = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            # Action déterministe (pas d'exploration)
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        
        episode_revenues.append(info['total_revenue'])
        episode_lengths.append(step)
    
    print(f"\nRésultats sur {n_episodes} épisodes:")
    print(f"  Revenu moyen: {np.mean(episode_revenues):.2f} ± {np.std(episode_revenues):.2f}")
    print(f"  Longueur moyenne: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_revenues


def compare_strategies(
    agent: PPOAgent,
    test_data_path: str,
    n_episodes: int = 100
):
    """Comparer l'agent RL avec des stratégies de référence (TWAP, VWAP)"""
    
    # Créer environnement de test avec données réelles
    env_test = OptimalExecutionEnv(
        data_path=test_data_path,
        initial_inventory=1.0,
        horizon_steps=60
    )
    
    # Évaluer l'agent RL
    print("\n=== Agent RL (PPO) ===")
    rl_revenues = evaluate_agent(agent, env_test, n_episodes)
    
    # Stratégie TWAP (Time-Weighted Average Price)
    print("\n=== Stratégie TWAP (baseline) ===")
    twap_revenues = []
    
    for episode in range(n_episodes):
        state, _ = env_test.reset()
        done = False
        
        while not done:
            # TWAP: vendre uniformément à chaque pas
            twap_action = 1  # 10% de l'inventaire à chaque pas
            state, reward, terminated, truncated, info = env_test.step(twap_action)
            done = terminated or truncated
        
        twap_revenues.append(info['total_revenue'])
    
    print(f"  Revenu moyen TWAP: {np.mean(twap_revenues):.2f} ± {np.std(twap_revenues):.2f}")
    
    # Stratégie agressive (vendre tout d'un coup)
    print("\n=== Stratégie agressive (baseline) ===")
    aggressive_revenues = []
    
    for episode in range(n_episodes):
        state, _ = env_test.reset()
        # Vendre 100% immédiatement
        state, reward, terminated, truncated, info = env_test.step(10)
        aggressive_revenues.append(info['total_revenue'])
    
    print(f"  Revenu moyen agressif: {np.mean(aggressive_revenues):.2f} ± {np.std(aggressive_revenues):.2f}")
    
    # Visualiser la comparaison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([rl_revenues, twap_revenues, aggressive_revenues],
                labels=['RL (PPO)', 'TWAP', 'Agressif'])
    plt.ylabel('Revenu total')
    plt.title('Comparaison des stratégies')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(rl_revenues, alpha=0.5, label='RL (PPO)', bins=20)
    plt.hist(twap_revenues, alpha=0.5, label='TWAP', bins=20)
    plt.hist(aggressive_revenues, alpha=0.5, label='Agressif', bins=20)
    plt.xlabel('Revenu total')
    plt.ylabel('Fréquence')
    plt.title('Distribution des revenus')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../sample/strategy_comparison.png', dpi=150)
    plt.show()
    
    # Calculer l'amélioration
    improvement_vs_twap = (np.mean(rl_revenues) - np.mean(twap_revenues)) / np.mean(twap_revenues) * 100
    print(f"\n=== Amélioration par rapport à TWAP: {improvement_vs_twap:.2f}% ===")


if __name__ == "__main__":
    # Chemins
    model_path = '../models/ppo_execution.pth'
    train_data_path = '../data/raw/BTCUSDT_1m_train_2023-01-01_to_2023-12-31.csv'
    test_data_path = '../data/raw/BTCUSDT_1m_test_2024-01-01_to_2024-12-31.csv'
    
    # Créer environnement et agent
    print("Chargement de l'environnement et de l'agent...")
    env = OptimalExecutionEnv(
        data_path=train_data_path,
        initial_inventory=1.0,
        horizon_steps=60
    )
    
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # Charger le modèle entraîné
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Modèle chargé depuis {model_path}")
    else:
        print(f"Erreur: Modèle non trouvé à {model_path}")
        exit(1)
    
    # Comparer les stratégies sur données de test
    compare_strategies(agent, test_data_path, n_episodes=100)