import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple


class ActorCritic(nn.Module):
    """Réseau Actor-Critic pour PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super(ActorCritic, self).__init__()
        
        # Réseau partagé
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Tête Actor (politique) - MODIFIÉ: Retourne des logits, pas des probabilités
        self.actor = nn.Linear(prev_dim, action_dim)
        
        # Tête Critic (valeur)
        self.critic = nn.Linear(prev_dim, 1)
        
    def forward(self, state):
        """Forward pass"""
        features = self.shared_net(state)
        action_logits = self.actor(features)  # MODIFIÉ: logits au lieu de probs
        state_value = self.critic(features)
        return action_logits, state_value


class PPOAgent:
    """Agent PPO pour l'exécution optimale"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.05,
        lambda_gae: float = 0.95,  # NOUVEAU: Paramètre GAE
        hidden_dims: List[int] = [64, 64],
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.lambda_gae = lambda_gae  # NOUVEAU
        
        # Réseau Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffer pour stocker les expériences
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Sélectionner une action selon la politique actuelle"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, state_value = self.policy(state)  # MODIFIÉ: logits
        
        # MODIFIÉ: Utiliser les logits directement
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=1).item()
        else:
            action = dist.sample().item()
        
        # Calculer log_prob pour l'action choisie
        log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, state_value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Stocker une transition dans le buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculer les returns et les advantages avec GAE (Generalized Advantage Estimation)
        
        Args:
            next_value: Valeur de l'état suivant le dernier état de l'épisode
        
        Returns:
            tuple: (returns, advantages)
        """
        num_steps = len(self.rewards)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        # Initialiser les tenseurs
        advantages = torch.zeros(num_steps).to(self.device)
        
        # Calculer GAE en remontant le temps
        last_gae_lam = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # Dernier pas : utiliser next_value fourni
                next_non_terminal = 1.0 - dones[t]
                next_value_tensor = torch.tensor(next_value).to(self.device)
            else:
                # Pas intermédiaire : utiliser la valeur de l'état suivant dans le buffer
                next_non_terminal = 1.0 - dones[t]
                next_value_tensor = values[t + 1]
            
            # Calcul du TD error (delta)
            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_tensor * next_non_terminal - values[t]
            
            # Calcul de l'avantage GAE
            # A_t = delta_t + (gamma * lambda) * (1 - done) * A_{t+1}
            last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        # Calculer les returns (cibles pour le critique)
        # returns = advantages + values
        returns = advantages + values
        
        # Normaliser les advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, next_value: float, epochs: int = 4, batch_size: int = 64):
        """
        Mettre à jour la politique avec PPO
        
        Args:
            next_value: Valeur de l'état suivant le dernier état
            epochs: Nombre d'époques de mise à jour
            batch_size: Taille des mini-batches
        """
        if len(self.states) == 0:
            return
        
        # Convertir en tenseurs
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Calculer returns et advantages avec GAE
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # Mise à jour PPO
        for _ in range(epochs):
            # Échantillonner des mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_logits, state_values = self.policy(batch_states)  # MODIFIÉ: logits
                dist = Categorical(logits=action_logits)  # MODIFIÉ: utiliser logits
                
                # Calculer les nouvelles log_probs
                new_log_probs = dist.log_prob(batch_actions)
                
                # Ratio de probabilité
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Loss de politique avec clipping
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Loss de valeur
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # Entropie pour encourager l'exploration
                entropy = dist.entropy().mean()
                
                # Loss totale
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # Mise à jour
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Vider le buffer
        self.clear_buffer()
    
    def clear_buffer(self):
        """Vider le buffer d'expériences"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def save(self, path: str):
        """Sauvegarder le modèle"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Charger le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])