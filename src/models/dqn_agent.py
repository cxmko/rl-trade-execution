import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class QNetwork(nn.Module):
    """Réseau Q-Value avec architecture identique au PPO (Tanh activation)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        gain = nn.init.calculate_gain('tanh')  # ≈ 5/3
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

        last_layer = self.net[-1]
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        nn.init.constant_(last_layer.bias, 0.0)

        
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Buffer de replay pour stocker les transitions"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agent DQN avec Double DQN pour l'exécution optimale d'ordres.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,  
        epsilon_decay_steps: int = 100000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        device: str = 'cpu'
    ):
        self.tau = 0.005
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = 2000
        self.updates_done = 0
        
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Réseaux
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        # Distribution biaisée vers la vente (exemple)
        self.exploration_probs = np.array(
            [0.02,0.13, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.05, 0.05, 0.05],
            dtype=np.float64
        )

        # Normalisation + sécurité
        if self.exploration_probs.shape[0] != self.action_dim:
            self.exploration_probs = np.ones(self.action_dim, dtype=np.float64) / self.action_dim
        else:
            s = self.exploration_probs.sum()
            self.exploration_probs = self.exploration_probs / (s if s > 0 else 1.0)
        
        self.exploration_probs = np.asarray(self.exploration_probs, dtype=np.float64)

        # Sécurité si action_dim change
        if self.exploration_probs.shape[0] != self.action_dim:
            self.exploration_probs = np.ones(self.action_dim, dtype=np.float64) / self.action_dim
        else:
            s = self.exploration_probs.sum()
            self.exploration_probs = self.exploration_probs / (s if s > 0 else 1.0)

    def soft_update_target(self):
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * pp.data)

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        if deterministic:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
            return action, 0.0, 0.0

        if random.random() < self.epsilon:
            action = int(np.random.choice(self.action_dim, p=self.exploration_probs))
            return action, 0.0, 0.0

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
        return action, 0.0, 0.0
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocker une transition dans le replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1
        
        # Décroissance linéaire de epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)
    
    def update(self) -> float:
        """Mise à jour du réseau avec Double DQN."""
        if self.steps_done < self.learning_starts:
            return 0.0

        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Échantillonner un batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convertir en tenseurs
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Q-values actuelles
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Loss Huber (plus robuste que MSE)
        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        self.updates_done += 1
        
        # Mise à jour du target network
        self.soft_update_target()
        
        return loss.item()
    
    def save(self, path: str):
        """Sauvegarder le modèle"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'updates_done': self.updates_done
        }, path)
        print(f"Modèle DQN sauvegardé dans {path}")
    
    def load(self, path: str):
        """Charger un modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.updates_done = checkpoint.get('updates_done', 0)
        print(f"Modèle DQN chargé depuis {path}")