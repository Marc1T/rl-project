# agents/dqn_agent.py
"""
Deep Q-Network (DQN) pour le PDP.

Implementation avec:
- Experience Replay Buffer
- Target Network (Double DQN optionnel)
- Discretisation des actions (comme les agents tabulaires)

Ameliorations possibles:
- Dueling DQN
- Prioritized Experience Replay
- Noisy Networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import os

from environments.base_pdp_env import BasePDPEnvironment
from .tabular_agents import ActionDiscretizer


class ReplayBuffer:
    """
    Experience Replay Buffer pour DQN.
    
    Stocke les transitions (s, a, r, s', done) et permet un echantillonnage aleatoire.
    """
    
    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Args:
            capacity: Taille maximale du buffer
            seed: Graine aleatoire
        """
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Ajoute une transition au buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Echantillonne un batch de transitions"""
        batch = self.rng.sample(list(self.buffer), batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Reseau de neurones pour approximer Q(s, a).
    
    Architecture: MLP avec couches cachees configurables.
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [128, 128]):
        """
        Args:
            state_dim: Dimension de l'etat
            n_actions: Nombre d'actions discretes
            hidden_dims: Dimensions des couches cachees
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: retourne Q(s, a) pour toutes les actions"""
        return self.network(x)


class DQNAgent:
    """
    Agent Deep Q-Network.
    
    Caracteristiques:
    - Experience Replay pour casser les correlations temporelles
    - Target Network pour stabiliser l'apprentissage
    - Double DQN (optionnel) pour reduire la surestimation
    """
    
    def __init__(
        self,
        env: BasePDPEnvironment,
        n_action_bins: int = 5,
        hidden_dims: List[int] = [128, 128],
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        double_dqn: bool = True,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            env: Environnement PDP
            n_action_bins: Discretisation des actions
            hidden_dims: Architecture du reseau
            gamma: Facteur de discount
            epsilon: Taux d'exploration initial
            epsilon_min: Taux d'exploration minimum
            epsilon_decay: Decay de l'exploration
            learning_rate: Taux d'apprentissage
            buffer_size: Taille du replay buffer
            batch_size: Taille des batches
            target_update_freq: Frequence de mise a jour du target network
            double_dqn: Utiliser Double DQN
            seed: Graine aleatoire
            device: Device (cpu/cuda)
        """
        self.env = env
        self.config = env.config
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # RNG
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        
        # Discretiseur d'actions
        self.action_discretizer = ActionDiscretizer(n_action_bins, self.config.n_products)
        self.n_actions = self.action_discretizer.n_actions
        
        # Dimension de l'etat (flatten l'observation)
        self.state_dim = self._get_state_dim()
        
        # Reseaux
        self.q_network = QNetwork(self.state_dim, self.n_actions, hidden_dims).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.n_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed)
        
        # Statistiques
        self.training_rewards = []
        self.training_losses = []
        self.training_steps = 0
    
    def _get_state_dim(self) -> int:
        """Calcule la dimension de l'etat a partir d'une observation"""
        obs, _ = self.env.reset()
        state = self._obs_to_state(obs)
        return len(state)
    
    def _obs_to_state(self, observation: Dict) -> np.ndarray:
        """Convertit l'observation dict en vecteur numpy"""
        parts = []
        
        # Stock
        parts.append(observation['current_stock'].flatten())
        
        # Demandes futures
        parts.append(observation['future_demands'].flatten())
        
        # Periode (si presente)
        if 'current_period' in observation:
            parts.append(observation['current_period'].flatten())
        
        return np.concatenate(parts).astype(np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selectionne une action avec epsilon-greedy.
        
        Args:
            state: Etat (vecteur numpy)
            training: Si True, utilise epsilon-greedy
            
        Returns:
            Index de l'action
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax(dim=1).item())
    
    def get_continuous_action(self, action_idx: int) -> np.ndarray:
        """Convertit l'action discrete en action continue"""
        return self.action_discretizer.discrete_to_continuous(action_idx)
    
    def update(self) -> Optional[float]:
        """
        Met a jour le Q-network avec un batch du replay buffer.
        
        Returns:
            Loss si mise a jour effectuee, None sinon
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Echantillonner un batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convertir en tenseurs
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values actuelles
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-values cibles
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: utiliser le Q-network pour selectionner l'action,
                # et le target network pour evaluer
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # DQN standard: max sur le target network
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss et backprop
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabilite
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copie les poids du Q-network vers le target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Reduit epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """
        Entraine l'agent DQN.
        
        Args:
            n_episodes: Nombre d'episodes
            verbose: Afficher la progression
            
        Returns:
            Liste des rewards par episode
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            state = self._obs_to_state(obs)
            done = False
            total_reward = 0
            episode_losses = []
            
            while not done:
                # Selection de l'action
                action_idx = self.select_action(state, training=True)
                action = self.get_continuous_action(action_idx)
                
                # Executer l'action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self._obs_to_state(obs)
                
                # Stocker la transition
                self.replay_buffer.push(state, action_idx, reward, next_state, done)
                
                # Mise a jour
                loss = self.update()
                if loss is not None:
                    episode_losses.append(loss)
                
                self.training_steps += 1
                
                # Mise a jour du target network
                if self.training_steps % self.target_update_freq == 0:
                    self.update_target_network()
                
                total_reward += reward
                state = next_state
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            
            if episode_losses:
                self.training_losses.append(np.mean(episode_losses))
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                mean_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"Loss: {mean_loss:.4f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Buffer: {len(self.replay_buffer)}")
        
        return episode_rewards
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evalue l'agent sans exploration"""
        rewards = []
        service_levels = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            state = self._obs_to_state(obs)
            done = False
            total_reward = 0
            episode_service = []
            
            while not done:
                action_idx = self.select_action(state, training=False)
                action = self.get_continuous_action(action_idx)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                done = terminated or truncated
                total_reward += reward
                episode_service.append(info['demand_fulfillment'])
                
                state = self._obs_to_state(obs)
            
            rewards.append(total_reward)
            service_levels.append(np.mean(episode_service))
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_service': np.mean(service_levels)
        }
    
    def save(self, path: str):
        """Sauvegarde l'agent"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }, path)
    
    def load(self, path: str):
        """Charge l'agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.training_rewards = checkpoint['training_rewards']
        self.training_losses = checkpoint['training_losses']
