# agents/sac_agent.py
"""
Soft Actor-Critic (SAC) pour le PDP.

Implementation avec:
- Maximum Entropy RL
- Twin Q-networks pour reduire la surestimation
- Automatic entropy tuning
- Actions continues natives (pas de discretisation)

SAC est l'algorithme state-of-the-art pour les espaces d'action continus.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import os

from environments.base_pdp_env import BasePDPEnvironment


class ReplayBuffer:
    """Experience Replay Buffer pour SAC"""
    
    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = self.rng.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """
    Politique stochastique gaussienne pour SAC.
    
    Output: mean et log_std pour une distribution normale.
    Utilise le reparameterization trick pour le gradient.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Reseau
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne mean et log_std"""
        features = self.net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Echantillonne une action avec reparameterization trick.
        
        Returns:
            action: Action echantillonnee (apres squashing)
            log_prob: Log-probabilite (avec correction Jacobienne)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Gradient passe a travers
        
        # Squashing avec tanh puis rescale vers [0, 1]
        y_t = torch.tanh(x_t)
        action = (y_t + 1) / 2  # Rescale de [-1, 1] vers [0, 1]
        
        # Log-probabilite avec correction Jacobienne pour tanh
        log_prob = normal.log_prob(x_t)
        # Correction pour le changement de variable (tanh)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Obtient une action (pour l'evaluation)"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
            action = (action + 1) / 2
        else:
            action, _ = self.sample(state)
        
        return action


class TwinQNetwork(nn.Module):
    """
    Twin Q-Networks pour SAC.
    
    Deux reseaux Q independants pour reduire la surestimation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Q1
        q1_layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2
        q2_layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne Q1(s,a) et Q2(s,a)"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Retourne seulement Q1(s,a)"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class SACAgent:
    """
    Agent Soft Actor-Critic (SAC).
    
    Caracteristiques:
    - Maximum entropy RL: maximise reward + entropie
    - Twin Q-networks: reduit la surestimation
    - Automatic entropy tuning: ajuste alpha automatiquement
    - Off-policy: utilise un replay buffer
    """
    
    def __init__(
        self,
        env: BasePDPEnvironment,
        hidden_dims: List[int] = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        learning_starts: int = 1000,
        update_freq: int = 1,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            env: Environnement PDP
            hidden_dims: Architecture des reseaux
            gamma: Facteur de discount
            tau: Coefficient pour soft update du target
            alpha: Temperature d'entropie (si auto_entropy_tuning=False)
            auto_entropy_tuning: Ajuster alpha automatiquement
            target_entropy: Entropie cible (defaut: -dim(action))
            learning_rate: Taux d'apprentissage
            buffer_size: Taille du replay buffer
            batch_size: Taille des batches
            learning_starts: Nombre de steps avant de commencer l'apprentissage
            update_freq: Frequence des mises a jour
            seed: Graine aleatoire
            device: Device (cpu/cuda)
        """
        self.env = env
        self.config = env.config
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_freq = update_freq
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # RNG
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        
        # Dimensions
        self.action_dim = 3 * self.config.n_products
        self.state_dim = self._get_state_dim()
        
        # Networks
        self.policy = GaussianPolicy(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q_networks = TwinQNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_q_networks = TwinQNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_networks.parameters(), lr=learning_rate)
        
        # Entropy tuning
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -self.action_dim
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed)
        
        # Statistiques
        self.training_rewards = []
        self.training_steps = 0
        self.updates = 0
    
    def _get_state_dim(self) -> int:
        """Calcule la dimension de l'etat"""
        obs, _ = self.env.reset()
        state = self._obs_to_state(obs)
        return len(state)
    
    def _obs_to_state(self, observation: Dict) -> np.ndarray:
        """Convertit l'observation dict en vecteur numpy"""
        parts = []
        parts.append(observation['current_stock'].flatten())
        parts.append(observation['future_demands'].flatten())
        if 'current_period' in observation:
            parts.append(observation['current_period'].flatten())
        return np.concatenate(parts).astype(np.float32)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Selectionne une action"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy.get_action(state_tensor, deterministic)
        return action.cpu().numpy().flatten()
    
    def update(self) -> Dict[str, float]:
        """
        Met a jour les reseaux.
        
        Returns:
            Dictionnaire avec les losses
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Echantillonner
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ===== Update Q-networks =====
        with torch.no_grad():
            # Sample next action from policy
            next_actions, next_log_probs = self.policy.sample(next_states)
            
            # Target Q-values (min of twin Q)
            target_q1, target_q2 = self.target_q_networks(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Soft target with entropy
            target_q = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_log_probs)
        
        # Current Q-values
        current_q1, current_q2 = self.q_networks(states, actions)
        
        # Q loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ===== Update Policy =====
        new_actions, log_probs = self.policy.sample(states)
        q1_new, q2_new = self.q_networks(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Policy loss: maximize Q - alpha * log_prob
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ===== Update Alpha (entropy coefficient) =====
        alpha_loss = 0.0
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        
        # ===== Soft update target networks =====
        self._soft_update()
        
        self.updates += 1
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss,
            'alpha': self.alpha,
            'entropy': -log_probs.mean().item()
        }
    
    def _soft_update(self):
        """Soft update des target networks"""
        for param, target_param in zip(self.q_networks.parameters(), self.target_q_networks.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """
        Entraine l'agent SAC.
        
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
                if self.training_steps < self.learning_starts:
                    # Random actions au debut
                    action = self.rng.uniform(0, 1, size=self.action_dim).astype(np.float32)
                else:
                    action = self.select_action(state)
                
                # Executer l'action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self._obs_to_state(obs)
                
                # Stocker la transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Mise a jour
                if self.training_steps >= self.learning_starts and self.training_steps % self.update_freq == 0:
                    losses = self.update()
                    if losses:
                        episode_losses.append(losses.get('q_loss', 0))
                
                self.training_steps += 1
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"Alpha: {self.alpha:.4f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Updates: {self.updates}")
        
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
                action = self.select_action(state, deterministic=True)
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
        save_dict = {
            'policy': self.policy.state_dict(),
            'q_networks': self.q_networks.state_dict(),
            'target_q_networks': self.target_q_networks.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'alpha': self.alpha,
            'training_steps': self.training_steps,
            'updates': self.updates,
            'training_rewards': self.training_rewards
        }
        if self.auto_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Charge l'agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q_networks.load_state_dict(checkpoint['q_networks'])
        self.target_q_networks.load_state_dict(checkpoint['target_q_networks'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.alpha = checkpoint['alpha']
        self.training_steps = checkpoint['training_steps']
        self.updates = checkpoint['updates']
        self.training_rewards = checkpoint['training_rewards']
        
        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
