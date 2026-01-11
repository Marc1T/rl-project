# agents/a2c_agent.py
"""
Advantage Actor-Critic (A2C) pour le PDP.

Implementation synchrone avec:
- Actor (politique): reseau qui output les actions
- Critic (valeur): reseau qui estime V(s)
- Advantage: A(s,a) = R + gamma*V(s') - V(s)

Supporte les actions continues (pas de discretisation).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
import os

from environments.base_pdp_env import BasePDPEnvironment


class ActorCriticNetwork(nn.Module):
    """
    Reseau Actor-Critic avec tete partagee.
    
    - Actor: output mean et std pour une distribution normale (actions continues)
    - Critic: output V(s)
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Args:
            state_dim: Dimension de l'etat
            action_dim: Dimension de l'action
            hidden_dims: Dimensions des couches cachees
            log_std_min/max: Limites pour log_std
        """
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Tronc commun (shared features)
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Tete Actor
        self.actor_hidden = nn.Linear(prev_dim, hidden_dims[-1])
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Linear(hidden_dims[-1], action_dim)
        
        # Tete Critic
        self.critic_hidden = nn.Linear(prev_dim, hidden_dims[-1])
        self.critic_value = nn.Linear(hidden_dims[-1], 1)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids pour stabilite"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Petite initialisation pour les outputs
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: Moyenne des actions
            std: Ecart-type des actions
            value: Estimation de V(s)
        """
        # Features partagees
        shared_features = self.shared(state)
        
        # Actor
        actor_hidden = F.relu(self.actor_hidden(shared_features))
        mean = torch.sigmoid(self.actor_mean(actor_hidden))  # Actions dans [0, 1]
        log_std = self.actor_log_std(actor_hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Critic
        critic_hidden = F.relu(self.critic_hidden(shared_features))
        value = self.critic_value(critic_hidden)
        
        return mean, std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Echantillonne une action.
        
        Returns:
            action: Action echantillonnee
            log_prob: Log-probabilite de l'action
            value: Estimation de V(s)
        """
        mean, std, value = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0, 1)  # Clip dans [0, 1]
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evalue des actions donnees.
        
        Returns:
            log_probs: Log-probabilites des actions
            values: Estimations de V(s)
            entropy: Entropie de la politique
        """
        mean, std, values = self.forward(states)
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy


class A2CAgent:
    """
    Agent Advantage Actor-Critic (A2C).
    
    Caracteristiques:
    - Actions continues (pas de discretisation)
    - Entrainement synchrone (un environnement)
    - Advantage pour reduire la variance
    - Entropie pour encourager l'exploration
    """
    
    def __init__(
        self,
        env: BasePDPEnvironment,
        hidden_dims: List[int] = [256, 256],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            env: Environnement PDP
            hidden_dims: Architecture du reseau
            gamma: Facteur de discount
            gae_lambda: Lambda pour GAE (Generalized Advantage Estimation)
            learning_rate: Taux d'apprentissage
            value_coef: Coefficient pour la loss de valeur
            entropy_coef: Coefficient pour le bonus d'entropie
            max_grad_norm: Gradient clipping
            n_steps: Nombre de steps avant mise a jour
            seed: Graine aleatoire
            device: Device (cpu/cuda)
        """
        self.env = env
        self.config = env.config
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # RNG
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        
        # Dimension de l'action (3 * n_products: regular, overtime, subcontracting)
        self.action_dim = 3 * self.config.n_products
        
        # Dimension de l'etat
        self.state_dim = self._get_state_dim()
        
        # Reseau Actor-Critic
        self.network = ActorCriticNetwork(
            self.state_dim, 
            self.action_dim, 
            hidden_dims
        ).to(self.device)
        
        # Optimiseur
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Statistiques
        self.training_rewards = []
        self.training_losses = []
        self.training_episodes = 0
    
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
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Selectionne une action.
        
        Returns:
            action: Action continue
            log_prob: Log-probabilite
            value: Estimation de V(s)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        return (
            action.cpu().numpy().flatten(),
            log_prob.cpu().item(),
            value.cpu().item()
        )
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les advantages avec GAE (Generalized Advantage Estimation).
        
        Returns:
            advantages: Avantages GAE
            returns: Retours (pour la loss de valeur)
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def update(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        old_log_probs: np.ndarray,
        returns: np.ndarray, 
        advantages: np.ndarray
    ) -> Dict[str, float]:
        """
        Met a jour le reseau avec les donnees collectees.
        
        Returns:
            Dictionnaire avec les losses
        """
        # Convertir en tenseurs
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normaliser les advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Evaluer les actions
        log_probs, values, entropy = self.network.evaluate_actions(states, actions)
        
        # Policy loss (A2C: pas de clipping comme PPO)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """
        Entraine l'agent A2C.
        
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
            
            # Buffers pour n-step
            states_buffer = []
            actions_buffer = []
            rewards_buffer = []
            values_buffer = []
            log_probs_buffer = []
            dones_buffer = []
            
            while not done:
                # Collecter n_steps transitions
                for _ in range(self.n_steps):
                    action, log_prob, value = self.select_action(state)
                    
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    next_state = self._obs_to_state(obs)
                    
                    states_buffer.append(state)
                    actions_buffer.append(action)
                    rewards_buffer.append(reward)
                    values_buffer.append(value)
                    log_probs_buffer.append(log_prob)
                    dones_buffer.append(done)
                    
                    total_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                # Calculer la valeur du prochain etat
                if done:
                    next_value = 0
                else:
                    _, _, next_value = self.select_action(state, deterministic=True)
                
                # Calculer GAE
                advantages, returns = self.compute_gae(
                    rewards_buffer, values_buffer, dones_buffer, next_value
                )
                
                # Mise a jour
                losses = self.update(
                    np.array(states_buffer),
                    np.array(actions_buffer),
                    np.array(log_probs_buffer),
                    returns,
                    advantages
                )
                
                self.training_losses.append(losses['total_loss'])
                
                # Reset buffers
                states_buffer = []
                actions_buffer = []
                rewards_buffer = []
                values_buffer = []
                log_probs_buffer = []
                dones_buffer = []
            
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            self.training_episodes += 1
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                mean_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"Loss: {mean_loss:.4f}")
        
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
                action, _, _ = self.select_action(state, deterministic=True)
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
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_episodes': self.training_episodes,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }, path)
    
    def load(self, path: str):
        """Charge l'agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_episodes = checkpoint['training_episodes']
        self.training_rewards = checkpoint['training_rewards']
        self.training_losses = checkpoint['training_losses']
