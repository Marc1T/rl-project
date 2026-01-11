# agents/tabular_agents.py
"""
Agents tabulaires classiques pour le PDP.
Ces agents necessitent une discretisation de l'espace d'action continu.

Agents implementes:
- MonteCarloAgent: Methode Monte Carlo first-visit
- QLearningAgent: Q-Learning (off-policy TD)
- SARSAAgent: SARSA (on-policy TD)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import pickle
import os

from environments.base_pdp_env import BasePDPEnvironment


class ActionDiscretizer:
    """
    Discretise l'espace d'action continu en actions discretes.
    
    L'espace d'action PDP est [0,1]^3 (regular, overtime, subcontracting ratios).
    On discretise chaque dimension en n_bins niveaux.
    """
    
    def __init__(self, n_bins: int = 5, n_products: int = 1):
        """
        Args:
            n_bins: Nombre de niveaux de discretisation par type d'action
            n_products: Nombre de produits
        """
        self.n_bins = n_bins
        self.n_products = n_products
        
        # Creer les valeurs discretes possibles pour chaque ratio
        self.bin_values = np.linspace(0, 1, n_bins)
        
        # Nombre total d'actions discretes = n_bins^3 par produit
        # Pour simplifier, on considere le meme ratio pour tous les produits
        self.n_actions = n_bins ** 3
        
        # Pre-calculer toutes les actions possibles
        self._build_action_mapping()
    
    def _build_action_mapping(self):
        """Construit le mapping index -> action continue"""
        self.action_map = {}
        idx = 0
        for r in self.bin_values:
            for o in self.bin_values:
                for s in self.bin_values:
                    self.action_map[idx] = np.array([r, o, s], dtype=np.float32)
                    idx += 1
    
    def discrete_to_continuous(self, action_idx: int) -> np.ndarray:
        """Convertit un index d'action en action continue"""
        base_action = self.action_map[action_idx]
        # Repliquer pour tous les produits
        return np.tile(base_action, self.n_products)
    
    def continuous_to_discrete(self, action: np.ndarray) -> int:
        """Convertit une action continue en index discret (plus proche)"""
        # Prendre les 3 premieres valeurs (pour le premier produit)
        r, o, s = action[:3]
        
        # Trouver les bins les plus proches
        r_bin = np.argmin(np.abs(self.bin_values - r))
        o_bin = np.argmin(np.abs(self.bin_values - o))
        s_bin = np.argmin(np.abs(self.bin_values - s))
        
        # Calculer l'index
        return r_bin * self.n_bins**2 + o_bin * self.n_bins + s_bin


class StateDiscretizer:
    """
    Discretise l'espace d'observation pour les methodes tabulaires.
    
    Utilise une discretisation simple basee sur:
    - Niveau de stock (bins)
    - Niveau de demande future (bins)
    - Periode courante
    """
    
    def __init__(self, config, n_stock_bins: int = 10, n_demand_bins: int = 5):
        """
        Args:
            config: Configuration de l'environnement
            n_stock_bins: Nombre de bins pour le stock
            n_demand_bins: Nombre de bins pour la demande
        """
        self.config = config
        self.n_stock_bins = n_stock_bins
        self.n_demand_bins = n_demand_bins
        
        # Limites pour la discretisation
        self.stock_min = -config.max_stock[0] * 0.5
        self.stock_max = config.max_stock[0]
        self.demand_max = config.regular_capacity[0] * 2
        
    def discretize(self, observation: Dict) -> Tuple:
        """
        Convertit une observation en tuple discret (hashable pour Q-table).
        
        Returns:
            Tuple (stock_bin, demand_bin, period)
        """
        # Stock
        stock = observation['current_stock'][0]
        stock_normalized = (stock - self.stock_min) / (self.stock_max - self.stock_min)
        stock_bin = int(np.clip(stock_normalized * self.n_stock_bins, 0, self.n_stock_bins - 1))
        
        # Demande moyenne future
        future_demands = observation['future_demands']
        avg_demand = np.mean(future_demands)
        demand_normalized = avg_demand / self.demand_max
        demand_bin = int(np.clip(demand_normalized * self.n_demand_bins, 0, self.n_demand_bins - 1))
        
        # Periode
        period = int(observation.get('current_period', [0])[0])
        
        return (stock_bin, demand_bin, period)


class BaseTabularAgent(ABC):
    """Classe de base pour les agents tabulaires"""
    
    def __init__(
        self, 
        env: BasePDPEnvironment,
        n_action_bins: int = 5,
        n_stock_bins: int = 10,
        n_demand_bins: int = 5,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Args:
            env: Environnement PDP
            n_action_bins: Discretisation des actions
            n_stock_bins: Discretisation du stock
            n_demand_bins: Discretisation de la demande
            gamma: Facteur de discount
            epsilon: Taux d'exploration initial
            epsilon_min: Taux d'exploration minimum
            epsilon_decay: Decay de l'exploration
            seed: Graine aleatoire
        """
        self.env = env
        self.config = env.config
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # RNG moderne
        self.rng = np.random.default_rng(seed)
        
        # Discretiseurs
        self.action_discretizer = ActionDiscretizer(n_action_bins, self.config.n_products)
        self.state_discretizer = StateDiscretizer(self.config, n_stock_bins, n_demand_bins)
        
        self.n_actions = self.action_discretizer.n_actions
        
        # Q-table: defaultdict pour gerer les etats non visites
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # Statistiques d'entrainement
        self.training_rewards = []
        self.training_episodes = 0
    
    def get_state(self, observation: Dict) -> Tuple:
        """Discretise l'observation en etat"""
        return self.state_discretizer.discretize(observation)
    
    def select_action(self, state: Tuple, training: bool = True) -> int:
        """
        Selectionne une action avec epsilon-greedy.
        
        Args:
            state: Etat discret
            training: Si True, utilise epsilon-greedy; sinon, greedy
            
        Returns:
            Index de l'action discrete
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))
    
    def get_continuous_action(self, action_idx: int) -> np.ndarray:
        """Convertit l'action discrete en action continue pour l'environnement"""
        return self.action_discretizer.discrete_to_continuous(action_idx)
    
    def decay_epsilon(self):
        """Reduit epsilon apres chaque episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    @abstractmethod
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """Entraine l'agent sur n_episodes"""
        pass
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evalue l'agent sans exploration"""
        rewards = []
        service_levels = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            state = self.get_state(obs)
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
                
                state = self.get_state(obs)
            
            rewards.append(total_reward)
            service_levels.append(np.mean(episode_service))
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_service': np.mean(service_levels),
            'n_states_visited': len(self.q_table)
        }
    
    def save(self, path: str):
        """Sauvegarde l'agent"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes,
            'training_rewards': self.training_rewards
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Charge l'agent"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.epsilon = data['epsilon']
        self.training_episodes = data['training_episodes']
        self.training_rewards = data['training_rewards']


class MonteCarloAgent(BaseTabularAgent):
    """
    Agent Monte Carlo First-Visit.
    
    Methode:
    - Joue des episodes complets
    - Met a jour Q(s,a) avec le retour moyen observe
    - Utilise first-visit: seule la premiere visite d'une paire (s,a) compte
    """
    
    def __init__(self, env: BasePDPEnvironment, **kwargs):
        super().__init__(env, **kwargs)
        # Compteur de visites pour la moyenne incrementale
        self.returns_count = defaultdict(lambda: np.zeros(self.n_actions))
    
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """
        Entraine l'agent avec Monte Carlo first-visit.
        
        Args:
            n_episodes: Nombre d'episodes d'entrainement
            verbose: Afficher la progression
            
        Returns:
            Liste des rewards par episode
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            # Jouer un episode complet
            trajectory = []  # [(state, action, reward), ...]
            obs, _ = self.env.reset()
            state = self.get_state(obs)
            done = False
            total_reward = 0
            
            while not done:
                action_idx = self.select_action(state, training=True)
                action = self.get_continuous_action(action_idx)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                trajectory.append((state, action_idx, reward))
                total_reward += reward
                
                state = self.get_state(obs)
            
            # Calculer les retours et mettre a jour Q
            G = 0
            visited = set()
            
            for t in reversed(range(len(trajectory))):
                state_t, action_t, reward_t = trajectory[t]
                G = self.gamma * G + reward_t
                
                # First-visit: ne compter que la premiere visite
                if (state_t, action_t) not in visited:
                    visited.add((state_t, action_t))
                    
                    # Mise a jour incrementale de la moyenne
                    self.returns_count[state_t][action_t] += 1
                    n = self.returns_count[state_t][action_t]
                    
                    # Q(s,a) = Q(s,a) + (G - Q(s,a)) / n
                    self.q_table[state_t][action_t] += (G - self.q_table[state_t][action_t]) / n
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            self.training_episodes += 1
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward moyen (100 derniers): {mean_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Etats visites: {len(self.q_table)}")
        
        return episode_rewards


class QLearningAgent(BaseTabularAgent):
    """
    Agent Q-Learning (off-policy TD).
    
    Methode:
    - Met a jour Q(s,a) apres chaque step avec la regle TD
    - Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    - Off-policy: utilise max pour la cible, epsilon-greedy pour le comportement
    """
    
    def __init__(
        self, 
        env: BasePDPEnvironment, 
        alpha: float = 0.1,
        **kwargs
    ):
        """
        Args:
            alpha: Taux d'apprentissage
        """
        super().__init__(env, **kwargs)
        self.alpha = alpha
    
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """
        Entraine l'agent avec Q-Learning.
        
        Args:
            n_episodes: Nombre d'episodes d'entrainement
            verbose: Afficher la progression
            
        Returns:
            Liste des rewards par episode
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            state = self.get_state(obs)
            done = False
            total_reward = 0
            
            while not done:
                # Selection de l'action (epsilon-greedy)
                action_idx = self.select_action(state, training=True)
                action = self.get_continuous_action(action_idx)
                
                # Executer l'action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self.get_state(obs)
                
                # Mise a jour Q-Learning (off-policy)
                # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * np.max(self.q_table[next_state])
                
                td_error = td_target - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * td_error
                
                total_reward += reward
                state = next_state
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            self.training_episodes += 1
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward moyen (100 derniers): {mean_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Etats visites: {len(self.q_table)}")
        
        return episode_rewards


class SARSAAgent(BaseTabularAgent):
    """
    Agent SARSA (on-policy TD).
    
    Methode:
    - Met a jour Q(s,a) apres chaque step avec la regle TD
    - Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
    - On-policy: utilise l'action reellement choisie pour la cible
    """
    
    def __init__(
        self, 
        env: BasePDPEnvironment, 
        alpha: float = 0.1,
        **kwargs
    ):
        """
        Args:
            alpha: Taux d'apprentissage
        """
        super().__init__(env, **kwargs)
        self.alpha = alpha
    
    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """
        Entraine l'agent avec SARSA.
        
        Args:
            n_episodes: Nombre d'episodes d'entrainement
            verbose: Afficher la progression
            
        Returns:
            Liste des rewards par episode
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            state = self.get_state(obs)
            action_idx = self.select_action(state, training=True)
            
            done = False
            total_reward = 0
            
            while not done:
                # Executer l'action
                action = self.get_continuous_action(action_idx)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self.get_state(obs)
                
                # Choisir la prochaine action (on-policy)
                next_action_idx = self.select_action(next_state, training=True)
                
                # Mise a jour SARSA (on-policy)
                # Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * self.q_table[next_state][next_action_idx]
                
                td_error = td_target - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * td_error
                
                total_reward += reward
                state = next_state
                action_idx = next_action_idx
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            self.training_episodes += 1
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward moyen (100 derniers): {mean_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Etats visites: {len(self.q_table)}")
        
        return episode_rewards


# Registre des agents tabulaires
TABULAR_AGENTS = {
    'monte_carlo': MonteCarloAgent,
    'q_learning': QLearningAgent,
    'sarsa': SARSAAgent
}
