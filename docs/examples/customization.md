# Personnalisation

Guide pour étendre et personnaliser RLPlanif selon vos besoins.

## 🎨 Personnalisation de l'Environnement

### Créer un Environnement Custom

```python
from environments.base_pdp_env import BasePDPEnv
from environments.env_registry import EnvironmentRegistry
import numpy as np

class MyCustomEnv(BasePDPEnv):
    """Environnement personnalisé avec fonctionnalités spécifiques"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Ajouter des attributs personnalisés
        self.quality_factor = 0.98
        self.machine_efficiency = np.ones(config.n_products)
    
    def _compute_production(self, action):
        """Override avec logique personnalisée"""
        base_production = super()._compute_production(action)
        
        # Appliquer le facteur de qualité
        effective_production = base_production * self.quality_factor
        
        # Simuler les pannes aléatoires
        if np.random.random() < 0.05:  # 5% de chance de panne
            effective_production *= 0.8
        
        return effective_production
    
    def _build_observation(self):
        """Enrichir l'observation"""
        base_obs = super()._build_observation()
        
        # Ajouter des informations supplémentaires
        base_obs['machine_efficiency'] = self.machine_efficiency.astype(np.float32)
        
        return base_obs

# Enregistrer l'environnement
EnvironmentRegistry.register('custom', MyCustomEnv)

# Utilisation
env = EnvironmentRegistry.create('custom', config)
```

### Modifier l'Espace d'Observation

```python
from gymnasium import spaces

class ExtendedObsEnv(BasePDPEnv):
    def __init__(self, config):
        super().__init__(config)
        
        # Étendre l'espace d'observation
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,  # Garder l'existant
            'weather_factor': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'market_trend': spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
```

## 🧩 Personnalisation des Composants

### Générateur de Demande Custom

```python
from components.demand_generators import DemandGenerator

class SeasonalDemandGenerator(DemandGenerator):
    """Générateur avec forte saisonnalité"""
    
    def __init__(self, config):
        super().__init__(config)
        self.seasonality_amplitude = 0.3
        self.peak_periods = [3, 4, 11, 12]  # Mars, Avril, Nov, Dec
    
    def generate(self, period: int) -> np.ndarray:
        base_demand = super().generate(period)
        
        # Ajouter la saisonnalité
        month = (period % 12) + 1
        if month in self.peak_periods:
            seasonal_factor = 1 + self.seasonality_amplitude
        else:
            seasonal_factor = 1 - self.seasonality_amplitude / 2
        
        return base_demand * seasonal_factor
```

### Calculateur de Récompense Custom

```python
from components.reward_calculators import RewardCalculator

class MultiObjectiveReward(RewardCalculator):
    """Récompense multi-objectifs avec pondération"""
    
    def __init__(self, config, weights=None):
        super().__init__(config)
        self.weights = weights or {
            'cost': 0.4,
            'service': 0.3,
            'stability': 0.2,
            'sustainability': 0.1
        }
        self.previous_production = None
    
    def compute_reward(self, costs, service_level, stock_level, production):
        # Composante coût
        cost_reward = -sum(costs.values()) / 1000
        
        # Composante service
        service_reward = service_level * 10
        
        # Composante stabilité (pénaliser les variations)
        if self.previous_production is not None:
            variation = abs(production - self.previous_production)
            stability_reward = -variation / 100
        else:
            stability_reward = 0
        self.previous_production = production
        
        # Composante durabilité (favoriser production régulière)
        regular_ratio = costs.get('regular', 0) / (sum(costs.values()) + 1e-6)
        sustainability_reward = regular_ratio * 5
        
        # Combiner avec les poids
        total_reward = (
            self.weights['cost'] * cost_reward +
            self.weights['service'] * service_reward +
            self.weights['stability'] * stability_reward +
            self.weights['sustainability'] * sustainability_reward
        )
        
        return total_reward
```

## 🤖 Personnalisation de l'Agent

### Modifier l'Architecture du Réseau

```python
from stable_baselines3 import PPO
import torch.nn as nn

# Architecture personnalisée
policy_kwargs = {
    'net_arch': {
        'pi': [256, 256, 128],  # Actor plus profond
        'vf': [256, 256, 128]   # Critic plus profond
    },
    'activation_fn': nn.LeakyReLU,
    'ortho_init': True,
}

model = PPO(
    'MultiInputPolicy',
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,
    verbose=1
)
```

### Learning Rate Scheduler

```python
from stable_baselines3.common.callbacks import BaseCallback

class LRSchedulerCallback(BaseCallback):
    """Decay du learning rate"""
    
    def __init__(self, initial_lr=3e-4, decay_rate=0.99, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def _on_step(self) -> bool:
        # Decay toutes les 10000 steps
        if self.num_timesteps % 10000 == 0:
            new_lr = self.initial_lr * (self.decay_rate ** (self.num_timesteps / 10000))
            self.model.learning_rate = new_lr
            if self.verbose:
                print(f"LR updated to {new_lr:.6f}")
        return True
```

## 📊 Personnalisation des Visualisations

### Graphiques Custom

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_custom_dashboard(metrics: list, title: str):
    """Dashboard personnalisé"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Production vs Demande',
            'Évolution des Stocks',
            'Répartition des Coûts',
            'Service Level'
        )
    )
    
    periods = list(range(1, len(metrics) + 1))
    
    # Production vs Demande
    fig.add_trace(
        go.Bar(x=periods, y=[m['total_production'] for m in metrics], name='Production'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=periods, y=[m['raw_metrics']['current_demand'][0] for m in metrics], 
                   name='Demande', mode='lines+markers'),
        row=1, col=1
    )
    
    # Stocks
    fig.add_trace(
        go.Scatter(x=periods, y=[m['inventory_level'][0] for m in metrics],
                   name='Stock', fill='tozeroy'),
        row=1, col=2
    )
    
    # Coûts (pie chart simulé avec bar)
    total_costs = {
        'Production': sum(m['costs']['production_cost'] for m in metrics),
        'Stockage': sum(m['costs']['inventory_cost'] for m in metrics),
        'Rupture': sum(m['costs']['shortage_cost'] for m in metrics),
    }
    fig.add_trace(
        go.Bar(x=list(total_costs.keys()), y=list(total_costs.values()), name='Coûts'),
        row=2, col=1
    )
    
    # Service Level
    fig.add_trace(
        go.Scatter(x=periods, y=[m['demand_fulfillment'] * 100 for m in metrics],
                   name='Service %', mode='lines+markers'),
        row=2, col=2
    )
    fig.add_hline(y=95, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(height=800, title_text=title, showlegend=True)
    
    return fig
```

## 🔧 Personnalisation de la Configuration

### Configuration Dynamique

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class DynamicPDPConfig:
    """Configuration avec paramètres dynamiques"""
    
    n_products: int = 1
    horizon: int = 12
    
    # Capacités dynamiques (peuvent varier par période)
    capacity_schedule: Callable[[int], float] = None
    
    # Demande saisonnière
    demand_seasonality: list = field(default_factory=lambda: [1.0] * 12)
    
    def get_capacity(self, period: int) -> float:
        if self.capacity_schedule:
            return self.capacity_schedule(period)
        return 100  # Défaut
    
    def get_seasonal_demand(self, base_demand: float, period: int) -> float:
        month = period % 12
        return base_demand * self.demand_seasonality[month]

# Utilisation
config = DynamicPDPConfig(
    demand_seasonality=[0.8, 0.8, 1.0, 1.2, 1.3, 1.1, 0.9, 0.9, 1.0, 1.1, 1.2, 1.4]
)
```

## 📁 Structure pour Extensions

```
RLPlanif/
├── extensions/                 # Vos extensions
│   ├── custom_envs/
│   │   └── my_env.py
│   ├── custom_components/
│   │   └── my_demand_gen.py
│   └── custom_callbacks/
│       └── my_callback.py
└── ...
```

### Exemple de Module d'Extension

```python
# extensions/__init__.py
from .custom_envs.my_env import MyCustomEnv
from .custom_components.my_demand_gen import SeasonalDemandGenerator

__all__ = ['MyCustomEnv', 'SeasonalDemandGenerator']
```

## Prochaine Étape

➡️ [Guide de Contribution](../contributing.md)
