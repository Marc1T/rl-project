# API Reference - Environnements

Documentation technique des environnements Gymnasium.

## BasePDPEnv

Environnement de base pour le Plan Directeur de Production.

### Classe

```python
class BasePDPEnv(gymnasium.Env):
    """
    Environnement Gymnasium pour l'optimisation du PDP.
    
    Attributes:
        config (PDPConfig): Configuration de l'environnement
        observation_space (Dict): Espace d'observation
        action_space (Box): Espace d'action continu [0,1]^3
        current_period (int): Période actuelle
        stocks (np.ndarray): Niveaux de stock par produit
    """
```

### Constructeur

```python
def __init__(self, config: PDPConfig):
    """
    Initialise l'environnement.
    
    Args:
        config: Configuration PDPConfig avec tous les paramètres
    
    Example:
        >>> config = PDPConfig(n_products=1, horizon=12)
        >>> env = BasePDPEnv(config)
    """
```

### Méthodes

#### reset

```python
def reset(
    self, 
    seed: Optional[int] = None,
    options: Optional[dict] = None
) -> Tuple[Dict, Dict]:
    """
    Réinitialise l'environnement.
    
    Args:
        seed: Graine pour le générateur aléatoire
        options: Options additionnelles (non utilisé)
    
    Returns:
        observation: Dict contenant l'état initial
        info: Dict vide
    
    Example:
        >>> obs, info = env.reset(seed=42)
        >>> print(obs['stocks'])
        array([50.], dtype=float32)
    """
```

#### step

```python
def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
    """
    Exécute une action dans l'environnement.
    
    Args:
        action: Array [regular, overtime, subcontracting] dans [0,1]^3
    
    Returns:
        observation: Nouvel état
        reward: Récompense (float)
        terminated: True si horizon atteint
        truncated: False (non utilisé)
        info: Dict avec métriques détaillées
    
    Example:
        >>> action = np.array([0.8, 0.2, 0.0])
        >>> obs, reward, done, truncated, info = env.step(action)
    """
```

### Espaces

#### Observation Space

```python
observation_space = Dict({
    'stocks': Box(
        low=0, 
        high=max_stock, 
        shape=(n_products,), 
        dtype=np.float32
    ),
    'current_demand': Box(
        low=0, 
        high=max_demand, 
        shape=(n_products,), 
        dtype=np.float32
    ),
    'current_period': Box(
        low=0, 
        high=horizon, 
        shape=(1,), 
        dtype=np.float32
    ),
})
```

#### Action Space

```python
action_space = Box(
    low=0.0,
    high=1.0,
    shape=(3,),  # [regular_ratio, overtime_ratio, subcontracting_ratio]
    dtype=np.float32
)
```

---

## StrategicPDPEnv

Environnement enrichi avec observations étendues.

### Classe

```python
class StrategicPDPEnv(BasePDPEnv):
    """
    Environnement stratégique avec informations contextuelles.
    
    Extends BasePDPEnv avec:
    - Prévisions de demande
    - Historique de production
    - Métriques détaillées
    
    Attributes:
        demand_generator (DemandGenerator): Générateur de demande
        cost_calculator (CostCalculator): Calculateur de coûts
        reward_calculator (RewardCalculator): Calculateur de récompense
    """
```

### Observation Enrichie

```python
observation_space = Dict({
    # État de base
    'stocks': Box(...),
    'current_demand': Box(...),
    'current_period': Box(...),
    
    # Informations stratégiques
    'demand_forecast': Box(
        low=0, 
        high=max_demand * 2,
        shape=(forecast_horizon, n_products),
        dtype=np.float32
    ),
    'remaining_periods': Box(
        low=0,
        high=horizon,
        shape=(1,),
        dtype=np.float32
    ),
    'capacity_utilization': Box(
        low=0,
        high=1,
        shape=(3,),  # [regular, overtime, subcontracting]
        dtype=np.float32
    ),
})
```

---

## EnvironmentRegistry

Factory pour créer des environnements.

### Méthodes de Classe

#### create

```python
@classmethod
def create(cls, env_type: str, config: PDPConfig) -> gymnasium.Env:
    """
    Crée un environnement du type spécifié.
    
    Args:
        env_type: Type d'environnement ('base', 'strategic')
        config: Configuration PDPConfig
    
    Returns:
        Instance de l'environnement
    
    Raises:
        ValueError: Si env_type inconnu
    
    Example:
        >>> env = EnvironmentRegistry.create('strategic', config)
    """
```

#### register

```python
@classmethod
def register(cls, name: str, env_class: Type[gymnasium.Env]):
    """
    Enregistre un nouveau type d'environnement.
    
    Args:
        name: Nom pour l'environnement
        env_class: Classe de l'environnement
    
    Example:
        >>> EnvironmentRegistry.register('custom', MyCustomEnv)
    """
```

#### list_available

```python
@classmethod
def list_available(cls) -> List[str]:
    """
    Liste les environnements disponibles.
    
    Returns:
        Liste des noms d'environnements
    
    Example:
        >>> print(EnvironmentRegistry.list_available())
        ['base', 'strategic', 'custom']
    """
```

---

## Info Dictionary

Structure du dictionnaire `info` retourné par `step()`:

```python
info = {
    # Métriques de production
    'total_production': float,          # Production totale
    'regular_production': float,        # Production régulière
    'overtime_production': float,       # Production en HS
    'subcontracting_production': float, # Sous-traitance
    
    # État
    'inventory_level': np.ndarray,      # Stock par produit
    'demand_fulfillment': float,        # Ratio demande satisfaite [0,1]
    
    # Coûts
    'costs': {
        'production_cost': float,       # Coût de production
        'inventory_cost': float,        # Coût de stockage
        'shortage_cost': float,         # Coût de rupture
    },
    
    # Métriques brutes
    'raw_metrics': {
        'current_demand': np.ndarray,   # Demande de la période
        'stock_before_prod': np.ndarray,# Stock avant production
        'stock_after_prod': np.ndarray, # Stock après production
        'demand_satisfied': np.ndarray, # Demande satisfaite
        'shortage': np.ndarray,         # Quantité en rupture
    }
}
```
