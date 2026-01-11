# API Reference - Composants

Documentation technique des composants modulaires.

## DemandGenerator

Génère la demande stochastique.

### Classe

```python
class DemandGenerator:
    """
    Générateur de demande avec support de différentes intensités.
    
    Attributes:
        config (PDPConfig): Configuration
        intensity (str): Niveau d'intensité
        rng (np.random.Generator): Générateur aléatoire
    """
```

### Constructeur

```python
def __init__(self, config: PDPConfig):
    """
    Args:
        config: Configuration avec demand_mean, demand_std, demand_intensity
    """
```

### Méthodes

#### generate

```python
def generate(self, period: int) -> np.ndarray:
    """
    Génère la demande pour une période.
    
    Args:
        period: Numéro de la période (0-indexed)
    
    Returns:
        Demande par produit (shape: n_products,)
    
    Example:
        >>> gen = DemandGenerator(config)
        >>> demand = gen.generate(0)
        >>> print(demand)
        array([85.3], dtype=float32)
    """
```

#### generate_forecast

```python
def generate_forecast(self, current_period: int, horizon: int) -> np.ndarray:
    """
    Génère des prévisions de demande.
    
    Args:
        current_period: Période actuelle
        horizon: Nombre de périodes à prévoir
    
    Returns:
        Prévisions (shape: horizon, n_products)
    """
```

### Niveaux d'Intensité

| Niveau | `multiplier` | `spike_prob` | `spike_range` |
|--------|--------------|--------------|---------------|
| `low` | 0.75 | 0.15 | (1.1, 1.2) |
| `medium` | 0.90 | 0.25 | (1.2, 1.35) |
| `high` | 1.05 | 0.35 | (1.3, 1.5) |
| `extreme` | 1.20 | 0.45 | (1.4, 1.6) |

---

## CostCalculator

Calcule les différents coûts.

### Classe

```python
class CostCalculator:
    """
    Calculateur de coûts de production, stockage et rupture.
    """
```

### Méthodes

#### compute_production_cost

```python
def compute_production_cost(
    self, 
    regular: float, 
    overtime: float, 
    subcontracting: float,
    product_idx: int = 0
) -> float:
    """
    Calcule le coût de production.
    
    Args:
        regular: Quantité production régulière
        overtime: Quantité heures supplémentaires
        subcontracting: Quantité sous-traitance
        product_idx: Index du produit
    
    Returns:
        Coût total de production
    """
```

#### compute_inventory_cost

```python
def compute_inventory_cost(
    self, 
    stock_level: float, 
    product_idx: int = 0
) -> float:
    """
    Calcule le coût de stockage.
    
    Args:
        stock_level: Niveau de stock (peut être négatif)
        product_idx: Index du produit
    
    Returns:
        Coût de stockage (0 si stock <= 0)
    """
```

#### compute_shortage_cost

```python
def compute_shortage_cost(
    self, 
    shortage: float, 
    product_idx: int = 0
) -> float:
    """
    Calcule le coût de rupture.
    
    Args:
        shortage: Quantité en rupture
        product_idx: Index du produit
    
    Returns:
        Coût de rupture
    """
```

#### compute_all_costs

```python
def compute_all_costs(
    self,
    production: Dict[str, float],
    stock: float,
    shortage: float,
    product_idx: int = 0
) -> Dict[str, float]:
    """
    Calcule tous les coûts.
    
    Args:
        production: {'regular': x, 'overtime': y, 'subcontracting': z}
        stock: Niveau de stock final
        shortage: Quantité en rupture
        product_idx: Index du produit
    
    Returns:
        {'production_cost': x, 'inventory_cost': y, 'shortage_cost': z}
    """
```

---

## RewardCalculator

Calcule la récompense RL.

### Classe

```python
class RewardCalculator:
    """
    Calculateur de récompense pour l'agent RL.
    
    La récompense combine:
    - Pénalité de coût (normalisée)
    - Bonus de service level
    - Bonus de stock de sécurité
    """
```

### Méthodes

#### compute_reward

```python
def compute_reward(
    self,
    costs: Dict[str, float],
    service_level: float,
    stock_level: float
) -> float:
    """
    Calcule la récompense totale.
    
    Args:
        costs: Dictionnaire des coûts
        service_level: Ratio de satisfaction demande [0, 1]
        stock_level: Niveau de stock actuel
    
    Returns:
        Récompense (float)
    
    Formula:
        R = -cost_norm + α*service + β*safety_bonus
    """
```

---

## ObservationBuilder

Construit les observations.

### Classe

```python
class ObservationBuilder:
    """
    Constructeur d'observations pour l'agent.
    """
```

### Méthodes

#### build

```python
def build(self, env_state: Dict) -> Dict[str, np.ndarray]:
    """
    Construit l'observation à partir de l'état.
    
    Args:
        env_state: État interne de l'environnement
    
    Returns:
        Observation formatée pour l'agent
    """
```

#### get_observation_space

```python
def get_observation_space(self) -> gymnasium.spaces.Dict:
    """
    Retourne l'espace d'observation.
    
    Returns:
        Espace Gymnasium Dict
    """
```

---

## ActionValidator

Valide et transforme les actions.

### Classe

```python
class ActionValidator:
    """
    Validateur d'actions avec contraintes de capacité.
    """
```

### Méthodes

#### validate

```python
def validate(self, action: np.ndarray) -> np.ndarray:
    """
    Valide et contraint l'action.
    
    Args:
        action: Action brute de l'agent [0,1]^3
    
    Returns:
        Action validée et clippée
    """
```

#### to_production

```python
def to_production(self, action: np.ndarray) -> Dict[str, float]:
    """
    Convertit l'action en quantités de production.
    
    Args:
        action: Action normalisée [0,1]^3
    
    Returns:
        {'regular': x, 'overtime': y, 'subcontracting': z}
    """
```
