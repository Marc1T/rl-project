# API Reference - Configuration

Documentation technique des classes de configuration.

## PDPConfig

Configuration principale de l'environnement.

### Classe

```python
@dataclass
class PDPConfig:
    """
    Configuration pour l'environnement PDP.
    
    Tous les paramètres de liste ont une valeur par produit.
    Pour un seul produit, utiliser des listes à un élément.
    """
```

### Attributs

#### Dimensions

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `n_products` | `int` | `1` | Nombre de produits |
| `horizon` | `int` | `12` | Nombre de périodes |

#### Capacités

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `regular_capacity` | `List[float]` | `[100.0]` | Capacité production régulière |
| `overtime_capacity` | `List[float]` | `[30.0]` | Capacité heures supplémentaires |
| `subcontracting_capacity` | `List[float]` | `[50.0]` | Capacité sous-traitance |

#### Coûts de Production

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `regular_cost` | `List[float]` | `[10.0]` | Coût unitaire production régulière |
| `overtime_cost` | `List[float]` | `[15.0]` | Coût unitaire HS |
| `subcontracting_cost` | `List[float]` | `[20.0]` | Coût unitaire sous-traitance |

#### Coûts de Stockage

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `holding_cost` | `List[float]` | `[2.0]` | Coût stockage par unité/période |
| `shortage_cost` | `List[float]` | `[50.0]` | Coût rupture par unité |

#### Stocks

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `initial_stock` | `List[float]` | `[50.0]` | Stock initial |
| `max_stock` | `List[float]` | `[500.0]` | Stock maximum |
| `safety_stock` | `List[float]` | `[20.0]` | Stock de sécurité |

#### Demande

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `demand_mean` | `List[float]` | `[80.0]` | Demande moyenne |
| `demand_std` | `List[float]` | `[15.0]` | Écart-type demande |
| `demand_pattern` | `str` | `'normal'` | Pattern: 'normal', 'seasonal', 'trend' |
| `demand_intensity` | `str` | `'medium'` | Intensité: 'low', 'medium', 'high', 'extreme' |

#### Paramètres Avancés

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `lead_time` | `int` | `0` | Délai de production |
| `setup_cost` | `float` | `0.0` | Coût de setup |
| `min_lot_size` | `float` | `0.0` | Taille de lot minimum |

### Exemple

```python
from config import PDPConfig

config = PDPConfig(
    n_products=1,
    horizon=12,
    regular_capacity=[100],
    overtime_capacity=[30],
    subcontracting_capacity=[50],
    regular_cost=[10],
    overtime_cost=[15],
    subcontracting_cost=[20],
    holding_cost=[2],
    shortage_cost=[50],
    initial_stock=[50],
    demand_mean=[80],
    demand_std=[15],
    demand_intensity='medium'
)
```

---

## TrainingConfig

Configuration pour l'entraînement.

### Classe

```python
@dataclass
class TrainingConfig:
    """
    Configuration des hyperparamètres d'entraînement.
    """
```

### Attributs

| Attribut | Type | Défaut | Description |
|----------|------|--------|-------------|
| `total_timesteps` | `int` | `100000` | Nombre total de pas |
| `learning_rate` | `float` | `3e-4` | Taux d'apprentissage |
| `n_steps` | `int` | `2048` | Pas par rollout |
| `batch_size` | `int` | `64` | Taille des batches |
| `n_epochs` | `int` | `10` | Époques par update |
| `gamma` | `float` | `0.99` | Facteur de discount |
| `gae_lambda` | `float` | `0.95` | Lambda GAE |
| `clip_range` | `float` | `0.2` | Clipping PPO |
| `ent_coef` | `float` | `0.0` | Coefficient entropie |
| `vf_coef` | `float` | `0.5` | Coefficient value function |
| `max_grad_norm` | `float` | `0.5` | Gradient clipping |
| `use_early_stopping` | `bool` | `True` | Activer early stopping |
| `early_stopping_patience` | `int` | `15` | Patience early stopping |

---

## Fonctions Utilitaires

### get_example_config

```python
def get_example_config(name: str) -> PDPConfig:
    """
    Retourne une configuration pré-définie.
    
    Args:
        name: Nom de l'exemple
            - 'rouleurs': Rouleurs industriels (12 périodes)
            - 'pdp_table': Exemple simplifié (6 périodes)
            - 'compresseurs': Compresseurs (8 périodes)
            - 'usinage': Atelier usinage (12 périodes)
    
    Returns:
        PDPConfig configuré
    
    Raises:
        ValueError: Si nom inconnu
    
    Example:
        >>> config = get_example_config('rouleurs')
        >>> print(config.horizon)
        12
    """
```

### load_config_from_json

```python
def load_config_from_json(path: str) -> PDPConfig:
    """
    Charge une configuration depuis un fichier JSON.
    
    Args:
        path: Chemin vers le fichier JSON
    
    Returns:
        PDPConfig
    
    Example:
        >>> config = load_config_from_json('configs/custom.json')
    """
```

### save_config_to_json

```python
def save_config_to_json(config: PDPConfig, path: str):
    """
    Sauvegarde une configuration en JSON.
    
    Args:
        config: Configuration à sauvegarder
        path: Chemin de destination
    """
```

### validate_config

```python
def validate_config(config: PDPConfig) -> List[str]:
    """
    Valide une configuration.
    
    Args:
        config: Configuration à valider
    
    Returns:
        Liste des erreurs (vide si valide)
    
    Example:
        >>> errors = validate_config(config)
        >>> if errors:
        ...     print("Erreurs:", errors)
    """
```
