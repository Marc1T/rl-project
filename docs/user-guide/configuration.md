# Configuration

Guide complet pour configurer RLPlanif selon vos besoins.

## 📋 Structure de Configuration

### PDPConfig

La classe principale de configuration :

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PDPConfig:
    """Configuration complète pour l'environnement PDP"""
    
    # ===== Dimensions =====
    n_products: int = 1
    horizon: int = 12
    
    # ===== Capacités =====
    regular_capacity: List[float] = field(default_factory=lambda: [100.0])
    overtime_capacity: List[float] = field(default_factory=lambda: [30.0])
    subcontracting_capacity: List[float] = field(default_factory=lambda: [50.0])
    
    # ===== Coûts de Production =====
    regular_cost: List[float] = field(default_factory=lambda: [10.0])
    overtime_cost: List[float] = field(default_factory=lambda: [15.0])
    subcontracting_cost: List[float] = field(default_factory=lambda: [20.0])
    
    # ===== Coûts de Stock =====
    holding_cost: List[float] = field(default_factory=lambda: [2.0])
    shortage_cost: List[float] = field(default_factory=lambda: [50.0])
    
    # ===== Stocks =====
    initial_stock: List[float] = field(default_factory=lambda: [50.0])
    max_stock: List[float] = field(default_factory=lambda: [500.0])
    safety_stock: List[float] = field(default_factory=lambda: [20.0])
    
    # ===== Demande =====
    demand_mean: List[float] = field(default_factory=lambda: [80.0])
    demand_std: List[float] = field(default_factory=lambda: [15.0])
    demand_pattern: str = 'normal'  # 'normal', 'seasonal', 'trend'
    demand_intensity: str = 'medium'  # 'low', 'medium', 'high', 'extreme'
    
    # ===== Paramètres Avancés =====
    lead_time: int = 0
    setup_cost: float = 0.0
    min_lot_size: float = 0.0
```

## ⚙️ Paramètres Détaillés

### Capacités

| Paramètre | Description | Exemple |
|-----------|-------------|---------|
| `regular_capacity` | Capacité de production normale par période | `[100]` |
| `overtime_capacity` | Capacité heures supplémentaires | `[30]` |
| `subcontracting_capacity` | Capacité sous-traitance | `[50]` |

!!! tip "Conseil"
    La capacité totale doit être suffisante pour couvrir les pics de demande :
    `regular + overtime + subcontracting ≥ max(demand)`

### Coûts

| Paramètre | Description | Ratio typique |
|-----------|-------------|---------------|
| `regular_cost` | Coût unitaire production normale | 1x (référence) |
| `overtime_cost` | Coût unitaire heures sup. | 1.5x |
| `subcontracting_cost` | Coût unitaire sous-traitance | 2x |
| `holding_cost` | Coût stockage par unité/période | 0.2x |
| `shortage_cost` | Coût rupture par unité | 5x |

### Demande

| Paramètre | Description | Valeurs |
|-----------|-------------|---------|
| `demand_mean` | Demande moyenne par période | `[80]` |
| `demand_std` | Écart-type de la demande | `[15]` |
| `demand_pattern` | Pattern de demande | `'normal'`, `'seasonal'`, `'trend'` |
| `demand_intensity` | Intensité (stress) | `'low'`, `'medium'`, `'high'`, `'extreme'` |

## 🎯 Exemples Pré-configurés

### Accès aux Exemples

```python
from config import get_example_config

# Charger un exemple
config = get_example_config('rouleurs')
print(config)
```

### Rouleurs

Production de rouleurs industriels sur 12 périodes.

```python
rouleurs_config = PDPConfig(
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
)
```

### PDP Table

Exemple compact sur 6 périodes.

```python
pdp_table_config = PDPConfig(
    n_products=1,
    horizon=6,
    regular_capacity=[80],
    overtime_capacity=[20],
    subcontracting_capacity=[30],
    initial_stock=[30],
    demand_mean=[70],
    demand_std=[10],
)
```

### Compresseurs

Production de compresseurs sur 8 périodes.

```python
compresseurs_config = PDPConfig(
    n_products=1,
    horizon=8,
    regular_capacity=[150],
    overtime_capacity=[40],
    subcontracting_capacity=[60],
    regular_cost=[15],
    overtime_cost=[22],
    subcontracting_cost=[30],
    holding_cost=[3],
    shortage_cost=[75],
    initial_stock=[80],
    demand_mean=[120],
    demand_std=[25],
)
```

## 📁 Configuration via JSON

### Format JSON

```json
{
    "n_products": 1,
    "horizon": 12,
    "regular_capacity": [100],
    "overtime_capacity": [30],
    "subcontracting_capacity": [50],
    "regular_cost": [10],
    "overtime_cost": [15],
    "subcontracting_cost": [20],
    "holding_cost": [2],
    "shortage_cost": [50],
    "initial_stock": [50],
    "max_stock": [500],
    "safety_stock": [20],
    "demand_mean": [80],
    "demand_std": [15],
    "demand_intensity": "medium"
}
```

### Chargement

```python
import json
from config import PDPConfig

with open('my_config.json', 'r') as f:
    data = json.load(f)

config = PDPConfig(**data)
```

### Sauvegarde

```python
import json
from dataclasses import asdict

with open('my_config.json', 'w') as f:
    json.dump(asdict(config), f, indent=2)
```

## 🔧 Configuration Multi-Produits

Pour gérer plusieurs produits :

```python
config = PDPConfig(
    n_products=3,
    horizon=12,
    regular_capacity=[100, 80, 60],      # Par produit
    overtime_capacity=[30, 25, 20],
    subcontracting_capacity=[50, 40, 30],
    regular_cost=[10, 12, 15],
    overtime_cost=[15, 18, 22],
    subcontracting_cost=[20, 24, 30],
    holding_cost=[2, 2.5, 3],
    shortage_cost=[50, 60, 80],
    initial_stock=[50, 40, 30],
    demand_mean=[80, 60, 40],
    demand_std=[15, 12, 8],
)
```

!!! warning "Attention"
    Toutes les listes doivent avoir la même longueur que `n_products`.

## 📊 Validation de Configuration

```python
def validate_config(config: PDPConfig) -> List[str]:
    """Valide une configuration et retourne les erreurs"""
    errors = []
    
    # Vérifier les dimensions
    for attr in ['regular_capacity', 'demand_mean', 'initial_stock']:
        if len(getattr(config, attr)) != config.n_products:
            errors.append(f"{attr} doit avoir {config.n_products} éléments")
    
    # Vérifier les valeurs positives
    if config.horizon <= 0:
        errors.append("horizon doit être positif")
    
    # Vérifier la capacité
    total_cap = config.regular_capacity[0] + config.overtime_capacity[0]
    if total_cap < config.demand_mean[0]:
        errors.append("Capacité insuffisante pour la demande moyenne")
    
    return errors
```

## Prochaine Étape

➡️ [Entraînement](training.md)
