# API Reference - Agents

Documentation technique des agents et stratégies.

## PPOTrainer

Classe principale pour l'entraînement PPO.

### Classe

```python
class PPOTrainer:
    """
    Trainer pour l'algorithme PPO avec Stable-Baselines3.
    
    Attributes:
        config (PDPConfig): Configuration de l'environnement
        model (PPO): Modèle Stable-Baselines3
        vec_env (VecNormalize): Environnement vectorisé normalisé
    """
```

### Constructeur

```python
def __init__(
    self,
    config: PDPConfig,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1
):
    """
    Args:
        config: Configuration PDPConfig
        learning_rate: Taux d'apprentissage
        n_steps: Pas par rollout
        batch_size: Taille des mini-batches
        n_epochs: Époques par update
        gamma: Facteur de discount
        gae_lambda: Lambda pour GAE
        clip_range: Paramètre de clipping PPO
        ent_coef: Coefficient d'entropie
        vf_coef: Coefficient de la value function
        max_grad_norm: Gradient clipping
        tensorboard_log: Chemin pour les logs TB
        verbose: Niveau de verbosité
    """
```

### Méthodes

#### train

```python
def train(
    self,
    total_timesteps: int,
    callbacks: Optional[List[BaseCallback]] = None,
    progress_bar: bool = True
) -> PPO:
    """
    Lance l'entraînement.
    
    Args:
        total_timesteps: Nombre total de pas
        callbacks: Liste de callbacks
        progress_bar: Afficher la barre de progression
    
    Returns:
        Modèle entraîné
    
    Example:
        >>> trainer = PPOTrainer(config)
        >>> model = trainer.train(100000)
    """
```

#### evaluate

```python
def evaluate(
    self,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict:
    """
    Évalue le modèle.
    
    Args:
        n_episodes: Nombre d'épisodes
        deterministic: Actions déterministes
    
    Returns:
        Dict avec rewards, costs, service_levels, etc.
    """
```

#### save

```python
def save(self, path: str):
    """
    Sauvegarde le modèle et VecNormalize.
    
    Args:
        path: Chemin du dossier de sauvegarde
    """
```

#### load

```python
def load(self, path: str):
    """
    Charge un modèle sauvegardé.
    
    Args:
        path: Chemin vers le dossier du modèle
    """
```

---

## BaseStrategy

Classe abstraite pour les stratégies baseline.

### Classe

```python
class BaseStrategy(ABC):
    """
    Classe de base pour les stratégies de production.
    
    Attributes:
        env: Environnement PDP
        config: Configuration
    """
```

### Constructeur

```python
def __init__(self, env: gymnasium.Env):
    """
    Args:
        env: Instance de l'environnement PDP
    """
```

### Méthodes Abstraites

#### get_action

```python
@abstractmethod
def get_action(self, observation: Dict) -> np.ndarray:
    """
    Retourne l'action pour l'observation donnée.
    
    Args:
        observation: Observation de l'environnement
    
    Returns:
        Action normalisée [0,1]^3
    """
```

### Méthodes Concrètes

#### run_episode

```python
def run_episode(self) -> Tuple[float, Dict]:
    """
    Exécute un épisode complet.
    
    Returns:
        total_reward: Récompense totale de l'épisode
        info: Dict avec 'metrics' contenant les infos par période
    """
```

---

## LotForLotStrategy

Stratégie Lot-for-Lot.

```python
class LotForLotStrategy(BaseStrategy):
    """
    Produire exactement la demande nette.
    
    Production = max(0, Demande - Stock)
    """
```

### Comportement

1. Calcule la demande nette
2. Utilise la capacité régulière en priorité
3. Puis heures supplémentaires si nécessaire
4. Puis sous-traitance en dernier recours

---

## ChaseStrategy

Stratégie Chase (poursuite).

```python
class ChaseStrategy(BaseStrategy):
    """
    Suivre la demande en utilisant proportionnellement les capacités.
    """
```

### Comportement

1. Calcule le ratio demande/capacité totale
2. Applique ce ratio à toutes les capacités

---

## LevelStrategy

Stratégie Level (lissage).

```python
class LevelStrategy(BaseStrategy):
    """
    Production constante sur l'horizon.
    
    Production = Demande_moyenne
    """
```

### Comportement

1. Calcule la production moyenne nécessaire
2. Produit cette quantité à chaque période
3. Utilise uniquement la capacité régulière

---

## EOQStrategy

Stratégie EOQ (Quantité Économique).

```python
class EOQStrategy(BaseStrategy):
    """
    Production par lots économiques.
    
    EOQ = sqrt(2*D*S/H)
    """
```

### Comportement

1. Calcule la quantité économique de commande
2. Produit EOQ quand stock < point de réapprovisionnement
3. Ne produit rien sinon

---

## BASELINE_STRATEGIES

Dictionnaire des stratégies disponibles.

```python
BASELINE_STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    'lot_for_lot': LotForLotStrategy,
    'chase': ChaseStrategy,
    'level': LevelStrategy,
    'eoq': EOQStrategy,
}
```

### Utilisation

```python
from agents.baseline_strategies import BASELINE_STRATEGIES

# Créer une stratégie
StrategyClass = BASELINE_STRATEGIES['chase']
strategy = StrategyClass(env)

# Exécuter
reward, info = strategy.run_episode()
```
