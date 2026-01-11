# Entraînement

Guide complet pour entraîner des modèles PPO avec RLPlanif.

## 🚀 Entraînement Rapide

### Via Streamlit

1. Lancez l'application : `streamlit run app.py`
2. Configurez l'environnement dans **⚙️ Configuration**
3. Allez dans **🏋️ Entraînement PPO**
4. Définissez les hyperparamètres
5. Cliquez sur **🚀 Lancer l'entraînement**

### Via CLI

```bash
python scripts/train.py --config rouleurs --timesteps 100000
```

## ⚙️ Hyperparamètres

### Paramètres Principaux

| Paramètre | Défaut | Description | Impact |
|-----------|--------|-------------|--------|
| `total_timesteps` | 50000 | Nombre total de pas | Plus = meilleur mais plus long |
| `learning_rate` | 3e-4 | Taux d'apprentissage | Trop haut = instable, trop bas = lent |
| `n_steps` | 2048 | Pas par rollout | Plus = plus stable |
| `batch_size` | 64 | Taille mini-batch | Dépend de la mémoire |
| `n_epochs` | 10 | Époques par update | Plus = surapprentissage possible |
| `gamma` | 0.99 | Facteur de discount | Proche de 1 = vision long terme |

### Recommandations par Cas

=== "Problème Simple (1 produit, ≤12 périodes)"

    ```python
    params = {
        'total_timesteps': 50000,
        'learning_rate': 3e-4,
        'n_steps': 1024,
        'batch_size': 64,
    }
    ```

=== "Problème Moyen (1-2 produits, 12-24 périodes)"

    ```python
    params = {
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
    }
    ```

=== "Problème Complexe (3+ produits, 24+ périodes)"

    ```python
    params = {
        'total_timesteps': 500000,
        'learning_rate': 1e-4,
        'n_steps': 4096,
        'batch_size': 128,
        'n_epochs': 15,
    }
    ```

## 📊 Callbacks

### EarlyStopping

Arrête l'entraînement si pas d'amélioration :

```python
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10,  # Évaluations sans amélioration
    min_evals=20,                 # Évaluations minimum
    verbose=1
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/best',
    eval_freq=5000,
    callback_after_eval=stop_callback,
    deterministic=True
)
```

### Checkpoint

Sauvegarde périodique du modèle :

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./models/checkpoints',
    name_prefix='ppo_pdp'
)
```

### Custom Callback

```python
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        # Logique personnalisée
        if self.locals.get('dones', [False])[0]:
            reward = self.locals['rewards'][0]
            self.episode_rewards.append(reward)
        return True
```

## 🔄 Script d'Entraînement Complet

```python
import os
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    StopTrainingOnNoModelImprovement
)

from config import get_example_config
from environments import EnvironmentRegistry

def train(
    config_name: str = 'rouleurs',
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
):
    # Charger la configuration
    config = get_example_config(config_name)
    
    # Créer le dossier de sortie
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'./models/training_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer l'environnement
    def make_env():
        return EnvironmentRegistry.create('strategic', config)
    
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # Callbacks
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15,
        min_evals=10,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / 'best_model'),
        log_path=str(output_dir),
        eval_freq=2500,
        callback_after_eval=stop_callback,
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / 'checkpoints'),
        name_prefix='ppo'
    )
    
    # Créer le modèle
    model = PPO(
        policy='MultiInputPolicy',
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(output_dir / 'tensorboard')
    )
    
    # Entraîner
    print(f"🚀 Début de l'entraînement...")
    print(f"📁 Dossier de sortie: {output_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Sauvegarder
    model.save(str(output_dir / 'final_model'))
    train_env.save(str(output_dir / 'vec_normalize.pkl'))
    
    print(f"✅ Entraînement terminé!")
    print(f"📊 Modèle sauvegardé dans: {output_dir}")
    
    return model, train_env

if __name__ == '__main__':
    train()
```

## 📈 Monitoring avec TensorBoard

### Lancement

```bash
tensorboard --logdir logs/tensorboard
```

### Métriques à Surveiller

| Métrique | Bon signe | Mauvais signe |
|----------|-----------|---------------|
| `ep_rew_mean` | ↗️ Croissant | Stagnant ou oscillant |
| `ep_len_mean` | Stable | Très variable |
| `loss` | ↘️ Décroissant | Croissant |
| `entropy_loss` | Décroît lentement | Décroît trop vite |
| `approx_kl` | < 0.02 | > 0.05 |
| `clip_fraction` | 0.1 - 0.3 | > 0.5 |

### Interprétation

??? success "Entraînement Réussi"
    - Récompense croissante puis plateau
    - Perte décroissante
    - KL divergence faible
    - Entropie décroît progressivement

??? warning "Problèmes Courants"
    **Récompense stagne** :
    
    - Augmenter `n_steps`
    - Augmenter `ent_coef` pour plus d'exploration
    - Vérifier la fonction de récompense
    
    **Instabilité (oscillations)** :
    
    - Réduire `learning_rate`
    - Augmenter `batch_size`
    - Réduire `clip_range`

## 🔧 Tuning Automatique

### Grid Search Simple

```python
from itertools import product

param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'n_steps': [1024, 2048],
    'batch_size': [32, 64],
}

best_reward = -float('inf')
best_params = None

for lr, steps, batch in product(*param_grid.values()):
    model, env = train(
        learning_rate=lr,
        n_steps=steps,
        batch_size=batch,
        total_timesteps=20000  # Court pour le tuning
    )
    
    # Évaluer
    rewards = evaluate(model, env, n_episodes=10)
    mean_reward = np.mean(rewards)
    
    if mean_reward > best_reward:
        best_reward = mean_reward
        best_params = {'lr': lr, 'steps': steps, 'batch': batch}

print(f"Meilleurs paramètres: {best_params}")
```

## Prochaine Étape

➡️ [Évaluation](evaluation.md)
