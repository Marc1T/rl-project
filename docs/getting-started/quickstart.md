# Premier Entraînement

Ce guide vous accompagne pour votre premier entraînement avec RLPlanif.

## 🎯 Objectif

Entraîner un agent PPO sur l'exemple "Rouleurs" et comparer avec les stratégies baseline.

## Option 1 : Interface Streamlit (Recommandé)

### 1. Lancer l'application

```bash
streamlit run app.py
```

L'interface s'ouvre dans votre navigateur à l'adresse `http://localhost:8501`.

### 2. Configurer l'environnement

1. Cliquez sur **⚙️ Configuration** dans la sidebar
2. Sélectionnez **🎯 Exemple Pré-configuré**
3. Choisissez **🔧 Rouleurs (12 périodes)**
4. Cliquez sur **✅ Utiliser cette configuration**

### 3. Lancer l'entraînement

1. Allez dans **🏋️ Entraînement PPO**
2. Configurez les paramètres :
   - **Timesteps** : 50 000 (pour un test rapide)
   - **Learning rate** : 0.0003
3. Cliquez sur **🚀 Lancer l'entraînement**

!!! tip "Conseil"
    Pour de meilleurs résultats, utilisez au moins 100 000 timesteps.

### 4. Évaluer le modèle

1. Allez dans **📊 Évaluation**
2. Sélectionnez votre modèle entraîné
3. Cliquez sur **Évaluer** pour comparer avec les baselines

## Option 2 : Ligne de Commande

### Entraînement simple

```bash
python scripts/train.py --config rouleurs --timesteps 50000
```

### Entraînement avec options avancées

```bash
python scripts/train.py \
    --config rouleurs \
    --timesteps 100000 \
    --learning-rate 0.0003 \
    --n-steps 2048 \
    --batch-size 64 \
    --name mon_premier_modele
```

### Évaluation

```bash
python scripts/evaluate.py --model models/mon_premier_modele/best_model.zip
```

## Structure des Résultats

Après l'entraînement, vous trouverez :

```
models/
└── mon_premier_modele/
    ├── best_model.zip      # Meilleur modèle (sauvegardé par callback)
    ├── final_model.zip     # Modèle final
    ├── vec_normalize.pkl   # Normalisation VecNormalize
    ├── monitor.csv         # Logs d'entraînement
    └── config.json         # Configuration utilisée
```

## Monitoring avec TensorBoard

Pour suivre l'entraînement en temps réel :

```bash
tensorboard --logdir logs/tensorboard
```

Puis ouvrez `http://localhost:6006` dans votre navigateur.

### Métriques Importantes

| Métrique | Description | Bon signe |
|----------|-------------|-----------|
| `rollout/ep_rew_mean` | Récompense moyenne | ↗️ Croissante |
| `rollout/ep_len_mean` | Longueur des épisodes | Stable |
| `train/loss` | Perte totale | ↘️ Décroissante |
| `train/entropy_loss` | Entropie | Décroît lentement |

## Exemple de Code Python

```python
from config import get_example_config
from environments import EnvironmentRegistry
from agents import PPOTrainer

# Charger la configuration
config = get_example_config('rouleurs')

# Créer l'environnement
env = EnvironmentRegistry.create('strategic', config)

# Créer et entraîner l'agent
trainer = PPOTrainer(
    config=config,
    total_timesteps=50000,
    learning_rate=3e-4
)

model = trainer.train()

# Évaluer
results = trainer.evaluate(n_episodes=10)
print(f"Récompense moyenne: {results['mean_reward']:.2f}")
```

## Prochaines Étapes

- ➡️ [Interface Streamlit](streamlit.md) - Guide complet de l'interface
- ➡️ [Configuration avancée](../user-guide/configuration.md) - Personnaliser les paramètres
- ➡️ [Comprendre le PDP](../concepts/pdp.md) - Théorie sous-jacente
