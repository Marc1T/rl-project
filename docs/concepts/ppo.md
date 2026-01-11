# Algorithme PPO

## 📖 Introduction

**PPO** (Proximal Policy Optimization) est un algorithme d'apprentissage par renforcement développé par OpenAI en 2017. Il est devenu l'un des algorithmes les plus populaires grâce à son équilibre entre **simplicité**, **stabilité** et **performance**.

## 🎯 Pourquoi PPO ?

| Critère | PPO | Autres |
|---------|-----|--------|
| **Stabilité** | ⭐⭐⭐ | Variable |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐ |
| **Simplicité** | ⭐⭐⭐ | ⭐⭐ |
| **Hyperparamètres** | Peu sensible | Très sensible |

## 🧮 Fonctionnement

### Architecture Actor-Critic

PPO utilise deux réseaux de neurones :

```mermaid
graph TB
    S[État s] --> A[Actor π_θ]
    S --> C[Critic V_φ]
    A --> Ac[Action a]
    C --> V[Valeur V(s)]
```

### Fonction Objectif

L'objectif de PPO est de maximiser :

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

Où :

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ : Ratio des politiques
- $\hat{A}_t$ : Estimateur de l'avantage
- $\epsilon$ : Paramètre de clipping (typiquement 0.2)

### Clipping Expliqué

Le clipping empêche des mises à jour trop importantes :

```
Si Avantage > 0 (bonne action) :
    ratio ≤ 1 + ε → Limite l'augmentation de probabilité

Si Avantage < 0 (mauvaise action) :
    ratio ≥ 1 - ε → Limite la diminution de probabilité
```

### Fonction de Valeur

La perte du Critic :

$$L^{VF}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - V_t^{target})^2 \right]$$

### Bonus d'Entropie

Pour encourager l'exploration :

$$S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$$

### Perte Totale

$$L^{TOTAL}(\theta, \phi) = L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 S[\pi_\theta]$$

## ⚙️ Hyperparamètres

### Paramètres Principaux

| Paramètre | Valeur Défaut | Description |
|-----------|---------------|-------------|
| `learning_rate` | 3e-4 | Taux d'apprentissage |
| `n_steps` | 2048 | Pas par rollout |
| `batch_size` | 64 | Taille des mini-batches |
| `n_epochs` | 10 | Époques par update |
| `gamma` | 0.99 | Facteur de discount |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | Paramètre ε de clipping |
| `ent_coef` | 0.0 | Coefficient d'entropie |
| `vf_coef` | 0.5 | Coefficient critic |

### Recommandations pour le PDP

```python
ppo_params = {
    'learning_rate': 3e-4,      # Standard
    'n_steps': 2048,            # Rollouts suffisants
    'batch_size': 64,           # Mini-batches
    'n_epochs': 10,             # Passes sur les données
    'gamma': 0.99,              # Horizon long
    'gae_lambda': 0.95,         # GAE standard
    'clip_range': 0.2,          # Clipping standard
    'ent_coef': 0.01,           # Légère exploration
    'max_grad_norm': 0.5,       # Gradient clipping
}
```

## 🔄 Algorithme

```
Pour chaque itération :
    1. Collecter T timesteps avec la politique actuelle π_θ
    2. Calculer les avantages Â avec GAE
    3. Pour chaque époque k = 1, ..., K :
        a. Échantillonner mini-batches
        b. Calculer le ratio r_t(θ)
        c. Calculer la perte clippée L^CLIP
        d. Mettre à jour θ par gradient ascent
    4. θ_old ← θ
```

## 📊 GAE (Generalized Advantage Estimation)

L'avantage est estimé par :

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Où $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ est l'erreur TD.

## 🛠️ Implémentation dans RLPlanif

### Configuration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Créer l'environnement
env = DummyVecEnv([lambda: EnvironmentRegistry.create('strategic', config)])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Créer le modèle PPO
model = PPO(
    policy='MultiInputPolicy',  # Pour Dict observations
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log='./logs/tensorboard'
)

# Entraîner
model.learn(total_timesteps=100000)
```

### Architecture du Réseau

```python
policy_kwargs = {
    'net_arch': dict(
        pi=[128, 128],  # Actor
        vf=[128, 128]   # Critic
    ),
    'activation_fn': torch.nn.ReLU
}
```

## 📈 Monitoring

### Métriques TensorBoard

| Métrique | Signification |
|----------|---------------|
| `rollout/ep_rew_mean` | Récompense moyenne ↗️ |
| `train/loss` | Perte totale ↘️ |
| `train/policy_gradient_loss` | Perte actor |
| `train/value_loss` | Perte critic |
| `train/entropy_loss` | Entropie ↘️ lentement |
| `train/clip_fraction` | Fraction clippée |
| `train/approx_kl` | KL divergence < 0.02 |

## 🔬 Tuning

### Problèmes Courants

??? warning "Récompense stagne"
    - Augmenter `n_steps` pour plus d'exploration
    - Augmenter `ent_coef` (0.01 → 0.05)
    - Vérifier la fonction de récompense

??? warning "Instabilité (oscillations)"
    - Réduire `learning_rate`
    - Réduire `clip_range` (0.2 → 0.1)
    - Augmenter `batch_size`

??? warning "KL divergence élevée"
    - Réduire `learning_rate`
    - Réduire `n_epochs`

## Prochaine Étape

➡️ [Architecture du Projet](../architecture/overview.md)
