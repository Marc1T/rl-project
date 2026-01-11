# Évaluation

Guide pour évaluer et comparer les performances des modèles.

## 🎯 Objectifs de l'Évaluation

1. **Mesurer les performances** du modèle entraîné
2. **Comparer** avec les stratégies baseline
3. **Analyser** les décisions de production
4. **Valider** avant mise en production

## 📊 Métriques d'Évaluation

### Métriques Financières

| Métrique | Description | Formule |
|----------|-------------|---------|
| **Coût Total** | Somme de tous les coûts | $\sum C_{prod} + C_{stock} + C_{rupture}$ |
| **Coût Moyen/Période** | Coût par période | $C_{total} / T$ |
| **Coût de Production** | Coûts directs | $\sum c \cdot P$ |
| **Coût de Stockage** | Coûts d'inventaire | $\sum h \cdot I$ |
| **Coût de Rupture** | Coûts de pénurie | $\sum b \cdot B$ |

### Métriques de Service

| Métrique | Description | Cible |
|----------|-------------|-------|
| **Service Level** | % demande satisfaite | ≥ 95% |
| **Fill Rate** | Taux de satisfaction immédiate | ≥ 90% |
| **Stockout Rate** | Fréquence des ruptures | ≤ 5% |

### Métriques de Production

| Métrique | Description |
|----------|-------------|
| **Utilisation Capacité** | % capacité utilisée |
| **Heures Sup. Ratio** | % production en HS |
| **Sous-traitance Ratio** | % production sous-traitée |

## 🔬 Évaluation via Streamlit

### Étapes

1. Allez dans **📊 Évaluation**
2. Sélectionnez le modèle à évaluer
3. Choisissez les stratégies de comparaison
4. Définissez le nombre d'épisodes (10-50)
5. Lancez l'évaluation

### Résultats Affichés

- Tableau comparatif des métriques
- Graphiques de performance
- Détails par stratégie

## 💻 Évaluation via Code

### Évaluation Simple

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from config import get_example_config
from environments import EnvironmentRegistry

def evaluate_model(model_path: str, n_episodes: int = 10):
    """Évalue un modèle PPO"""
    config = get_example_config('rouleurs')
    
    # Recréer l'environnement
    def make_env():
        return EnvironmentRegistry.create('strategic', config)
    
    eval_env = DummyVecEnv([make_env])
    
    # Charger VecNormalize si disponible
    vec_norm_path = model_path.replace('model.zip', 'vec_normalize.pkl')
    if os.path.exists(vec_norm_path):
        eval_env = VecNormalize.load(vec_norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    
    # Charger le modèle
    model = PPO.load(model_path, env=eval_env)
    
    # Évaluer
    results = {
        'rewards': [],
        'costs': [],
        'service_levels': [],
        'episode_metrics': []
    }
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        service_levels = []
        metrics = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            total_reward += reward[0]
            total_cost += sum(info[0]['costs'].values())
            service_levels.append(info[0]['demand_fulfillment'])
            metrics.append(info[0])
        
        results['rewards'].append(total_reward)
        results['costs'].append(total_cost)
        results['service_levels'].append(np.mean(service_levels))
        results['episode_metrics'].append(metrics)
    
    return results
```

### Comparaison avec Baselines

```python
from agents.baseline_strategies import BASELINE_STRATEGIES

def compare_with_baselines(model_path: str, n_episodes: int = 10):
    """Compare le modèle PPO avec les baselines"""
    config = get_example_config('rouleurs')
    
    all_results = {}
    
    # Évaluer PPO
    ppo_results = evaluate_model(model_path, n_episodes)
    all_results['PPO'] = {
        'mean_reward': np.mean(ppo_results['rewards']),
        'std_reward': np.std(ppo_results['rewards']),
        'mean_cost': np.mean(ppo_results['costs']),
        'mean_service': np.mean(ppo_results['service_levels']),
    }
    
    # Évaluer les baselines
    for name, StrategyClass in BASELINE_STRATEGIES.items():
        rewards = []
        costs = []
        service_levels = []
        
        for _ in range(n_episodes):
            env = EnvironmentRegistry.create('strategic', config)
            strategy = StrategyClass(env)
            reward, info = strategy.run_episode()
            
            total_cost = sum(
                m['costs']['production_cost'] + 
                m['costs']['inventory_cost'] + 
                m['costs']['shortage_cost']
                for m in info['metrics']
            )
            
            rewards.append(reward)
            costs.append(total_cost)
            service_levels.append(
                np.mean([m['demand_fulfillment'] for m in info['metrics']])
            )
        
        all_results[name] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_cost': np.mean(costs),
            'mean_service': np.mean(service_levels),
        }
    
    return all_results
```

### Génération de Rapport

```python
import pandas as pd

def generate_report(results: dict) -> pd.DataFrame:
    """Génère un rapport comparatif"""
    data = []
    
    for strategy, metrics in results.items():
        data.append({
            'Stratégie': strategy,
            'Récompense Moy.': f"{metrics['mean_reward']:.2f}",
            'Écart-Type': f"{metrics['std_reward']:.2f}",
            'Coût Total': f"{metrics['mean_cost']:.0f}",
            'Service Level': f"{metrics['mean_service']:.1%}",
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Récompense Moy.', ascending=False)
    
    return df

# Utilisation
results = compare_with_baselines('./models/best_model/model.zip')
report = generate_report(results)
print(report.to_string(index=False))
```

## 📈 Analyse Détaillée

### Évolution Temporelle

```python
import plotly.graph_objects as go

def plot_episode_analysis(metrics: list, title: str = "Analyse d'Épisode"):
    """Visualise un épisode en détail"""
    periods = list(range(1, len(metrics) + 1))
    
    fig = go.Figure()
    
    # Production
    fig.add_trace(go.Bar(
        x=periods,
        y=[m['total_production'] for m in metrics],
        name='Production',
        marker_color='blue'
    ))
    
    # Demande
    fig.add_trace(go.Scatter(
        x=periods,
        y=[m['raw_metrics']['current_demand'][0] for m in metrics],
        mode='lines+markers',
        name='Demande',
        line=dict(color='red', width=2)
    ))
    
    # Stock
    fig.add_trace(go.Scatter(
        x=periods,
        y=[m['inventory_level'][0] for m in metrics],
        mode='lines+markers',
        name='Stock',
        line=dict(color='green', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Période',
        yaxis_title='Unités',
        yaxis2=dict(
            title='Stock',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig
```

### Répartition des Coûts

```python
def plot_cost_breakdown(metrics: list):
    """Visualise la répartition des coûts"""
    costs = {
        'Production': sum(m['costs']['production_cost'] for m in metrics),
        'Stockage': sum(m['costs']['inventory_cost'] for m in metrics),
        'Rupture': sum(m['costs']['shortage_cost'] for m in metrics),
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(costs.keys()),
        values=list(costs.values()),
        hole=0.4
    )])
    
    fig.update_layout(title='Répartition des Coûts')
    
    return fig
```

## ✅ Critères de Validation

### Avant Mise en Production

| Critère | Seuil | Validation |
|---------|-------|------------|
| Service Level | ≥ 95% | ✅ Obligatoire |
| Coût vs L4L | ≤ 100% | ✅ Recommandé |
| Stabilité (std) | < 10% | ✅ Recommandé |
| Consistance | 10+ épisodes | ✅ Obligatoire |

### Checklist

- [ ] Évaluation sur 50+ épisodes
- [ ] Comparaison avec toutes les baselines
- [ ] Service level ≥ 95%
- [ ] Coût total ≤ meilleure baseline
- [ ] Analyse des cas extrêmes
- [ ] Test avec demande "extreme"

## Prochaine Étape

➡️ [Stratégies Baseline](baselines.md)
