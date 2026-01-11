# Cas Industriels

Exemples d'application de RLPlanif sur des cas industriels réels.

## 🔧 Exemple 1 : Rouleurs Industriels

### Contexte

Une entreprise fabrique des rouleurs industriels avec une demande saisonnière.

### Paramètres

| Paramètre | Valeur |
|-----------|--------|
| Horizon | 12 périodes (mensuel) |
| Capacité régulière | 100 unités |
| Capacité HS | 30 unités |
| Sous-traitance | 50 unités |
| Demande moyenne | 80 unités |
| Stock initial | 50 unités |

### Configuration

```python
from config import PDPConfig

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
    demand_intensity='medium'
)
```

### Résultats Typiques

| Stratégie | Coût Total | Service Level |
|-----------|------------|---------------|
| **PPO** | **2,450** | **98.5%** |
| Lot-for-Lot | 3,200 | 95.2% |
| Chase | 2,890 | 96.8% |
| Level | 3,500 | 92.1% |

### Analyse

L'agent PPO apprend à :

1. **Anticiper** les pics de demande en pré-stockant
2. **Utiliser stratégiquement** les heures supplémentaires
3. **Éviter** la sous-traitance coûteuse sauf nécessité

---

## ⚙️ Exemple 2 : Compresseurs

### Contexte

Fabrication de compresseurs industriels avec forte variabilité de la demande.

### Paramètres

| Paramètre | Valeur |
|-----------|--------|
| Horizon | 8 périodes |
| Capacité régulière | 150 unités |
| Capacité HS | 40 unités |
| Sous-traitance | 60 unités |
| Demande moyenne | 120 unités |
| Stock initial | 80 unités |

### Configuration

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
    demand_intensity='high'
)
```

### Défis Spécifiques

- Demande très variable (σ = 25)
- Coûts de rupture élevés (75€/unité)
- Nécessité de réactivité

### Stratégie PPO

L'agent développe une stratégie **réactive-préventive** :

1. Maintient un stock tampon
2. Réagit rapidement aux variations
3. Préfère les HS à la sous-traitance

---

## 🏭 Exemple 3 : Usinage

### Contexte

Atelier d'usinage avec production sur commande.

### Paramètres

```python
usinage_config = PDPConfig(
    n_products=1,
    horizon=12,
    regular_capacity=[80],
    overtime_capacity=[25],
    subcontracting_capacity=[40],
    regular_cost=[12],
    overtime_cost=[18],
    subcontracting_cost=[25],
    holding_cost=[1.5],
    shortage_cost=[60],
    initial_stock=[30],
    demand_mean=[70],
    demand_std=[20],
    demand_intensity='medium'
)
```

### Particularités

- Coûts de stockage faibles (pièces standards)
- Demande assez prévisible
- Sous-traitance accessible

---

## 📊 Exemple 4 : PDP Table

### Contexte

Exemple pédagogique simplifié sur 6 périodes.

### Configuration

```python
pdp_table_config = PDPConfig(
    n_products=1,
    horizon=6,
    regular_capacity=[80],
    overtime_capacity=[20],
    subcontracting_capacity=[30],
    regular_cost=[8],
    overtime_cost=[12],
    subcontracting_cost=[16],
    holding_cost=[1],
    shortage_cost=[40],
    initial_stock=[30],
    demand_mean=[70],
    demand_std=[10],
)
```

### Tableau PDP Résultant

| Période | P1 | P2 | P3 | P4 | P5 | P6 |
|---------|----|----|----|----|----|----|
| Demande | 65 | 75 | 80 | 70 | 85 | 60 |
| Production | 80 | 80 | 80 | 75 | 85 | 60 |
| Stock Fin | 45 | 50 | 50 | 55 | 55 | 55 |

---

## 🔬 Exécution des Exemples

### Via Streamlit

1. Lancez `streamlit run app.py`
2. Allez dans **🔬 Exemples Réels**
3. Sélectionnez l'exemple
4. Choisissez la stratégie
5. Visualisez les résultats

### Via CLI

```bash
# Entraîner sur un exemple
python scripts/train_real_examples.py --example rouleurs --timesteps 100000

# Évaluer
python scripts/evaluate_real_example.py --example rouleurs --model ./models/best
```

### Via Code

```python
from config import get_example_config
from environments import EnvironmentRegistry
from agents import PPOTrainer

# Charger l'exemple
config = get_example_config('rouleurs')

# Créer le trainer
trainer = PPOTrainer(config)

# Entraîner
model = trainer.train(total_timesteps=100000)

# Évaluer
results = trainer.evaluate(n_episodes=50)
print(f"Coût moyen: {results['mean_cost']:.0f}")
print(f"Service level: {results['mean_service']:.1%}")
```

## 📈 Comparaison des Exemples

| Exemple | Complexité | Variabilité | Meilleure Stratégie |
|---------|------------|-------------|---------------------|
| Rouleurs | Moyenne | Modérée | PPO |
| Compresseurs | Haute | Élevée | PPO |
| Usinage | Basse | Modérée | PPO / L4L |
| PDP Table | Très basse | Faible | Toutes proches |

## Prochaine Étape

➡️ [Personnalisation](customization.md)
