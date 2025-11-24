# 🚀 Guide de Démarrage Rapide - Environnement PDP Corrigé

## 📥 Installation

```bash
# Cloner le projet
cd rl-project

# Installer les dépendances
pip install -r requirements.txt
```

## 🔍 Étape 1: Diagnostic de l'Environnement (5 min)

Lancez le script de diagnostic pour vérifier que tout fonctionne:

```bash
python scripts/test_env_diagnostic.py
```

**Résultat attendu:**
```
✅ PASS: Fonctionnalités de base
✅ PASS: Échelle des rewards
✅ PASS: Cohérence d'épisode
✅ PASS: Normalisation

4/4 tests réussis
🎉 Tous les tests sont passés! L'environnement est prêt.
```

**Si un test échoue:**
- Vérifiez que vous avez bien remplacé les fichiers corrigés
- Consultez les messages d'erreur détaillés

---

## 🏋️ Étape 2: Entraînement Initial (30-60 min)

### Entraînement Court (Test)

```bash
python scripts/train.py \
    --products 1 \
    --timesteps 50000 \
    --horizon 12 \
    --env_type strategic
```

**Ce qui se passe:**
- L'environnement est créé et normalisé
- PPO commence l'entraînement avec les hyperparamètres optimisés
- Les modèles sont sauvegardés toutes les 10k timesteps
- Le meilleur modèle est sauvegardé automatiquement

**Temps estimé:** ~30 minutes sur CPU, ~10 minutes sur GPU

### Entraînement Complet

```bash
python scripts/train.py \
    --products 1 \
    --timesteps 200000 \
    --horizon 12 \
    --env_type strategic
```

---

## 📊 Étape 3: Monitoring (En parallèle)

Dans un autre terminal, lancez TensorBoard:

```bash
tensorboard --logdir ./logs/tensorboard/
```

Ouvrez votre navigateur: `http://localhost:6006`

**Métriques à surveiller:**

1. **ep_rew_mean** (Reward moyen par épisode)
   - ❌ Mauvais: Reste constant ou diminue
   - ✅ Bon: Augmente progressivement

2. **ep_len_mean** (Longueur moyenne des épisodes)
   - Devrait être constant = horizon (12)

3. **value_loss** (Perte de la value function)
   - ❌ Mauvais: Explose ou reste très élevé
   - ✅ Bon: Diminue progressivement

4. **policy_loss** (Perte de la policy)
   - Devrait rester stable et faible

---

## 📈 Étape 4: Évaluation (5 min)

Une fois l'entraînement terminé:

```bash
python scripts/evaluate.py \
    --model ./models/ppo_pdp_strategic_1prod_[DATE]/best_model \
    --episodes 10 \
    --env_type strategic
```

**Résultat attendu:**

```
📊 PERFORMANCE MOYENNE:
   Reward: -2450.3 ± 180.2
   Stock final: 75.2
   Niveau service: 0.945
```

**Interprétation:**
- **Reward:** Plus élevé = meilleur (moins négatif)
- **Service level > 0.90:** ✅ Bon
- **Stock final 50-150:** ✅ Équilibré

---

## 🔄 Étape 5: Comparaison avec Baselines (10 min)

Comparez votre modèle RL avec les stratégies classiques:

```bash
python scripts/compare_strategies.py
```

**Résultat attendu:**

```
COMPARAISON DES STRATÉGIES
┌─────────────────────┬───────────┬─────────────┬──────────┐
│ Stratégie           │ Reward    │ Service     │ Stock    │
├─────────────────────┼───────────┼─────────────┼──────────┤
│ Level               │ -3200.5   │ 0.850       │ 180.2    │
│ Chase               │ -2800.3   │ 0.920       │ 45.8     │
│ Fixed Moderate      │ -3500.1   │ 0.780       │ 220.5    │
│ PPO (votre modèle)  │ -2450.3   │ 0.945       │ 75.2     │
└─────────────────────┴───────────┴─────────────┴──────────┘

🏆 Meilleure stratégie: PPO
```

---

## 🐛 Troubleshooting

### Problème 1: Le reward ne s'améliore pas

**Symptômes:**
- Le reward moyen stagne
- Le reward oscille sans converger

**Solutions:**
1. Vérifier que VecNormalize est bien utilisé
2. Augmenter le nombre de timesteps (100k → 200k)
3. Réduire le learning rate: `learning_rate: float = 1e-4`
4. Essayer l'environnement `base` au lieu de `strategic`

### Problème 2: Les rewards explosent

**Symptômes:**
- Rewards > 1000 ou < -10000
- Value loss explose

**Solutions:**
1. Vérifier la normalisation dans `normalizers.py`
2. Augmenter `clip_reward` dans VecNormalize
3. Réduire les poids des rewards dans `environment_configs.py`

### Problème 3: Service level toujours faible

**Symptômes:**
- Service level < 0.80 après entraînement

**Solutions:**
1. Augmenter `service_bonus` weight dans la config
2. Augmenter `shortage_cost` dans `base_config.py`
3. Vérifier que les demandes ne sont pas trop élevées

### Problème 4: Stock négatif permanent

**Symptômes:**
- Stock toujours < 0
- Coût de shortage très élevé

**Solutions:**
1. Augmenter `initial_stock` dans la config
2. Vérifier que `allow_backorders=True`
3. Ajuster les capacités de production

---

## 📁 Structure des Résultats

Après l'entraînement, vous devriez avoir:

```
rl-project/
├── models/
│   └── ppo_pdp_strategic_1prod_20241124_143022/
│       ├── best_model.zip          # Meilleur modèle
│       ├── final_model.zip         # Modèle final
│       ├── vec_normalize.pkl       # Normalisation
│       └── monitor.csv             # Logs d'entraînement
├── logs/
│   └── tensorboard/
│       └── ppo_pdp_training_1/     # Logs TensorBoard
└── evaluation_metrics.json          # Métriques d'évaluation
```

---

## 🎯 Objectifs de Performance

### Niveau Débutant (Baseline)
- ✅ Le modèle termine l'entraînement sans erreur
- ✅ Service level > 0.80
- ✅ Reward meilleur que "Fixed Moderate" strategy

### Niveau Intermédiaire
- ✅ Service level > 0.90
- ✅ Reward meilleur que "Level" strategy
- ✅ Stock final entre 50-150

### Niveau Avancé
- ✅ Service level > 0.95
- ✅ Reward meilleur que toutes les baselines
- ✅ Stock stable avec faible variance
- ✅ Coûts de production optimisés

---

## 🔧 Configurations Avancées

### Multi-Produits

```bash
python scripts/train.py \
    --products 3 \
    --timesteps 300000 \
    --horizon 12 \
    --env_type strategic
```

### Horizon Plus Long

```bash
python scripts/train.py \
    --products 1 \
    --timesteps 200000 \
    --horizon 24 \
    --env_type strategic
```

### Environnement de Base (Plus Simple)

```bash
python scripts/train.py \
    --products 1 \
    --timesteps 100000 \
    --horizon 12 \
    --env_type base
```

---

## 📚 Ressources Supplémentaires

- **Stable-Baselines3 Docs:** https://stable-baselines3.readthedocs.io/
- **PPO Paper:** https://arxiv.org/abs/1707.06347
- **RL Debugging:** https://andyljones.com/posts/rl-debugging.html

---

## ✅ Checklist de Vérification

Avant de signaler un problème:

- [ ] J'ai lancé `test_env_diagnostic.py` et tous les tests passent
- [ ] J'ai vérifié TensorBoard et les métriques sont logiques
- [ ] J'ai comparé avec les baselines
- [ ] J'ai essayé avec différents seeds
- [ ] J'ai vérifié que VecNormalize est bien sauvegardé/chargé

---

## 🎉 Prochaines Étapes

Une fois que votre modèle fonctionne bien:

1. **Expérimentation:**
   - Tester différents poids de reward
   - Essayer différentes architectures de réseau
   - Ajouter des contraintes supplémentaires

2. **Validation:**
   - Tester sur des scénarios de demande variés
   - Analyser la robustesse aux perturbations
   - Comparer avec des données réelles

3. **Déploiement:**
   - Créer une interface de visualisation
   - Intégrer avec un système de gestion de production
   - Mettre en place un monitoring en production

