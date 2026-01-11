# 🏭 RLPlanif - Plan Directeur de Production Intelligent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.2+-green.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**Optimisation du Plan Directeur de Production avec l'Apprentissage par Renforcement**

[Documentation](https://rl-project.readthedocs.io/) · [Démo](#-interface-streamlit) · [Installation](#-installation)

</div>

---

## 👨‍🎓 Informations du Projet

| | |
|---|---|
| **Auteur** | NANKOULI Marc Thierry |
| **Encadrant** | Prof. TAWFIK Masrour |
| **Institution** | ENSAM Meknès |
| **Filière** | IATD-SI (Intelligence Artificielle et Technologie des Données : Systèmes Industriels ) |
| **Module** | Reinforcement Learning |
| **Année** | 2025/2026 |

---

## 🎯 Présentation

**RLPlanif** est un système d'aide à la décision pour l'optimisation du **Plan Directeur de Production (PDP)** utilisant l'algorithme **PPO (Proximal Policy Optimization)**.

### Problématique

Comment planifier efficacement la production pour :
- ✅ Satisfaire la demande client
- ✅ Minimiser les coûts (production, stockage, rupture)
- ✅ Optimiser l'utilisation des capacités

### Solution

Un agent RL apprend automatiquement à équilibrer trois leviers de production :

| Levier | Description | Coût Relatif |
|--------|-------------|--------------|
| ⚙️ **Production régulière** | Capacité standard | 1x |
| ⏰ **Heures supplémentaires** | Flexibilité additionnelle | 1.5x |
| 🏢 **Sous-traitance** | Capacité externe | 2x |

---

## ✨ Fonctionnalités

- 🤖 **Agent PPO** entraîné avec Stable-Baselines3
- 📊 **Comparaison** avec 4 stratégies baseline (Lot-for-Lot, Chase, Level, EOQ)
- 🎮 **Interface Streamlit** interactive pour configuration et visualisation
- 📈 **Visualisations** Plotly des performances
- 📋 **Tableaux PDP** détaillés exportables
- 🔬 **Exemples industriels** : Rouleurs, Compresseurs, Usinage

---

## 📥 Installation

### Prérequis

- Python 3.9+ (recommandé : 3.11)
- Conda ou pip

### Installation Rapide

```bash
# Cloner le projet
git clone https://github.com/Marc1T/rl-project.git
cd rl-project

# Créer l'environnement conda
conda create -n rl-sb python=3.11 -y
conda activate rl-sb

# Installer les dépendances
pip install -r requirements.txt
```

### Vérification

```bash
python scripts/test_env_diagnostic.py
```

**Résultat attendu :**
```
✅ PASS: Fonctionnalités de base
✅ PASS: Échelle des rewards
✅ PASS: Cohérence d'épisode
✅ PASS: Normalisation

4/4 tests réussis
🎉 Tous les tests sont passés!
```

---

## 🚀 Démarrage Rapide

### Option 1 : Interface Streamlit (Recommandé)

```bash
streamlit run app.py
```

L'interface s'ouvre sur `http://localhost:8501` avec :
- ⚙️ Configuration de l'environnement
- 🏋️ Entraînement PPO
- 📊 Évaluation et comparaison
- 📈 Visualisations interactives

### Option 2 : Ligne de Commande

```bash
# Entraînement
python scripts/train.py --products 1 --timesteps 100000 --horizon 12

# Évaluation
python scripts/evaluate.py --model ./models/[VOTRE_MODELE]/best_model

# Comparaison avec baselines
python scripts/compare_strategies.py
```

---

## 🖥️ Interface Streamlit

<table>
<tr>
<td width="50%">

### 🏠 Accueil
- Présentation du projet
- Statistiques du système
- Guide de démarrage

</td>
<td width="50%">

### ⚙️ Configuration
- Exemples pré-configurés
- Configuration personnalisée
- Import/Export JSON

</td>
</tr>
<tr>
<td>

### 🏋️ Entraînement
- Paramètres PPO ajustables
- Barre de progression
- Logs en temps réel

</td>
<td>

### 📊 Évaluation
- Comparaison PPO vs Baselines
- Métriques détaillées
- Graphiques interactifs

</td>
</tr>
</table>

---

## 📁 Structure du Projet

```
rl-project/
├── 🖥️ app.py                 # Interface Streamlit
├── 📋 requirements.txt       # Dépendances
│
├── 🎮 environments/          # Environnements Gymnasium
│   ├── base_pdp_env.py
│   ├── strategic_pdp_env.py
│   └── env_registry.py
│
├── 🤖 agents/                # Agents RL
│   ├── ppo_trainer.py
│   ├── baseline_strategies.py
│   └── rl_utils.py
│
├── 🧩 components/            # Composants modulaires
│   ├── demand_generators.py
│   ├── cost_calculators.py
│   ├── reward_calculators.py
│   └── ...
│
├── ⚙️ config/                # Configurations
│   ├── base_config.py
│   ├── environment_configs.py
│   └── real_examples_configs.py
│
├── 📜 scripts/               # Scripts CLI
│   ├── train.py
│   ├── evaluate.py
│   └── compare_strategies.py
│
├── 📚 docs/                  # Documentation MkDocs
│
├── 📊 models/                # Modèles sauvegardés
└── 📈 logs/                  # Logs TensorBoard
```

---

## 📊 Résultats Typiques

### Performance sur l'exemple "Rouleurs"

| Stratégie | Coût Total | Service Level | Avantage PPO |
|-----------|------------|---------------|--------------|
| **PPO** | **2,450** | **98.5%** | - |
| Lot-for-Lot | 3,200 | 95.2% | +30% |
| Chase | 2,890 | 96.8% | +18% |
| Level | 3,500 | 92.1% | +43% |
| EOQ | 2,750 | 97.3% | +12% |

---

## 📈 Monitoring avec TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Ouvrez `http://localhost:6006` pour suivre :
- 📈 Récompense moyenne
- 📉 Perte d'entraînement
- 🎲 Entropie de la politique

---

## 🔧 Configuration Avancée

### Intensité de la Demande

```python
# Dans la configuration
config = PDPConfig(
    demand_intensity='high'  # 'low', 'medium', 'high', 'extreme'
)
```

| Intensité | Multiplicateur | Effet |
|-----------|----------------|-------|
| `low` | 0.75 | Demande modérée |
| `medium` | 0.90 | Standard |
| `high` | 1.05 | Demande élevée, plus de HS |
| `extreme` | 1.20 | Stress test |

### Multi-Produits

```bash
python scripts/train.py --products 3 --timesteps 300000
```

---

## 📚 Documentation

La documentation complète est disponible sur [ReadTheDocs](https://rl-project.readthedocs.io/).

### Sections

- 📖 [Guide de démarrage](https://rl-project.readthedocs.io/getting-started/installation/)
- 🎓 [Concepts théoriques](https://rl-project.readthedocs.io/concepts/pdp/) (PDP, RL, PPO)
- 🏗️ [Architecture](https://rl-project.readthedocs.io/architecture/overview/)
- 📘 [API Reference](https://rl-project.readthedocs.io/api/environments/)

---

## 🛠️ Technologies

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| Python | 3.9+ | Langage principal |
| Stable-Baselines3 | 2.2.1 | Algorithme PPO |
| Gymnasium | 0.29.1 | Environnements RL |
| PyTorch | 2.1.0 | Backend deep learning |
| Streamlit | 1.28+ | Interface web |
| Plotly | 5.18+ | Visualisations |
| NumPy/Pandas | Latest | Calcul et données |

---

## 🛠️ Dépannage Rapide

### Erreur CUDA
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### Erreur de mémoire
Réduire `batch_size` dans la configuration:
```python
training_config = PPOTrainingConfig(
    batch_size=32,  # Réduire de 64 à 32
    n_steps=512     # Réduire de 1024
)
```

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Prof. TAWFIK Masrour** pour l'encadrement et les conseils
- **ENSAM Meknès** pour le cadre académique
- **OpenAI** pour l'algorithme PPO
- **Stable-Baselines3** pour l'implémentation

---

<div align="center">

**Projet réalisé par NANKOULI Marc Thierry**  
ENSAM Meknès - IATD-SI - 2025/2026

[⬆ Retour en haut](#-rlplanif---plan-directeur-de-production-intelligent)

</div>
