# 🏭 RLPlanif

<div align="center">

**Optimisation Intelligente du Plan Directeur de Production avec l'Apprentissage par Renforcement**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.2+-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

</div>

---

## 🎯 Qu'est-ce que RLPlanif?

**RLPlanif** est un système avancé d'aide à la décision pour la gestion du **Plan Directeur de Production (PDP)**. Il utilise l'algorithme **PPO (Proximal Policy Optimization)** pour apprendre automatiquement les meilleures stratégies de production face à une demande variable.

Le système optimise trois leviers de production :

| Levier | Description | Coût |
|--------|-------------|------|
| ⚙️ **Production régulière** | Capacité standard | Optimal |
| ⏰ **Heures supplémentaires** | Flexibilité additionnelle | Modéré |
| 🏢 **Sous-traitance** | Capacité externe | Premium |

## ✨ Fonctionnalités

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **Agent PPO Intelligent**

    ---

    Algorithme d'apprentissage par renforcement de pointe pour l'optimisation

-   :material-chart-line:{ .lg .middle } **Comparaison Stratégies**

    ---

    Benchmarking avec Lot-for-Lot, Chase, Level, EOQ

-   :material-monitor-dashboard:{ .lg .middle } **Interface Streamlit**

    ---

    Dashboard interactif pour configuration, entraînement et visualisation

-   :material-factory:{ .lg .middle } **Exemples Industriels**

    ---

    Cas réels : Rouleurs, Compresseurs, Usinage, PDP Table

</div>

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner le repository
git clone https://github.com/NANKOULI/rlplanif.git
cd rlplanif

# Créer l'environnement conda
conda create -n rl-sb python=3.11
conda activate rl-sb

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'interface

```bash
streamlit run app.py
```

### Premier entraînement (CLI)

```bash
python scripts/train.py --config rouleurs --timesteps 50000
```

## 📖 Documentation

| Section | Description |
|---------|-------------|
| [Guide de Démarrage](getting-started/installation.md) | Installation et premier pas |
| [Concepts](concepts/pdp.md) | Théorie du PDP et RL |
| [Architecture](architecture/overview.md) | Structure du projet |
| [Guide Utilisateur](user-guide/configuration.md) | Utilisation détaillée |
| [API Reference](api/environments.md) | Documentation technique |
| [Exemples](examples/industrial-cases.md) | Cas d'usage pratiques |

## 🏗️ Architecture

```
RLPlanif/
├── 🎮 environments/     # Environnements Gymnasium
├── 🤖 agents/           # Agents RL (PPO, baselines)
├── 🧩 components/       # Composants modulaires
├── ⚙️ config/           # Configurations
├── 📜 scripts/          # Scripts CLI
├── 🧪 tests/            # Tests unitaires
├── 📊 models/           # Modèles entraînés
└── 🖥️ app.py           # Interface Streamlit
```

## 📊 Résultats Typiques

L'agent PPO surpasse généralement les stratégies classiques :

| Stratégie | Coût Total | Service Level | Avantage |
|-----------|------------|---------------|----------|
| **PPO** | **-2,450** | **98.5%** | Référence |
| Lot-for-Lot | -3,200 | 95.2% | +30% coût |
| Chase | -2,890 | 96.8% | +18% coût |
| Level | -3,500 | 92.1% | +43% coût |
| EOQ | -2,750 | 97.3% | +12% coût |

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir le [guide de contribution](contributing.md).

## 📄 Licence

Ce projet est sous licence MIT. Voir [LICENSE](https://github.com/NANKOULI/rlplanif/blob/main/LICENSE) pour plus de détails.

---

<div align="center">

**Made with ❤️ for Production Planning**

[GitHub](https://github.com/NANKOULI/rlplanif) · [Documentation](https://rlplanif.readthedocs.io/) · [Issues](https://github.com/NANKOULI/rlplanif/issues)

</div>
