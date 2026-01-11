# Changelog

Toutes les modifications notables de ce projet sont documentées ici.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.0] - 2026-01-11

### ✨ Ajouté

- **Environnements Gymnasium**
  - `BasePDPEnv` : Environnement de base pour le PDP
  - `StrategicPDPEnv` : Environnement enrichi avec observations étendues
  - `EnvironmentRegistry` : Factory pattern pour la création d'environnements

- **Agent PPO**
  - `PPOTrainer` : Classe d'entraînement avec Stable-Baselines3
  - Support de VecNormalize pour la normalisation des observations
  - Callbacks : EarlyStopping, Checkpoint, Evaluation

- **Stratégies Baseline**
  - Lot-for-Lot : Production = Demande nette
  - Chase : Suivre la demande proportionnellement
  - Level : Production constante
  - EOQ : Quantité économique de commande

- **Composants Modulaires**
  - `DemandGenerator` : Génération de demande avec intensités
  - `CostCalculator` : Calcul des coûts de production
  - `RewardCalculator` : Calcul de la récompense RL
  - `ObservationBuilder` : Construction des observations
  - `ActionValidator` : Validation des actions

- **Interface Streamlit**
  - Page d'accueil avec présentation du projet
  - Configuration : exemples pré-configurés, personnalisée, JSON
  - Entraînement PPO avec barre de progression
  - Évaluation et comparaison avec baselines
  - Visualisations interactives avec Plotly
  - Tableau PDP détaillé
  - Exemples industriels réels

- **Configuration**
  - `PDPConfig` : Dataclass complète pour les paramètres
  - Exemples : Rouleurs, Compresseurs, Usinage, PDP Table
  - Support de l'intensité de demande (low, medium, high, extreme)

- **Documentation**
  - Documentation complète avec MkDocs Material
  - Guide de démarrage rapide
  - Concepts théoriques (PDP, RL, PPO)
  - Architecture du projet
  - API Reference
  - Exemples d'utilisation

### 🔧 Corrigé

- Correction de la stratégie Chase (utilisation correcte des capacités)
- Désactivation de la double normalisation (VecNormalize)
- Normalisation des poids de récompense
- Modernisation du générateur aléatoire (np.random.default_rng)
- Activation du callback EarlyStopping
- Nettoyage des imports inutilisés

### 📚 Documentation

- README complet avec badges et exemples
- Documentation ReadTheDocs
- Docstrings pour toutes les classes et méthodes

## [0.1.0] - 2025-11-24

### Ajouté

- Première version du projet
- Environnement de base
- Agent PPO simple
- Scripts d'entraînement et d'évaluation

---

## Légende

- ✨ **Ajouté** : Nouvelles fonctionnalités
- 🔄 **Modifié** : Changements dans les fonctionnalités existantes
- 🗑️ **Déprécié** : Fonctionnalités bientôt supprimées
- 🔧 **Corrigé** : Corrections de bugs
- 🔒 **Sécurité** : Corrections de vulnérabilités
- 📚 **Documentation** : Mises à jour de la documentation
