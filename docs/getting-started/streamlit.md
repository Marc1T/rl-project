# Interface Streamlit

RLPlanif dispose d'une interface web interactive construite avec Streamlit.

## 🚀 Lancement

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

## 📋 Pages Disponibles

### 🏠 Accueil

Page d'accueil avec :

- Présentation du projet
- Statistiques du système (modèles, configurations)
- Guide rapide de démarrage
- Liste des modèles récemment entraînés

### ⚙️ Configuration

Trois modes de configuration disponibles :

=== "Exemple Pré-configuré"

    Sélectionnez parmi 4 exemples industriels :
    
    - **Rouleurs** : 12 périodes, 1 produit
    - **PDP Table** : 6 périodes, 1 produit
    - **Compresseurs** : 8 périodes, 1 produit
    - **Usinage** : 12 périodes, 1 produit

=== "Configuration Personnalisée"

    Définissez tous les paramètres :
    
    - Nombre de produits et horizon
    - Capacités de production
    - Coûts (production, stockage, rupture)
    - Paramètres de demande

=== "Charger JSON"

    Importez une configuration depuis un fichier JSON :
    
    ```json
    {
      "n_products": 1,
      "horizon": 12,
      "regular_capacity": [100],
      "overtime_capacity": [30],
      "costs": {...}
    }
    ```

### 🏋️ Entraînement PPO

Interface d'entraînement avec :

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| Timesteps | Nombre total de pas | 50 000 |
| Learning Rate | Taux d'apprentissage | 3e-4 |
| N Steps | Pas par rollout | 2048 |
| Batch Size | Taille des batches | 64 |
| Gamma | Facteur de discount | 0.99 |

**Fonctionnalités :**

- ✅ Barre de progression en temps réel
- ✅ Logs d'entraînement
- ✅ Sauvegarde automatique du meilleur modèle
- ✅ Callback EarlyStopping

### 📊 Évaluation

Comparez votre modèle PPO aux stratégies baseline :

1. Sélectionnez un modèle entraîné
2. Choisissez les stratégies à comparer
3. Définissez le nombre d'épisodes
4. Analysez les résultats

**Métriques affichées :**

- Récompense totale
- Coût total
- Niveau de service
- Niveau de stock

### 📈 Visualisation

Graphiques interactifs avec Plotly :

- **Production par période** : Régulière, HS, Sous-traitance
- **Demande vs Production** : Comparaison temporelle
- **Évolution des stocks** : Niveaux par période
- **Structure des coûts** : Répartition par catégorie

### 📋 Tableau PDP

Tableau de production détaillé format industriel :

| Indicateur | P1 | P2 | P3 | ... |
|------------|----|----|----| --- |
| 📦 Demande | 80 | 95 | 110 | ... |
| 📈 Production | 100 | 100 | 100 | ... |
| 🔴 Stock Final | 70 | 75 | 65 | ... |
| 💰 Coût Total | 450 | 480 | 520 | ... |

**Export disponible en CSV**

### 🔬 Exemples Réels

Exécution rapide sur les cas industriels pré-configurés :

1. Sélectionnez un exemple
2. Choisissez la stratégie (PPO ou baseline)
3. Visualisez les résultats

## ⌨️ Raccourcis

| Raccourci | Action |
|-----------|--------|
| `R` | Rerun l'application |
| `C` | Effacer le cache |

## 🎨 Personnalisation du Thème

Créez un fichier `.streamlit/config.toml` :

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#1a1a2e"
font = "sans serif"
```

## Prochaine Étape

➡️ [Concepts : Plan Directeur de Production](../concepts/pdp.md)
