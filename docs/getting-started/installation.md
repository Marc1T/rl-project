# Installation

## Prérequis

Avant d'installer RLPlanif, assurez-vous d'avoir :

- **Python 3.9+** (recommandé : 3.11)
- **Conda** ou **pip** pour la gestion des packages
- **Git** pour cloner le repository

## Installation avec Conda (Recommandé)

```bash
# 1. Cloner le repository
git clone https://github.com/NANKOULI/rlplanif.git
cd rlplanif

# 2. Créer l'environnement conda
conda create -n rl-sb python=3.11 -y
conda activate rl-sb

# 3. Installer PyTorch (avec CUDA si GPU disponible)
# Pour CPU uniquement :
pip install torch torchvision torchaudio

# Pour GPU CUDA 11.8 :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Installer les dépendances
pip install -r requirements.txt
```

## Installation avec pip

```bash
# 1. Cloner le repository
git clone https://github.com/NANKOULI/rlplanif.git
cd rlplanif

# 2. Créer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Dépendances Principales

| Package | Version | Description |
|---------|---------|-------------|
| `stable-baselines3` | ≥2.2.1 | Algorithmes RL (PPO) |
| `gymnasium` | ≥0.29.1 | Environnements RL |
| `torch` | ≥2.1.0 | Backend Deep Learning |
| `numpy` | ≥1.24.0 | Calcul numérique |
| `pandas` | ≥2.0.0 | Manipulation de données |
| `streamlit` | ≥1.28.0 | Interface web |
| `plotly` | ≥5.18.0 | Visualisations |
| `tensorboard` | ≥2.15.0 | Monitoring entraînement |

## Vérification de l'installation

```bash
# Tester l'environnement
python scripts/test_env.py

# Lancer les tests unitaires
python -m pytest tests/
```

!!! success "Installation réussie"
    Si les tests passent, vous êtes prêt à utiliser RLPlanif !

## Problèmes Courants

??? warning "Erreur CUDA / GPU non détecté"
    Si vous avez un GPU NVIDIA mais qu'il n'est pas détecté :
    
    1. Vérifiez que les drivers NVIDIA sont à jour
    2. Installez la version CUDA correspondante de PyTorch
    3. Testez avec : `python -c "import torch; print(torch.cuda.is_available())"`

??? warning "Conflit de versions"
    En cas de conflit de packages :
    
    ```bash
    # Créer un environnement propre
    conda create -n rl-sb python=3.11 -y
    conda activate rl-sb
    pip install -r requirements.txt --no-cache-dir
    ```

## Prochaine Étape

➡️ [Premier Entraînement](quickstart.md)
