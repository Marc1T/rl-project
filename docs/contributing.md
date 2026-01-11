# Guide de Contribution

Merci de votre intérêt pour contribuer à RLPlanif ! 🎉

## 📋 Comment Contribuer

### 1. Fork et Clone

```bash
# Fork le repository sur GitHub, puis:
git clone https://github.com/VOTRE_USERNAME/rl-project.git
cd rl-project

# Ajouter le remote upstream
git remote add upstream https://github.com/Marc1T/rl-project.git
```

### 2. Créer une Branche

```bash
# Synchroniser avec upstream
git fetch upstream
git checkout main
git merge upstream/main

# Créer une branche pour votre feature
git checkout -b feature/ma-nouvelle-feature
```

### 3. Développer

```bash
# Installer les dépendances de développement
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Si disponible

# Faire vos modifications
# ...

# Tester
python -m pytest tests/
```

### 4. Commit et Push

```bash
# Commiter avec un message clair
git add .
git commit -m "feat: ajoute support multi-produits amélioré"

# Push vers votre fork
git push origin feature/ma-nouvelle-feature
```

### 5. Créer une Pull Request

1. Allez sur GitHub
2. Cliquez "New Pull Request"
3. Sélectionnez votre branche
4. Décrivez vos modifications

## 📝 Conventions de Code

### Style Python

- Suivre PEP 8
- Utiliser des docstrings Google style
- Maximum 100 caractères par ligne
- Utiliser des type hints

```python
def ma_fonction(param1: str, param2: int = 10) -> bool:
    """
    Description courte de la fonction.
    
    Args:
        param1: Description du paramètre 1
        param2: Description du paramètre 2
    
    Returns:
        Description de la valeur retournée
    
    Raises:
        ValueError: Si param2 est négatif
    """
    if param2 < 0:
        raise ValueError("param2 doit être positif")
    return True
```

### Conventions de Commit

Utiliser le format [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[body optionnel]

[footer optionnel]
```

Types :
- `feat`: Nouvelle fonctionnalité
- `fix`: Correction de bug
- `docs`: Documentation
- `style`: Formatage (pas de changement de code)
- `refactor`: Refactoring
- `test`: Ajout de tests
- `chore`: Maintenance

Exemples :
```
feat(env): ajoute support pour contraintes de setup
fix(reward): corrige le calcul du service level
docs(readme): met à jour les instructions d'installation
```

## 🧪 Tests

### Exécuter les Tests

```bash
# Tous les tests
python -m pytest tests/

# Avec coverage
python -m pytest tests/ --cov=. --cov-report=html

# Un fichier spécifique
python -m pytest tests/test_components.py -v
```

### Écrire des Tests

```python
import pytest
from components.demand_generators import DemandGenerator
from config import PDPConfig

class TestDemandGenerator:
    @pytest.fixture
    def config(self):
        return PDPConfig(demand_mean=[80], demand_std=[15])
    
    def test_generate_positive(self, config):
        """La demande générée doit être positive."""
        gen = DemandGenerator(config)
        demand = gen.generate(0)
        assert demand[0] >= 0
    
    def test_generate_shape(self, config):
        """La demande doit avoir la bonne forme."""
        gen = DemandGenerator(config)
        demand = gen.generate(0)
        assert demand.shape == (1,)
```

## 📁 Structure du Projet

```
rl-project/
├── agents/              # Agents RL
├── components/          # Composants modulaires
├── config/              # Configurations
├── environments/        # Environnements Gymnasium
├── scripts/             # Scripts CLI
├── tests/               # Tests unitaires
├── docs/                # Documentation
└── app.py              # Interface Streamlit
```

## 🎯 Domaines de Contribution

### Priorité Haute

- [ ] Support multi-produits avancé
- [ ] Nouvelles stratégies baseline
- [ ] Amélioration des visualisations
- [ ] Tests unitaires supplémentaires

### Priorité Moyenne

- [ ] Export des résultats (Excel, PDF)
- [ ] Nouveaux patterns de demande
- [ ] Intégration avec d'autres algorithmes RL
- [ ] Documentation en anglais

### Idées Futures

- [ ] Interface API REST
- [ ] Déploiement cloud
- [ ] Optimisation multi-objectif
- [ ] Apprentissage continu

## ❓ Questions

Pour toute question :

1. Vérifiez d'abord la [documentation](https://rl-project.readthedocs.io/)
2. Cherchez dans les [issues existantes](https://github.com/Marc1T/rl-project/issues)
3. Créez une nouvelle issue si nécessaire

## 📄 Licence

En contribuant, vous acceptez que vos contributions soient sous licence MIT.

---

**Merci de contribuer à RLPlanif !** 🙏
