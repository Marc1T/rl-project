# components/demand_generators.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from config.base_config import BaseConfig

class DemandGenerator:
    """
    Génère les scénarios de demande pour l'environnement.
    Peut générer des demandes synthétiques ou importer des cas réels.
    """
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.base_demand = np.array(config.regular_capacity) * 0.8 # Base pour la génération

    def generate_synthetic_demands(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Génère des demandes aléatoires avec une variation autour d'une base.
        Retourne un tableau numpy de forme (n_products, horizon).
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_products = self.config.n_products
        horizon = self.config.horizon
        
        demands = np.zeros((n_products, horizon), dtype=np.float32)
        
        for i in range(n_products):
            demands[i, :] = np.random.uniform(
                low=0.7 * self.base_demand[i],
                high=1.3 * self.base_demand[i],
                size=horizon
            )
            
        return demands

    def import_real_demands(self, file_path: str) -> np.ndarray:
        """
        Importe les demandes à partir d'un fichier (ex: CSV).
        Nécessite que le fichier soit structuré correctement.
        (Implémentation simplifiée pour l'instant)
        """
        try:
            # Supposons que le fichier a les produits en lignes et les périodes en colonnes
            df = pd.read_csv(file_path, index_col=0)
            demands = df.values.astype(np.float32)
            
            # Vérification de la cohérence
            if demands.shape[0] != self.config.n_products:
                raise ValueError("Le nombre de produits dans le fichier ne correspond pas à la configuration.")
            if demands.shape[1] < self.config.horizon:
                print("Avertissement: L'horizon de demande est plus court que l'horizon de l'environnement. Remplissage avec des zéros.")
                padding = np.zeros((self.config.n_products, self.config.horizon - demands.shape[1]))
                demands = np.hstack([demands, padding])
            elif demands.shape[1] > self.config.horizon:
                demands = demands[:, :self.config.horizon]
                
            return demands
            
        except FileNotFoundError:
            print(f"Erreur: Fichier de demande non trouvé à {file_path}. Utilisation de demandes synthétiques.")
            return self.generate_synthetic_demands()
        except Exception as e:
            print(f"Erreur lors de l'importation des demandes: {e}. Utilisation de demandes synthétiques.")
            return self.generate_synthetic_demands()
