# components/demand_generators.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from config.base_config import BaseConfig


class DemandGenerator:
    """
    Génère les scénarios de demande pour l'environnement.
    Peut générer des demandes synthétiques ou importer des cas réels.
    
    La demande est calibrée pour dépasser régulièrement la capacité normale,
    forçant le recours aux heures supplémentaires et à la sous-traitance.
    
    Utilise np.random.Generator (moderne) au lieu de np.random.seed (déprécié).
    """
    
    def __init__(self, config: BaseConfig, demand_intensity: str = "medium"):
        """
        Args:
            config: Configuration de l'environnement
            demand_intensity: Intensité de la demande
                - "low": Demande < capacité normale (facile)
                - "medium": Demande ~ capacité normale (moyen)
                - "high": Demande > capacité normale, nécessite heures supp (difficile)
                - "extreme": Demande très haute, nécessite sous-traitance (très difficile)
        """
        self.config = config
        self.demand_intensity = demand_intensity
        
        # Calculer la capacité totale pour référence
        self.regular_capacity = np.array(config.regular_capacity)
        self.overtime_capacity = np.array(config.overtime_capacity)
        self.subcontracting_capacity = np.array(config.subcontracting_capacity)
        self.total_capacity = (
            self.regular_capacity + 
            self.overtime_capacity + 
            self.subcontracting_capacity
        )
        
        # Définir la base de demande selon l'intensité
        self._set_demand_base()
        
        # RNG moderne pour reproductibilité
        self._rng: Optional[np.random.Generator] = None

    def _set_demand_base(self):
        """
        Configure les paramètres de demande selon l'intensité.
        
        Calibration basée sur la capacité TOTALE (normale + overtime + sous-traitance):
        - low: demande < capacité normale (jamais besoin d'heures supp)
        - medium: demande parfois > capacité normale (heures supp occasionnelles)
        - high: demande souvent > capacité normale (heures supp fréquentes, sous-traitance parfois)
        - extreme: demande proche de la capacité totale (tout est utilisé)
        """
        # Ratio capacité totale / capacité normale
        total_ratio = self.total_capacity / self.regular_capacity
        # En général: total_ratio ≈ 1.8 (100 + 30 + 50 = 180% de la capacité normale)
        
        intensity_params = {
            "low": {
                "base_ratio": 0.75,     # 75% de la capacité normale
                "variation": 0.15,      # ±15%
                "spike_prob": 0.1,      # 10% de chance de pic
                "spike_multiplier": 1.2 # Max: 75% * 1.15 * 1.2 = 103% → légères heures supp
            },
            "medium": {
                "base_ratio": 0.90,     # 90% de la capacité normale
                "variation": 0.20,      # ±20%
                "spike_prob": 0.15,     # 15% de chance de pic
                "spike_multiplier": 1.25 # Max: 90% * 1.2 * 1.25 = 135% → heures supp + un peu sous-trait
            },
            "high": {
                "base_ratio": 1.05,     # 105% de la capacité normale → heures supp régulières
                "variation": 0.20,      # ±20%
                "spike_prob": 0.20,     # 20% de chance de pic
                "spike_multiplier": 1.3 # Max: 105% * 1.2 * 1.3 = 164% → proche de capacité totale
            },
            "extreme": {
                "base_ratio": 1.20,     # 120% de la capacité normale
                "variation": 0.25,      # ±25%
                "spike_prob": 0.25,     # 25% de chance de pic
                "spike_multiplier": 1.35 # Max: 120% * 1.25 * 1.35 = 202% → dépasse parfois
            }
        }
        
        params = intensity_params.get(self.demand_intensity, intensity_params["high"])
        self.base_ratio = params["base_ratio"]
        self.variation = params["variation"]
        self.spike_prob = params["spike_prob"]
        self.spike_multiplier = params["spike_multiplier"]
        
        # Base de demande
        self.base_demand = self.regular_capacity * self.base_ratio

    def generate_synthetic_demands(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Génère des demandes aléatoires avec pics occasionnels.
        Calibrée pour dépasser la capacité normale régulièrement.
        
        Returns:
            np.ndarray: Demandes de forme (n_products, horizon)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
            
        n_products = self.config.n_products
        horizon = self.config.horizon
        
        demands = np.zeros((n_products, horizon), dtype=np.float32)
        
        for i in range(n_products):
            for t in range(horizon):
                # Demande de base avec variation
                base = self.base_demand[i]
                variation = self._rng.uniform(
                    1 - self.variation, 
                    1 + self.variation
                )
                demand = base * variation
                
                # Pics aléatoires (simule promotions, saisonnalité)
                if self._rng.random() < self.spike_prob:
                    demand *= self.spike_multiplier
                
                # Tendance saisonnière (optionnel)
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * t / horizon)
                demand *= seasonal_factor
                
                demands[i, t] = max(0, demand)
        
        # Log pour debug
        self._log_demand_stats(demands)
        
        return demands

    def generate_challenging_scenario(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Génère un scénario spécifiquement conçu pour être difficile.
        Inclut des pics de demande qui dépassent la capacité totale.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
            
        n_products = self.config.n_products
        horizon = self.config.horizon
        demands = np.zeros((n_products, horizon), dtype=np.float32)
        
        for i in range(n_products):
            for t in range(horizon):
                # Phase du cycle économique
                phase = t / horizon
                
                if phase < 0.25:
                    # Début: demande normale
                    base_mult = 0.9
                elif phase < 0.5:
                    # Croissance: demande croissante
                    base_mult = 1.1 + 0.3 * (phase - 0.25) / 0.25
                elif phase < 0.75:
                    # Pic: demande très haute (nécessite tout)
                    base_mult = 1.4 + self._rng.uniform(0, 0.3)
                else:
                    # Déclin: retour à la normale
                    base_mult = 1.2 - 0.3 * (phase - 0.75) / 0.25
                
                # Ajouter du bruit
                noise = self._rng.uniform(0.9, 1.1)
                demands[i, t] = self.regular_capacity[i] * base_mult * noise
        
        self._log_demand_stats(demands)
        return demands

    def _log_demand_stats(self, demands: np.ndarray):
        """Affiche les statistiques de la demande générée"""
        for i in range(demands.shape[0]):
            prod_demand = demands[i]
            reg_cap = self.regular_capacity[i]
            total_cap = self.total_capacity[i]
            
            pct_above_regular = np.mean(prod_demand > reg_cap) * 100
            pct_above_total = np.mean(prod_demand > total_cap) * 100
            
            print(f"  📦 Produit {i+1}: demande moy={prod_demand.mean():.1f}, "
                  f"cap_reg={reg_cap:.1f}, "
                  f">{reg_cap:.0f}: {pct_above_regular:.0f}%, "
                  f">{total_cap:.0f}: {pct_above_total:.0f}%")

    def import_real_demands(self, file_path: str) -> np.ndarray:
        """
        Importe les demandes à partir d'un fichier (ex: CSV).
        """
        try:
            df = pd.read_csv(file_path, index_col=0)
            demands = df.values.astype(np.float32)
            
            if demands.shape[0] != self.config.n_products:
                raise ValueError("Le nombre de produits dans le fichier ne correspond pas à la configuration.")
            if demands.shape[1] < self.config.horizon:
                print("Avertissement: Horizon trop court. Remplissage avec des zéros.")
                padding = np.zeros((self.config.n_products, self.config.horizon - demands.shape[1]))
                demands = np.hstack([demands, padding])
            elif demands.shape[1] > self.config.horizon:
                demands = demands[:, :self.config.horizon]
            
            self._log_demand_stats(demands)
            return demands
            
        except FileNotFoundError:
            print(f"Erreur: Fichier non trouvé {file_path}. Utilisation de demandes synthétiques.")
            return self.generate_synthetic_demands()
        except Exception as e:
            print(f"Erreur: {e}. Utilisation de demandes synthétiques.")
            return self.generate_synthetic_demands()
