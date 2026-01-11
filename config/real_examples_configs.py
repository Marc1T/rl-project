# config/real_examples_configs.py
"""
Configurations basées sur les exemples réels d'exercices du cours de gestion de production
"""

from dataclasses import dataclass
import numpy as np
from config.environment_configs import PDPEnvironmentConfig

# ========== EXEMPLE 1: Fabrication de Rouleurs ==========
# Production: 400 000 rouleurs sur 11 mois (août chômé)
# Effectif: 200 opérateurs, productivité +10% année N+1
# Demandes: [16, 15, 10, 10, 10, 10, 50, 50, 80, 120, 50, 20] x 1000

def get_rouleurs_config() -> PDPEnvironmentConfig:
    """
    Configuration pour l'exemple Rouleurs (Image 1)
    """
    # Calcul de la productivité
    # Année N: 400 000 / 11 mois = 36 364 rouleurs/mois
    # Année N+1: +10% => 40 000 rouleurs/mois environ
    
    # Avec 200 opérateurs, capacité régulière = 40 000/mois
    regular_capacity = 40.0  # en milliers
    overtime_capacity = 12.0  # 30% de la capacité régulière
    subcontracting_capacity = 20.0  # 50% de la capacité régulière
    
    # Demandes en milliers
    demands = np.array([16, 15, 10, 10, 10, 10, 50, 50, 80, 120, 50, 20], dtype=np.float32)
    
    # Stock minimum: demi-mois de vente
    # Pour août: 1.5 mois => 75 (50*1.5)
    initial_stock = 25.0  # Stock initial raisonnable
    
    return PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        regular_capacity=[regular_capacity],
        overtime_capacity=[overtime_capacity],
        subcontracting_capacity=[subcontracting_capacity],
        initial_stock=[initial_stock],
        max_stock=[200.0],  # Stock max raisonnable
        
        # Coûts (en unités monétaires par millier)
        regular_cost=[10.0],
        overtime_cost=[15.0],
        subcontracting_cost=[20.0],
        holding_cost=[2.0],
        shortage_cost=[50.0],
        
        normalize_observations=True
    )


# ========== EXEMPLE 2: Tableau PDP avec Stock Initial ==========
# Demandes: [10000, 8000, 13000, 26000, 32000, 34000, 17000, 24000, 38000, 40000, 20000, 16000]

def get_pdp_table_config() -> PDPEnvironmentConfig:
    """
    Configuration pour l'exemple du tableau PDP (Image 2)
    """
    # Capacité de production environ 28800/mois (basé sur les données)
    regular_capacity = 28.8  # en milliers
    overtime_capacity = 8.6  # 30% environ
    subcontracting_capacity = 14.4  # 50% environ
    
    # Demandes en milliers
    demands = np.array([10, 8, 13, 26, 32, 34, 17, 24, 38, 40, 20, 16], dtype=np.float32)
    
    # Stock initial visible dans le tableau
    initial_stock = 13.0  # 1300 dans le tableau original
    
    return PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        regular_capacity=[regular_capacity],
        overtime_capacity=[overtime_capacity],
        subcontracting_capacity=[subcontracting_capacity],
        initial_stock=[initial_stock],
        max_stock=[300.0],
        
        # Coûts
        regular_cost=[10.0],
        overtime_cost=[15.0],
        subcontracting_cost=[20.0],
        holding_cost=[2.0],
        shortage_cost=[100.0],  # Coût élevé pour éviter ruptures
        
        normalize_observations=True
    )


# ========== EXEMPLE 3: Compresseurs pour Climatiseurs ==========
# Capacité: 250 unités/jour, 8h/jour
# Heures supp max: 30% des heures normales
# Saisonnalité de 6 mois

def get_compresseurs_config() -> PDPEnvironmentConfig:
    """
    Configuration pour l'exemple Compresseurs (Image 3)
    """
    # Demandes mensuelles
    demands = np.array([3200, 4200, 4800, 7000, 7500, 3500], dtype=np.float32)
    
    # Jours ouvrables par mois
    working_days = np.array([22, 20, 22, 21, 19, 21])
    
    # Capacité régulière: 250 unités/jour * jours ouvrables
    # On prend une moyenne de 21 jours
    regular_capacity = 250 * 21  # = 5250 unités/mois en moyenne
    
    # Heures supplémentaires: 30% max
    overtime_capacity = regular_capacity * 0.3  # = 1575 unités
    
    # Sous-traitance
    subcontracting_capacity = regular_capacity * 0.5  # = 2625 unités
    
    return PDPEnvironmentConfig(
        n_products=1,
        horizon=6,  # 6 mois
        regular_capacity=[regular_capacity],
        overtime_capacity=[overtime_capacity],
        subcontracting_capacity=[subcontracting_capacity],
        initial_stock=[1000.0],  # Stock initial raisonnable
        max_stock=[8000.0],
        
        # Coûts
        regular_cost=[10.0],
        overtime_cost=[15.0],
        subcontracting_cost=[20.0],
        holding_cost=[2.0],
        shortage_cost=[100.0],
        
        normalize_observations=True
    )


# ========== EXEMPLE 4: Atelier d'Usinage (Familles A et B) ==========
# Capacité: 700 heures/mois (régulières) + 100 heures supp
# Famille A: 100 DH, temps usinage 1h
# Famille B: 300 DH, temps usinage 1h
# Coût stockage: 2%/mois

def get_usinage_config() -> PDPEnvironmentConfig:
    """
    Configuration pour l'exemple Atelier d'usinage (Image 4)
    Multi-produits: Familles A et B
    """
    # Demandes pour 6 mois (Janvier à Juin)
    demands_A = np.array([50, 70, 140, 150, 280, 250], dtype=np.float32)
    demands_B = np.array([50, 730, 560, 380, 520, 460], dtype=np.float32)
    
    # Capacité en heures
    regular_capacity_hours = 700.0
    overtime_capacity_hours = 100.0
    
    # Chaque pièce prend 1 heure, donc capacité = heures disponibles
    # On divise entre les 2 produits
    regular_per_product = regular_capacity_hours / 2
    overtime_per_product = overtime_capacity_hours / 2
    
    return PDPEnvironmentConfig(
        n_products=2,
        horizon=6,
        regular_capacity=[regular_per_product, regular_per_product],
        overtime_capacity=[overtime_per_product, overtime_per_product],
        subcontracting_capacity=[200.0, 200.0],  # Sous-traitance possible
        initial_stock=[100.0, 100.0],
        max_stock=[1500.0, 1500.0],
        
        # Coûts basés sur les coûts de revient
        regular_cost=[100.0, 300.0],  # Coût de fabrication
        overtime_cost=[125.0, 375.0],  # +25% pour heures supp
        subcontracting_cost=[150.0, 450.0],  # +50% pour sous-traitance
        holding_cost=[2.0, 6.0],  # 2% par mois du coût de revient
        shortage_cost=[200.0, 600.0],  # 2x le coût de revient
        
        normalize_observations=True
    )


# ========== Fonction utilitaire ==========

def get_example_config(example_name: str) -> PDPEnvironmentConfig:
    """
    Retourne la configuration pour un exemple donné
    
    Args:
        example_name: 'rouleurs', 'pdp_table', 'compresseurs', ou 'usinage'
    """
    configs = {
        'rouleurs': get_rouleurs_config,
        'pdp_table': get_pdp_table_config,
        'compresseurs': get_compresseurs_config,
        'usinage': get_usinage_config
    }
    
    if example_name not in configs:
        raise ValueError(f"Exemple inconnu: {example_name}. "
                        f"Disponibles: {list(configs.keys())}")
    
    return configs[example_name]()


def save_example_demands(example_name: str, output_path: str):
    """
    Sauvegarde les demandes d'un exemple dans un fichier CSV
    pour utilisation avec DemandGenerator
    """
    import pandas as pd
    
    config = get_example_config(example_name)
    
    if example_name == 'rouleurs':
        demands = [16, 15, 10, 10, 10, 10, 50, 50, 80, 120, 50, 20]
    elif example_name == 'pdp_table':
        demands = [10, 8, 13, 26, 32, 34, 17, 24, 38, 40, 20, 16]
    elif example_name == 'compresseurs':
        demands = [3200, 4200, 4800, 7000, 7500, 3500]
    elif example_name == 'usinage':
        # Multi-produits
        demands_A = [50, 70, 140, 150, 280, 250]
        demands_B = [50, 730, 560, 380, 520, 460]
        df = pd.DataFrame({
            'Product_A': demands_A,
            'Product_B': demands_B
        })
        df.to_csv(output_path, index=False)
        print(f"✅ Demandes sauvegardées: {output_path}")
        return
    
    # Pour les exemples mono-produit
    df = pd.DataFrame({'Product_0': demands})
    df.to_csv(output_path, index=False)
    print(f"✅ Demandes sauvegardées: {output_path}")


# ========== Exemple d'utilisation ==========
if __name__ == "__main__":
    # Afficher les configurations disponibles
    print("📋 EXEMPLES DISPONIBLES:\n")
    
    examples = ['rouleurs', 'pdp_table', 'compresseurs', 'usinage']
    
    for ex in examples:
        print(f"--- {ex.upper()} ---")
        config = get_example_config(ex)
        print(f"   Horizon: {config.horizon} périodes")
        print(f"   Produits: {config.n_products}")
        print(f"   Capacité régulière: {config.regular_capacity}")
        print(f"   Stock initial: {config.initial_stock}")
        print()
