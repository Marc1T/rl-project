import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
import json
from typing import Dict, List, Any

# Ajout du chemin racine du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_metrics_from_run(log_dir: str) -> List[Dict[str, Any]]:
    """
    Charge les métriques d'un run d'entraînement ou d'évaluation.
    Pour l'évaluation, on suppose que les métriques sont stockées dans un fichier 'evaluation_metrics.json'
    dans le répertoire du modèle.
    """
    
    metrics_file = os.path.join(log_dir, 'evaluation_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        print(f"Avertissement: Fichier de métriques non trouvé à {metrics_file}. Impossible de visualiser.")
        return []

def plot_metrics(metrics_data: List[Dict[str, Any]], title: str):
    """
    Trace les métriques clés (coût, stock, niveau de service) sur l'horizon de temps.
    """
    if not metrics_data:
        return

    # Convertir les métriques en DataFrame pour faciliter le traitement
    all_metrics = []
    for episode_metrics in metrics_data:
        for step_metric in episode_metrics:
            all_metrics.append(step_metric)
            
    df = pd.DataFrame(all_metrics)
    
    if df.empty:
        print("Aucune donnée à tracer.")
        return

    # Calculer la moyenne par période
    df_mean = df.groupby('period').mean().reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Analyse des Métriques par Période - {title}', fontsize=16)

    # 1. Coût Total
    total_cost = df_mean['costs'].apply(lambda x: x['production_cost'] + x['inventory_cost'] + x['shortage_cost'])
    axes[0].plot(df_mean['period'], total_cost, label='Coût Total Moyen', color='red')
    axes[0].set_ylabel('Coût (€)')
    axes[0].set_title('Coût Total par Période')
    axes[0].grid(True)

    # 2. Niveau de Stock
    # Pour l'instant, on suppose un seul produit (n_products=1)
    stock_level = df_mean['inventory_level'].apply(lambda x: x[0])
    axes[1].plot(df_mean['period'], stock_level, label='Stock Moyen', color='blue')
    axes[1].set_ylabel('Stock (Unités)')
    axes[1].set_title('Niveau de Stock Moyen par Période')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].grid(True)

    # 3. Niveau de Service
    axes[2].plot(df_mean['period'], df_mean['demand_fulfillment'], label='Niveau de Service Moyen', color='green')
    axes[2].set_ylabel('Niveau de Service')
    axes[2].set_title('Niveau de Service Moyen par Période')
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel('Période')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualisation des métriques d'un run d'évaluation.")
    parser.add_argument('--log_dir', type=str, required=True, help='Chemin vers le répertoire contenant les métriques (ex: répertoire du modèle RL ou un fichier de métriques baseline).')
    parser.add_argument('--title', type=str, default='Résultats d\'Évaluation', help='Titre du graphique.')
    
    args = parser.parse_args()
    
    metrics = load_metrics_from_run(args.log_dir)
    if metrics:
        plot_metrics(metrics, args.title)

if __name__ == "__main__":
    print("Pour utiliser ce script, vous devez d'abord générer un fichier 'evaluation_metrics.json' contenant les métriques par pas de temps.")
    print("Exemple d'utilisation: python scripts/visualize_metrics.py --log_dir ./models/run_1 --title 'PPO Model'")