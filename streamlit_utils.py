# streamlit_utils.py - Fonctions utilitaires pour Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any
import json

def plot_episode_metrics(metrics: List[Dict], title: str = "Métriques de l'Épisode"):
    """
    Crée des visualisations pour les métriques d'un épisode
    
    Args:
        metrics: Liste des métriques par période
        title: Titre du graphique
    """
    periods = [m['period'] for m in metrics]
    stocks = [m['inventory_level'][0] for m in metrics]
    productions = [m['total_production'] for m in metrics]
    service_levels = [m['demand_fulfillment'] for m in metrics]
    
    # Créer les subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Évolution du Stock', 'Production', 
                       'Niveau de Service', 'Coûts Cumulés'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Stock
    fig.add_trace(
        go.Scatter(x=periods, y=stocks, mode='lines+markers', 
                  name='Stock', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                 row=1, col=1, annotation_text="Rupture")
    
    # Production
    fig.add_trace(
        go.Bar(x=periods, y=productions, name='Production',
              marker_color='lightblue'),
        row=1, col=2
    )
    
    # Service Level
    fig.add_trace(
        go.Bar(x=periods, y=service_levels, name='Service Level',
              marker_color='green'),
        row=2, col=1
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                 row=2, col=1, annotation_text="Cible")
    
    # Coûts cumulés
    costs_prod = [m['costs']['production_cost'] for m in metrics]
    costs_inv = [m['costs']['inventory_cost'] for m in metrics]
    costs_short = [m['costs']['shortage_cost'] for m in metrics]
    
    cum_prod = np.cumsum(costs_prod)
    cum_inv = np.cumsum(costs_inv)
    cum_short = np.cumsum(costs_short)
    
    fig.add_trace(
        go.Scatter(x=periods, y=cum_prod, mode='lines', 
                  name='Production', line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=periods, y=cum_inv, mode='lines',
                  name='Stockage', line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=periods, y=cum_short, mode='lines',
                  name='Rupture', line=dict(width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text=title,
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="Période", row=2, col=1)
    fig.update_xaxes(title_text="Période", row=2, col=2)
    fig.update_yaxes(title_text="Stock", row=1, col=1)
    fig.update_yaxes(title_text="Quantité", row=1, col=2)
    fig.update_yaxes(title_text="Service Level", row=2, col=1)
    fig.update_yaxes(title_text="Coût Cumulé", row=2, col=2)
    
    return fig

def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Crée un tableau de comparaison formaté
    
    Args:
        results: Dictionnaire de résultats par stratégie
    
    Returns:
        DataFrame formaté
    """
    df = pd.DataFrame(results).T
    df = df.round(2)
    
    # Renommer les colonnes
    column_mapping = {
        'reward': 'Reward',
        'cost': 'Coût Total',
        'service': 'Service Level',
        'stock': 'Stock Final'
    }
    df = df.rename(columns=column_mapping)
    
    return df

def highlight_best(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.DataFrame:
    """
    Met en évidence la meilleure valeur d'une colonne
    
    Args:
        df: DataFrame à formater
        column: Colonne à examiner
        ascending: True si la plus petite valeur est meilleure
    
    Returns:
        DataFrame stylisé
    """
    def highlight_max(s):
        if ascending:
            is_best = s == s.min()
        else:
            is_best = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_best]
    
    return df.style.apply(highlight_max, subset=[column])

def load_saved_models() -> List[Dict]:
    """
    Charge la liste des modèles sauvegardés
    
    Returns:
        Liste de dictionnaires avec les infos des modèles
    """
    models_dir = Path("./models")
    models = []
    
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                best_model = model_dir / "best_model.zip"
                final_model = model_dir / "final_model.zip"
                
                if best_model.exists() or final_model.exists():
                    models.append({
                        'name': model_dir.name,
                        'path': str(best_model if best_model.exists() else final_model),
                        'date': model_dir.stat().st_mtime
                    })
    
    # Trier par date (plus récent en premier)
    models.sort(key=lambda x: x['date'], reverse=True)
    
    return models

def save_config_to_json(config, filename: str):
    """
    Sauvegarde une configuration en JSON
    
    Args:
        config: Configuration à sauvegarder
        filename: Nom du fichier de sortie
    """
    config_dict = {
        'n_products': config.n_products,
        'horizon': config.horizon,
        'regular_capacity': config.regular_capacity,
        'overtime_capacity': config.overtime_capacity,
        'subcontracting_capacity': config.subcontracting_capacity,
        'initial_stock': config.initial_stock,
        'max_stock': config.max_stock,
        'regular_cost': config.regular_cost,
        'overtime_cost': config.overtime_cost,
        'subcontracting_cost': config.subcontracting_cost,
        'holding_cost': config.holding_cost,
        'shortage_cost': config.shortage_cost,
        'service_level_target': config.service_level_target
    }
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config_from_json(filename: str):
    """
    Charge une configuration depuis un fichier JSON
    
    Args:
        filename: Chemin du fichier JSON
    
    Returns:
        Configuration chargée
    """
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    from config.environment_configs import PDPEnvironmentConfig
    return PDPEnvironmentConfig(**config_dict)

def format_number(num: float, decimals: int = 2) -> str:
    """
    Formate un nombre avec des séparateurs
    
    Args:
        num: Nombre à formater
        decimals: Nombre de décimales
    
    Returns:
        Chaîne formatée
    """
    return f"{num:,.{decimals}f}".replace(",", " ")

def create_metric_card(label: str, value: Any, delta: Any = None, 
                      delta_color: str = "normal"):
    """
    Crée une carte de métrique stylisée
    
    Args:
        label: Label de la métrique
        value: Valeur à afficher
        delta: Variation (optionnel)
        delta_color: Couleur du delta
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

def plot_training_history(log_dir: str):
    """
    Affiche l'historique d'entraînement depuis les logs
    
    Args:
        log_dir: Chemin vers le répertoire des logs
    """
    # Cette fonction pourrait lire les logs TensorBoard ou Monitor
    # et créer des visualisations
    pass

def export_results_to_excel(results: Dict, filename: str):
    """
    Exporte les résultats vers un fichier Excel
    
    Args:
        results: Dictionnaire de résultats
        filename: Nom du fichier de sortie
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Résumé
        summary_df = pd.DataFrame(results).T
        summary_df.to_excel(writer, sheet_name='Résumé')
        
        # Détails par stratégie (si disponible)
        # ... ajouter d'autres feuilles si nécessaire

def validate_config(config) -> List[str]:
    """
    Valide une configuration et retourne les erreurs éventuelles
    
    Args:
        config: Configuration à valider
    
    Returns:
        Liste des messages d'erreur (vide si tout est OK)
    """
    errors = []
    
    if config.n_products <= 0:
        errors.append("Le nombre de produits doit être > 0")
    
    if config.horizon <= 0:
        errors.append("L'horizon doit être > 0")
    
    if any(c <= 0 for c in config.regular_capacity):
        errors.append("Les capacités régulières doivent être > 0")
    
    if any(c < 0 for c in config.overtime_capacity):
        errors.append("Les capacités heures sup doivent être >= 0")
    
    if config.service_level_target < 0 or config.service_level_target > 1:
        errors.append("Le niveau de service cible doit être entre 0 et 1")
    
    return errors

def show_progress_with_eta(current: int, total: int, start_time: float):
    """
    Affiche une barre de progression avec temps estimé
    
    Args:
        current: Étape actuelle
        total: Nombre total d'étapes
        start_time: Timestamp de début
    """
    import time
    
    progress = current / total
    elapsed = time.time() - start_time
    
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
    else:
        eta_str = "Calcul..."
    
    st.progress(progress, text=f"Progression: {current}/{total} - ETA: {eta_str}")

def create_download_link(df: pd.DataFrame, filename: str, text: str = "Télécharger"):
    """
    Crée un lien de téléchargement pour un DataFrame
    
    Args:
        df: DataFrame à télécharger
        filename: Nom du fichier
        text: Texte du lien
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=text,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
