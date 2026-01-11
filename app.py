# app.py - Interface Streamlit pour PDP RL avec PPO
# Version améliorée avec gestion d'erreurs et interface plus robuste

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json
from pathlib import Path
import traceback

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.environment_configs import PDPEnvironmentConfig
from config.training_configs import PPOTrainingConfig
from config.real_examples_configs import get_example_config, save_example_demands
from environments.env_registry import EnvironmentRegistry
from agents.baseline_strategies import BASELINE_STRATEGIES
from agents.ppo_trainer import PPOTrainer

# Configuration de la page
st.set_page_config(
    page_title="RLPlanif - PDP Intelligent",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé amélioré
st.markdown("""
    <style>
    /* ========== VARIABLES DE COULEUR ========== */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #00c853;
        --warning-color: #ff9800;
        --danger-color: #f44336;
        --dark-bg: #1a1a2e;
        --light-bg: #f8f9fa;
    }
    
    /* ========== HEADER PRINCIPAL ========== */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* ========== CARTES MÉTRIQUES ========== */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-card h3 {
        color: #667eea;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
    }
    
    .metric-card p, .metric-card li {
        color: #555;
        font-size: 0.95rem;
    }
    
    /* ========== CARTES FEATURES ========== */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.25);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* ========== HERO SECTION ========== */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .hero-section h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .hero-section p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* ========== STATS CARDS ========== */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* ========== BADGES ========== */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-success {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    /* ========== BOXES ========== */
    .success-box {
        padding: 1.2rem;
        border-radius: 0.8rem;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        color: #155724;
    }
    
    .warning-box {
        padding: 1.2rem;
        border-radius: 0.8rem;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    
    .info-box {
        padding: 1.2rem;
        border-radius: 0.8rem;
        background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%);
        border-left: 4px solid #007bff;
        color: #004085;
    }
    
    /* ========== TIMELINE ========== */
    .timeline-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.5rem;
    }
    
    .timeline-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
        flex-shrink: 0;
    }
    
    .timeline-content h4 {
        margin: 0 0 0.3rem 0;
        color: #333;
    }
    
    .timeline-content p {
        margin: 0;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* ========== BOUTONS ========== */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    
    /* ========== DATAFRAME ========== */
    .dataframe {
        font-size: 0.9rem;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* ========== PROGRESS ========== */
    .custom-progress {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        overflow: hidden;
    }
    
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation de la session state
def init_session_state():
    """Initialise toutes les variables de session"""
    defaults = {
        'trained_models': [],
        'current_config': None,
        'training_in_progress': False,
        'evaluation_results': None,
        'detailed_metrics': None,
        'last_error': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Titre principal avec animation
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <p class="main-header">🏭 RLPlanif</p>
    <p class="sub-header">Plan Directeur de Production Intelligent avec Apprentissage par Renforcement</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Navigation améliorée
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
    <span style="font-size: 2.5rem;">🏭</span>
    <h2 style="margin: 0.5rem 0 0 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        RLPlanif
    </h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📋 Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Accueil", "⚙️ Configuration", "🏋️ Entraînement PPO", 
     "📊 Évaluation", "📈 Visualisation", "📋 Tableau PDP", "🔬 Exemples Réels"],
    label_visibility="collapsed"
)

# Afficher la config actuelle dans la sidebar
if st.session_state.current_config:
    st.sidebar.divider()
    st.sidebar.markdown("### 📌 Config Active")
    cfg = st.session_state.current_config
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 0.5rem; 
                border-left: 3px solid #667eea; font-size: 0.9rem; color: #fff;">
        <div>🎲 <strong>Produits:</strong> {cfg.n_products}</div>
        <div>📅 <strong>Horizon:</strong> {cfg.horizon}</div>
        <div>⚙️ <strong>Cap. régulière:</strong> {cfg.regular_capacity[0]}</div>
    </div>
    """, unsafe_allow_html=True)

# Footer sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem; padding: 1rem 0;">
    <div>Made with ❤️ for Production Planning</div>
    <div style="margin-top: 0.3rem;">PPO • Stable-Baselines3 • Gymnasium</div>
</div>
""", unsafe_allow_html=True)


# ============================
# FONCTIONS UTILITAIRES
# ============================
def create_detailed_production_table(metrics: list, config, strategy_name: str) -> pd.DataFrame:
    """
    Crée un tableau de production détaillé type PDP
    """
    n_periods = len(metrics)
    periods = [f"P{i+1}" for i in range(n_periods)]
    
    # Extraire les données avec gestion d'erreurs
    try:
        demands = [m['raw_metrics']['current_demand'][0] for m in metrics]
        stock_avant = [m['raw_metrics']['stock_before_prod'][0] for m in metrics]
        stock_apres_prod = [m['raw_metrics']['stock_after_prod'][0] for m in metrics]
        stock_final = [m['inventory_level'][0] for m in metrics]
        production = [m['total_production'] for m in metrics]
        demand_satisfied = [m['raw_metrics']['demand_satisfied'][0] for m in metrics]
        shortage = [m['raw_metrics']['shortage'][0] for m in metrics]
        service_level = [m['demand_fulfillment'] for m in metrics]
        
        # Coûts
        cost_prod = [m['costs']['production_cost'] for m in metrics]
        cost_stock = [m['costs']['inventory_cost'] for m in metrics]
        cost_rupture = [m['costs']['shortage_cost'] for m in metrics]
        cost_total = [sum(m['costs'].values()) for m in metrics]
        
        # Cumuls
        cumul_prod = np.cumsum(production)
        cumul_demand = np.cumsum(demands)
        cumul_cost = np.cumsum(cost_total)
        
    except KeyError as e:
        st.error(f"Erreur de structure des métriques: {e}")
        return pd.DataFrame()
    
    # Créer le DataFrame
    data = {
        '📊 Indicateur': [
            '📦 Demande',
            '📈 Production',
            '📊 Cumul Production',
            '📊 Cumul Demande',
            '🔵 Stock Début',
            '🟢 Stock Après Prod',
            '🔴 Stock Final',
            '✅ Demande Satisfaite',
            '❌ Rupture',
            '🎯 Service Level',
            '💰 Coût Production',
            '💰 Coût Stockage',
            '💰 Coût Rupture',
            '💰 Coût Total Période',
            '📈 Cumul Coûts'
        ]
    }
    
    for i in range(n_periods):
        data[periods[i]] = [
            f"{demands[i]:.0f}",
            f"{production[i]:.0f}",
            f"{cumul_prod[i]:.0f}",
            f"{cumul_demand[i]:.0f}",
            f"{stock_avant[i]:.0f}",
            f"{stock_apres_prod[i]:.0f}",
            f"{stock_final[i]:.0f}",
            f"{demand_satisfied[i]:.0f}",
            f"{shortage[i]:.0f}",
            f"{service_level[i]:.1%}",
            f"{cost_prod[i]:.0f}",
            f"{cost_stock[i]:.0f}",
            f"{cost_rupture[i]:.0f}",
            f"{cost_total[i]:.0f}",
            f"{cumul_cost[i]:.0f}"
        ]
    
    return pd.DataFrame(data)


def get_available_models() -> list:
    """Retourne la liste des modèles disponibles"""
    models_dir = Path("./models")
    if not models_dir.exists():
        return []
    
    models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            best = model_dir / "best_model.zip"
            final = model_dir / "final_model.zip"
            best_folder = model_dir / "best_model"
            
            if best.exists():
                models.append(str(best))
            elif (best_folder / "model.zip").exists():
                models.append(str(best_folder / "model.zip"))
            elif final.exists():
                models.append(str(final))
    
    return sorted(models, reverse=True)


def evaluate_ppo_model(model_path: str, config, n_episodes: int = 10):
    """Évalue un modèle PPO et retourne les résultats détaillés"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    
    def make_env():
        return EnvironmentRegistry.create('strategic', config)
    
    eval_env = DummyVecEnv([make_env])
    
    # Charger VecNormalize si disponible
    model_dir = Path(model_path).parent
    vec_normalize_path = model_dir / "vec_normalize.pkl"
    if not vec_normalize_path.exists():
        vec_normalize_path = model_dir.parent / "vec_normalize.pkl"
    
    if vec_normalize_path.exists():
        eval_env = VecNormalize.load(str(vec_normalize_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    
    model = PPO.load(model_path, env=eval_env)
    
    results = {
        'rewards': [],
        'service_levels': [],
        'costs': [],
        'stocks': [],
        'detailed_metrics': []
    }
    
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        total_cost = 0
        service_levels = []
        episode_metrics = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            
            reward = rewards[0]
            info = infos[0]
            done = dones[0]
            
            total_reward += reward
            total_cost += sum(info['costs'].values())
            service_levels.append(info['demand_fulfillment'])
            episode_metrics.append(info)
        
        results['rewards'].append(total_reward)
        results['costs'].append(total_cost)
        results['service_levels'].append(np.mean(service_levels))
        results['stocks'].append(info['inventory_level'][0])
        results['detailed_metrics'].append(episode_metrics)
    
    return results


def evaluate_baseline(strategy_name: str, config, n_episodes: int = 10):
    """Évalue une stratégie baseline"""
    StrategyClass = BASELINE_STRATEGIES[strategy_name]
    
    results = {
        'rewards': [],
        'costs': [],
        'service_levels': [],
        'all_metrics': []
    }
    
    for _ in range(n_episodes):
        env = EnvironmentRegistry.create('strategic', config)
        strategy = StrategyClass(env)
        total_reward, info = strategy.run_episode()
        metrics = info['metrics']
        
        total_cost = sum(
            m['costs']['production_cost'] + 
            m['costs']['inventory_cost'] + 
            m['costs']['shortage_cost']
            for m in metrics
        )
        
        results['rewards'].append(total_reward)
        results['costs'].append(total_cost)
        results['service_levels'].append(np.mean([m['demand_fulfillment'] for m in metrics]))
        results['all_metrics'].append(metrics)
    
    return results


# ============================
# PAGE 1: ACCUEIL
# ============================
if page == "🏠 Accueil":
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🏭 Bienvenue dans RLPlanif</h1>
        <p>Optimisation Intelligente du Plan Directeur de Production avec l'Apprentissage par Renforcement</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Badges de technologies
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="badge badge-primary">🤖 PPO Algorithm</span>
        <span class="badge badge-primary">🔥 PyTorch</span>
        <span class="badge badge-primary">📊 Stable-Baselines3</span>
        <span class="badge badge-primary">🎮 Gymnasium</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Section: Qu'est-ce que RLPlanif?
    st.markdown("### 💡 Qu'est-ce que RLPlanif?")
    st.markdown("""
    <div class="metric-card">
        <p><strong>RLPlanif</strong> est un système avancé d'aide à la décision pour la gestion 
        du <strong>Plan Directeur de Production (PDP)</strong>. Il utilise l'algorithme 
        <strong>PPO (Proximal Policy Optimization)</strong> pour apprendre automatiquement 
        les meilleures stratégies de production face à une demande variable.</p>
        <br>
        <p>Le système optimise trois leviers de production :</p>
        <ul>
            <li>⚙️ <strong>Production régulière</strong> - Capacité standard au coût optimal</li>
            <li>⏰ <strong>Heures supplémentaires</strong> - Flexibilité à coût modéré</li>
            <li>🏢 <strong>Sous-traitance</strong> - Capacité externe à coût premium</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section: Comment ça marche? (Timeline)
    st.markdown("### 🚀 Comment Utiliser l'Application?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="timeline-item">
            <div class="timeline-number">1</div>
            <div class="timeline-content">
                <h4>⚙️ Configuration</h4>
                <p>Définissez votre environnement industriel : capacités, coûts, demande attendue...</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-number">2</div>
            <div class="timeline-content">
                <h4>🏋️ Entraînement</h4>
                <p>Lancez l'entraînement de l'agent PPO sur votre configuration.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="timeline-item">
            <div class="timeline-number">3</div>
            <div class="timeline-content">
                <h4>📊 Évaluation</h4>
                <p>Comparez les performances du modèle aux stratégies classiques.</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-number">4</div>
            <div class="timeline-content">
                <h4>📋 Analyse</h4>
                <p>Visualisez les tableaux PDP détaillés et exportez vos plans.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section: Stratégies
    st.markdown("### 📊 Stratégies de Production Disponibles")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    strategies_info = [
        (col1, "🤖", "PPO", "Agent RL intelligent et adaptatif"),
        (col2, "🎯", "Lot-for-Lot", "Production exacte = Demande"),
        (col3, "🔄", "Chase", "Suivre la demande au plus près"),
        (col4, "📏", "Level", "Production lissée constante"),
        (col5, "💰", "EOQ", "Quantité économique optimale"),
    ]
    
    for col, icon, name, desc in strategies_info:
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3>{name}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Section: Statistiques du Système
    st.markdown("### 📈 État du Système")
    
    models = get_available_models()
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats_data = [
        (col1, len(models), "Modèles Entraînés", "🤖"),
        (col2, len(BASELINE_STRATEGIES), "Stratégies Baseline", "📋"),
        (col3, 4, "Exemples Industriels", "🔬"),
        (col4, "Oui" if st.session_state.current_config else "Non", "Config Active", "⚙️"),
    ]
    
    for col, value, label, icon in stats_data:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="stat-number">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section: Modèles récents
    if models:
        st.markdown("### 🕐 Modèles Récemment Entraînés")
        
        for i, model in enumerate(models[:5]):
            model_name = Path(model).parent.name
            st.markdown(f"""
            <div class="metric-card" style="padding: 0.8rem 1.2rem;">
                <span class="badge badge-success">{i+1}</span>
                <strong>{model_name}</strong>
                <span style="color: #888; font-size: 0.85rem; margin-left: 1rem;">📁 {Path(model).name}</span>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Aucun modèle entraîné</strong><br>
            Configurez un environnement et lancez un entraînement pour commencer!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section: Guide Rapide
    st.markdown("### ⚡ Guide Rapide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <strong>✅ Pour Commencer Rapidement:</strong><br>
            1. Allez dans <strong>⚙️ Configuration</strong><br>
            2. Sélectionnez un exemple (ex: Rouleurs)<br>
            3. Lancez l'entraînement dans <strong>🏋️ Entraînement PPO</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <strong>💡 Conseil:</strong><br>
            Pour de meilleurs résultats, utilisez au moins <strong>50 000 timesteps</strong> 
            d'entraînement et ajustez l'intensité de la demande selon vos besoins.
        </div>
        """, unsafe_allow_html=True)


# ============================
# PAGE 2: CONFIGURATION
# ============================
elif page == "⚙️ Configuration":
    st.header("⚙️ Configuration de l'Environnement")
    
    config_mode = st.radio(
        "Mode de configuration",
        ["🎯 Exemple Pré-configuré", "✏️ Configuration Personnalisée", "📁 Charger JSON"],
        horizontal=True
    )
    
    if config_mode == "🎯 Exemple Pré-configuré":
        st.subheader("Choisir un exemple industriel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            example = st.selectbox(
                "Exemple",
                ["rouleurs", "pdp_table", "compresseurs", "usinage"],
                format_func=lambda x: {
                    "rouleurs": "🔧 Rouleurs (12 périodes)",
                    "pdp_table": "📊 PDP Table (6 périodes)", 
                    "compresseurs": "⚙️ Compresseurs (8 périodes)",
                    "usinage": "🏭 Usinage (12 périodes)"
                }.get(x, x)
            )
        
        with col2:
            if example:
                config = get_example_config(example)
                
                st.markdown("**📋 Paramètres de l'exemple**")
                params_df = pd.DataFrame({
                    'Paramètre': ['Horizon', 'Produits', 'Cap. Régulière', 'Cap. HS', 'Cap. Sous-trait.', 'Stock Initial'],
                    'Valeur': [
                        f"{config.horizon} périodes",
                        str(config.n_products),
                        str(config.regular_capacity[0]),
                        str(config.overtime_capacity[0]),
                        str(config.subcontracting_capacity[0]),
                        str(config.initial_stock[0])
                    ]
                })
                st.dataframe(params_df, hide_index=True, width='stretch')
        
        if st.button("✅ Utiliser cette configuration", type="primary"):
            st.session_state.current_config = config
            st.success(f"✅ Configuration '{example}' chargée!")
            st.rerun()
    
    elif config_mode == "✏️ Configuration Personnalisée":
        st.subheader("Créer une configuration personnalisée")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏭 Paramètres Généraux**")
            n_products = st.number_input("Nombre de produits", 1, 5, 1)
            horizon = st.number_input("Horizon (périodes)", 3, 24, 12)
            
            st.markdown("**📦 Stocks**")
            initial_stock = st.number_input("Stock initial", 0.0, 1000.0, 50.0)
            max_stock = st.number_input("Stock maximum", 100.0, 5000.0, 500.0)
        
        with col2:
            st.markdown("**⚡ Capacités de Production**")
            regular_capacity = st.number_input("Capacité régulière", 1.0, 1000.0, 100.0)
            overtime_capacity = st.number_input("Capacité heures sup", 0.0, 500.0, 30.0)
            subcontracting_capacity = st.number_input("Capacité sous-traitance", 0.0, 500.0, 50.0)
            
            st.markdown("**🎯 Objectif**")
            service_level_target = st.slider("Niveau de service cible", 0.8, 1.0, 0.95)
        
        with col3:
            st.markdown("**💵 Coûts unitaires**")
            regular_cost = st.number_input("Coût régulier", 0.1, 100.0, 10.0)
            overtime_cost = st.number_input("Coût heures sup", 0.1, 200.0, 15.0)
            subcontracting_cost = st.number_input("Coût sous-traitance", 0.1, 300.0, 20.0)
            holding_cost = st.number_input("Coût de stockage", 0.0, 50.0, 2.0)
            shortage_cost = st.number_input("Coût de rupture", 0.0, 500.0, 100.0)
        
        # Intensité de la demande
        st.markdown("**📈 Génération de Demande**")
        demand_intensity = st.select_slider(
            "Intensité de la demande",
            options=["low", "medium", "high", "extreme"],
            value="high",
            format_func=lambda x: {
                "low": "🟢 Faible (< capacité)",
                "medium": "🟡 Moyenne (~ capacité)", 
                "high": "🟠 Haute (> capacité, heures supp)",
                "extreme": "🔴 Extrême (sous-traitance nécessaire)"
            }.get(x, x)
        )
        
        if st.button("✅ Créer la configuration", type="primary"):
            config = PDPEnvironmentConfig(
                n_products=n_products,
                horizon=horizon,
                regular_capacity=[regular_capacity] * n_products,
                overtime_capacity=[overtime_capacity] * n_products,
                subcontracting_capacity=[subcontracting_capacity] * n_products,
                initial_stock=[initial_stock] * n_products,
                max_stock=[max_stock] * n_products,
                regular_cost=[regular_cost] * n_products,
                overtime_cost=[overtime_cost] * n_products,
                subcontracting_cost=[subcontracting_cost] * n_products,
                holding_cost=[holding_cost] * n_products,
                shortage_cost=[shortage_cost] * n_products,
                service_level_target=service_level_target,
                demand_intensity=demand_intensity,
                normalize_observations=True
            )
            st.session_state.current_config = config
            st.success("✅ Configuration personnalisée créée!")
            st.rerun()
    
    else:  # Charger JSON
        st.subheader("Charger depuis un fichier JSON")
        
        uploaded_file = st.file_uploader("Fichier de configuration", type=['json'])
        
        if uploaded_file:
            try:
                config_dict = json.load(uploaded_file)
                config = PDPEnvironmentConfig(**config_dict)
                st.session_state.current_config = config
                st.success("✅ Configuration chargée depuis le fichier!")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    # Afficher la configuration actuelle
    if st.session_state.current_config:
        st.divider()
        st.subheader("📌 Configuration Actuelle")
        
        cfg = st.session_state.current_config
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Produits", cfg.n_products)
        col2.metric("Horizon", f"{cfg.horizon} périodes")
        col3.metric("Cap. Totale", f"{cfg.regular_capacity[0] + cfg.overtime_capacity[0] + cfg.subcontracting_capacity[0]:.0f}")
        col4.metric("Service Cible", f"{cfg.service_level_target:.0%}")


# ============================
# PAGE 3: ENTRAÎNEMENT PPO
# ============================
elif page == "🏋️ Entraînement PPO":
    st.header("🏋️ Entraînement du Modèle PPO")
    
    if st.session_state.current_config is None:
        st.warning("⚠️ Veuillez d'abord créer une configuration dans la page Configuration")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Paramètres d'Entraînement")
            
            timesteps = st.select_slider(
                "Nombre de timesteps",
                options=[10000, 30000, 50000, 100000, 200000, 500000],
                value=100000,
                format_func=lambda x: f"{x:,}"
            )
            
            learning_rate = st.select_slider(
                "Learning rate",
                options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                value=3e-4,
                format_func=lambda x: f"{x:.0e}"
            )
            
            run_name = st.text_input(
                "Nom de l'expérience",
                value=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        with col2:
            st.subheader("🧠 Architecture du Réseau")
            
            architecture = st.selectbox(
                "Architecture",
                ["small", "medium", "large"],
                index=1,
                format_func=lambda x: {
                    "small": "🔹 Small (64, 64)",
                    "medium": "🔸 Medium (256, 128, 64)",
                    "large": "🔺 Large (512, 256, 128)"
                }.get(x, x)
            )
            
            arch_map = {
                "small": [64, 64],
                "medium": [256, 128, 64],
                "large": [512, 256, 128]
            }
            
            n_epochs = st.slider("Epochs PPO", 5, 30, 10)
            batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=64)
            
            # Estimation du temps
            estimated_time = timesteps / 10000 * 2  # ~2 min par 10k steps
            st.info(f"⏱️ Temps estimé: ~{estimated_time:.0f} minutes")
        
        st.divider()
        
        if st.button("🚀 Lancer l'Entraînement", type="primary", width='stretch'):
            
            training_config = PPOTrainingConfig(
                total_timesteps=timesteps,
                learning_rate=learning_rate,
                policy_arch=arch_map[architecture],
                n_epochs=n_epochs,
                batch_size=batch_size,
                model_save_path=f"./models/{run_name}",
                tensorboard_log_path=f"./logs/tensorboard/{run_name}"
            )
            
            progress_bar = st.progress(0, text="Initialisation...")
            status_container = st.empty()
            
            try:
                with status_container.container():
                    st.info("🔧 Configuration de l'environnement...")
                
                progress_bar.progress(10, text="Configuration...")
                
                trainer = PPOTrainer(st.session_state.current_config, training_config)
                trainer.setup(env_name='strategic')
                
                progress_bar.progress(20, text="Entraînement en cours...")
                
                with status_container.container():
                    st.info(f"🏋️ Entraînement PPO: {timesteps:,} timesteps...")
                
                trainer.train()
                
                progress_bar.progress(100, text="Terminé!")
                
                st.session_state.trained_models.append({
                    'name': run_name,
                    'path': training_config.model_save_path,
                    'timesteps': timesteps,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                })
                
                with status_container.container():
                    st.success(f"""
                    ✅ **Entraînement terminé avec succès!**
                    
                    📁 Modèle sauvegardé: `{training_config.model_save_path}`
                    
                    📊 Pour voir les logs TensorBoard:
                    ```
                    tensorboard --logdir {training_config.tensorboard_log_path}
                    ```
                    """)
                
            except Exception as e:
                progress_bar.progress(0, text="Erreur!")
                with status_container.container():
                    st.error(f"❌ Erreur: {e}")
                    with st.expander("Détails de l'erreur"):
                        st.code(traceback.format_exc())
        
        # Liste des modèles entraînés
        if st.session_state.trained_models:
            st.divider()
            st.subheader("📚 Historique des Entraînements")
            models_df = pd.DataFrame(st.session_state.trained_models)
            st.dataframe(models_df, width='stretch', hide_index=True)


# ============================
# PAGE 4: ÉVALUATION
# ============================
elif page == "📊 Évaluation":
    st.header("📊 Évaluation et Comparaison")
    
    if st.session_state.current_config is None:
        st.warning("⚠️ Veuillez d'abord créer une configuration")
    else:
        available_models = get_available_models()
        
        if not available_models:
            st.info("ℹ️ Aucun modèle PPO disponible. Entraînez d'abord un modèle.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_model = st.selectbox(
                    "🤖 Modèle PPO à évaluer",
                    available_models,
                    format_func=lambda x: Path(x).parent.name
                )
            
            with col2:
                n_episodes = st.slider("Nombre d'épisodes", 1, 50, 10)
                compare_baselines = st.checkbox("Comparer avec les baselines", value=True)
            
            if st.button("🎯 Lancer l'Évaluation", type="primary", width='stretch'):
                progress = st.progress(0, text="Évaluation PPO...")
                
                try:
                    # Évaluer PPO
                    ppo_results = evaluate_ppo_model(
                        selected_model, 
                        st.session_state.current_config, 
                        n_episodes
                    )
                    
                    st.session_state.evaluation_results = {
                        'PPO (RL)': {
                            'reward': np.mean(ppo_results['rewards']),
                            'cost': np.mean(ppo_results['costs']),
                            'service': np.mean(ppo_results['service_levels']),
                            'stock': np.mean(ppo_results['stocks']),
                            'metrics': ppo_results['detailed_metrics'][0]
                        }
                    }
                    
                    progress.progress(30, text="PPO évalué!")
                    
                    # Évaluer baselines
                    if compare_baselines:
                        strategies = list(BASELINE_STRATEGIES.keys())
                        for i, strategy_name in enumerate(strategies):
                            progress.progress(
                                30 + int(70 * (i+1) / len(strategies)),
                                text=f"Évaluation {strategy_name}..."
                            )
                            
                            baseline_results = evaluate_baseline(
                                strategy_name, 
                                st.session_state.current_config, 
                                n_episodes
                            )
                            
                            st.session_state.evaluation_results[strategy_name] = {
                                'reward': np.mean(baseline_results['rewards']),
                                'cost': np.mean(baseline_results['costs']),
                                'service': np.mean(baseline_results['service_levels']),
                                'stock': 0,
                                'metrics': baseline_results['all_metrics'][0]
                            }
                    
                    progress.progress(100, text="Terminé!")
                    st.success("✅ Évaluation terminée!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    with st.expander("Détails"):
                        st.code(traceback.format_exc())
            
            # Afficher les résultats
            if st.session_state.evaluation_results:
                st.divider()
                st.subheader("📊 Résultats")
                
                # Créer DataFrame sans la colonne 'metrics' (qui contient les détails)
                display_data = {
                    name: {k: v for k, v in data.items() if k != 'metrics'}
                    for name, data in st.session_state.evaluation_results.items()
                }
                results_df = pd.DataFrame(display_data).T
                results_df.columns = ['Reward', 'Coût', 'Service', 'Stock Final']
                results_df = results_df.round(3)
                
                # Mettre en évidence les meilleures valeurs
                st.dataframe(
                    results_df.style.highlight_max(subset=['Reward', 'Service'], color='lightgreen')
                                   .highlight_min(subset=['Coût'], color='lightgreen'),
                    width='stretch'
                )
                
                # Identifier les meilleurs
                best_reward = results_df['Reward'].idxmax()
                best_service = results_df['Service'].idxmax()
                best_cost = results_df['Coût'].idxmin()
                
                col1, col2, col3 = st.columns(3)
                col1.success(f"🏆 Meilleur Reward: **{best_reward}**")
                col2.success(f"🎯 Meilleur Service: **{best_service}** ({results_df.loc[best_service, 'Service']:.1%})")
                col3.success(f"💰 Coût Minimal: **{best_cost}**")


# ============================
# PAGE 5: VISUALISATION
# ============================
elif page == "📈 Visualisation":
    st.header("📈 Visualisation des Résultats")
    
    if st.session_state.evaluation_results is None:
        st.warning("⚠️ Veuillez d'abord effectuer une évaluation")
    else:
        # Créer DataFrame sans la colonne 'metrics'
        display_data = {
            name: {k: v for k, v in data.items() if k != 'metrics'}
            for name, data in st.session_state.evaluation_results.items()
        }
        results_df = pd.DataFrame(display_data).T
        
        # Tabs pour différentes vues
        tab1, tab2, tab3 = st.tabs(["📊 Comparaison", "🎯 Radar", "📈 Détails"])
        
        with tab1:
            st.subheader("Comparaison des Stratégies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_reward = px.bar(
                    x=results_df.index,
                    y=results_df['reward'],
                    title='Reward Moyen par Stratégie',
                    labels={'x': 'Stratégie', 'y': 'Reward'},
                    color=results_df['reward'],
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_reward, width='stretch')
            
            with col2:
                fig_service = px.bar(
                    x=results_df.index,
                    y=results_df['service'],
                    title='Niveau de Service par Stratégie',
                    labels={'x': 'Stratégie', 'y': 'Service Level'},
                    color=results_df['service'],
                    color_continuous_scale='RdYlGn'
                )
                fig_service.add_hline(y=0.95, line_dash="dash", line_color="red",
                                     annotation_text="Cible 95%")
                st.plotly_chart(fig_service, width='stretch')
            
            # Coûts
            fig_cost = px.bar(
                x=results_df.index,
                y=results_df['cost'],
                title='Coût Total par Stratégie',
                labels={'x': 'Stratégie', 'y': 'Coût'},
                color=results_df['cost'],
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_cost, width='stretch')
        
        with tab2:
            st.subheader("Vue Multi-Critères (Radar)")
            
            # Normaliser pour le radar
            norm_df = results_df.copy()
            for col in ['reward', 'service']:
                norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min() + 1e-6)
            norm_df['cost'] = 1 - (norm_df['cost'] - norm_df['cost'].min()) / (norm_df['cost'].max() - norm_df['cost'].min() + 1e-6)
            
            fig_radar = go.Figure()
            
            for strategy in norm_df.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[norm_df.loc[strategy, 'reward'],
                       norm_df.loc[strategy, 'service'],
                       norm_df.loc[strategy, 'cost']],
                    theta=['Reward', 'Service', 'Coût (inversé)'],
                    fill='toself',
                    name=strategy
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig_radar, width='stretch')
        
        with tab3:
            st.subheader("Statistiques Détaillées")
            
            # Tableau complet avec stats
            stats_data = []
            for name, data in st.session_state.evaluation_results.items():
                stats_data.append({
                    'Stratégie': name,
                    'Reward': f"{data['reward']:.2f}",
                    'Coût Total': f"{data['cost']:.0f}",
                    'Service Level': f"{data['service']:.2%}",
                    'Stock Final': f"{data['stock']:.0f}"
                })
            
            st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)
        
        # Export
        st.divider()
        csv = results_df.to_csv()
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            csv,
            f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


# ============================
# PAGE 6: TABLEAU PDP
# ============================
elif page == "📋 Tableau PDP":
    st.header("📋 Plan Directeur de Production")
    
    if st.session_state.evaluation_results is None:
        st.warning("⚠️ Veuillez d'abord effectuer une évaluation")
    else:
        strategies = list(st.session_state.evaluation_results.keys())
        
        selected_strategy = st.selectbox(
            "Choisir une stratégie",
            strategies,
            format_func=lambda x: f"{'🤖' if 'PPO' in x else '📊'} {x}"
        )
        
        if selected_strategy and 'metrics' in st.session_state.evaluation_results[selected_strategy]:
            metrics = st.session_state.evaluation_results[selected_strategy]['metrics']
            
            st.subheader(f"📊 Tableau de Production - {selected_strategy}")
            
            # Créer le tableau détaillé
            df_detailed = create_detailed_production_table(
                metrics, 
                st.session_state.current_config,
                selected_strategy
            )
            
            if not df_detailed.empty:
                st.dataframe(df_detailed, width='stretch', height=600)
                
                # Métriques résumées
                st.divider()
                st.subheader("📈 Résumé")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_prod = sum([m['total_production'] for m in metrics])
                total_demand = sum([m['raw_metrics']['current_demand'][0] for m in metrics])
                avg_service = np.mean([m['demand_fulfillment'] for m in metrics])
                total_cost = sum([sum(m['costs'].values()) for m in metrics])
                
                col1.metric("📈 Production Totale", f"{total_prod:.0f}")
                col2.metric("📦 Demande Totale", f"{total_demand:.0f}")
                col3.metric("🎯 Service Moyen", f"{avg_service:.1%}")
                col4.metric("💰 Coût Total", f"{total_cost:.0f} €")
                
                # Graphiques
                st.divider()
                tab1, tab2, tab3 = st.tabs(["📈 Stock", "💰 Coûts", "🎯 Service"])
                
                periods = [f"P{m['period']+1}" for m in metrics]
                
                with tab1:
                    stocks = [m['inventory_level'][0] for m in metrics]
                    productions = [m['total_production'] for m in metrics]
                    demands = [m['raw_metrics']['current_demand'][0] for m in metrics]
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(x=periods, y=stocks, name="Stock", 
                                  line=dict(color='blue', width=3)),
                        secondary_y=False
                    )
                    fig.add_trace(
                        go.Bar(x=periods, y=productions, name="Production",
                              marker_color='lightblue', opacity=0.7),
                        secondary_y=True
                    )
                    fig.add_trace(
                        go.Scatter(x=periods, y=demands, name="Demande",
                                  line=dict(color='red', dash='dash')),
                        secondary_y=True
                    )
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_layout(title="Évolution du Stock et Production", height=400)
                    st.plotly_chart(fig, width='stretch')
                
                with tab2:
                    cost_prod = [m['costs']['production_cost'] for m in metrics]
                    cost_stock = [m['costs']['inventory_cost'] for m in metrics]
                    cost_rupture = [m['costs']['shortage_cost'] for m in metrics]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=periods, y=cost_prod, name='Production', marker_color='steelblue'))
                    fig.add_trace(go.Bar(x=periods, y=cost_stock, name='Stockage', marker_color='lightgreen'))
                    fig.add_trace(go.Bar(x=periods, y=cost_rupture, name='Rupture', marker_color='salmon'))
                    fig.update_layout(barmode='stack', title="Répartition des Coûts", height=400)
                    st.plotly_chart(fig, width='stretch')
                
                with tab3:
                    service_levels = [m['demand_fulfillment'] for m in metrics]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=periods, y=service_levels, name='Service',
                        marker_color=['green' if s >= 0.95 else 'orange' if s >= 0.8 else 'red' for s in service_levels]
                    ))
                    fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Cible 95%")
                    fig.update_layout(title="Niveau de Service par Période", yaxis_range=[0, 1.05], height=400)
                    st.plotly_chart(fig, width='stretch')
                
                # Export
                st.divider()
                csv = df_detailed.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger le tableau PDP (CSV)",
                    csv,
                    f"pdp_{selected_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        else:
            st.warning("⚠️ Pas de métriques disponibles pour cette stratégie")


# ============================
# PAGE 7: EXEMPLES RÉELS
# ============================
elif page == "🔬 Exemples Réels":
    st.header("🔬 Tester sur des Cas Réels")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        example = st.selectbox(
            "Choisir un exemple",
            ["rouleurs", "pdp_table", "compresseurs", "usinage"],
            format_func=lambda x: {
                "rouleurs": "🔧 Rouleurs",
                "pdp_table": "📊 PDP Table",
                "compresseurs": "⚙️ Compresseurs",
                "usinage": "🏭 Usinage"
            }.get(x, x)
        )
    
    with col2:
        config = get_example_config(example)
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Horizon", f"{config.horizon} périodes")
        col_b.metric("Produits", config.n_products)
        col_c.metric("Cap. Totale", f"{config.regular_capacity[0] + config.overtime_capacity[0]:.0f}")
    
    st.divider()
    
    # Test rapide des baselines
    if st.button("🧪 Tester les Stratégies Baseline", width='stretch'):
        progress = st.progress(0)
        
        results = {}
        strategies = list(BASELINE_STRATEGIES.keys())
        
        for i, strategy_name in enumerate(strategies):
            progress.progress((i + 1) / len(strategies), text=f"Test de {strategy_name}...")
            
            try:
                env = EnvironmentRegistry.create('strategic', config)
                StrategyClass = BASELINE_STRATEGIES[strategy_name]
                strategy = StrategyClass(env)
                
                total_reward, info = strategy.run_episode()
                metrics = info['metrics']
                
                total_cost = sum(
                    m['costs']['production_cost'] + 
                    m['costs']['inventory_cost'] + 
                    m['costs']['shortage_cost']
                    for m in metrics
                )
                
                results[strategy_name] = {
                    'Reward': round(total_reward, 2),
                    'Coût': round(total_cost, 0),
                    'Service': round(np.mean([m['demand_fulfillment'] for m in metrics]), 3),
                    'Stock Final': round(info['final_info']['inventory_level'][0], 0)
                }
            except Exception as e:
                results[strategy_name] = {'Erreur': str(e)}
        
        progress.progress(100, text="Terminé!")
        
        st.subheader("📊 Résultats")
        
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df, width='stretch')
        
        # Graphique
        if 'Reward' in results_df.columns:
            fig = px.bar(
                results_df.reset_index(),
                x='index',
                y='Reward',
                color='Service',
                title=f"Performance des Baselines - {example.upper()}",
                labels={'index': 'Stratégie'},
                color_continuous_scale='RdYlGn'
            ) 
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Entraîner sur cet exemple
    st.subheader("🏋️ Entraîner PPO sur cet Exemple")
    
    col1, col2 = st.columns(2)
    with col1:
        timesteps_ex = st.selectbox("Timesteps", [30000, 50000, 100000, 200000], index=1)
    with col2:
        run_name_ex = st.text_input("Nom", f"ppo_{example}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    
    if st.button("🚀 Lancer l'Entraînement", key="train_example"):
        with st.spinner(f"Entraînement sur '{example}'..."):
            try:
                training_config = PPOTrainingConfig(
                    total_timesteps=timesteps_ex,
                    model_save_path=f"./models/{run_name_ex}",
                    tensorboard_log_path=f"./logs/tensorboard/{run_name_ex}"
                )
                
                trainer = PPOTrainer(config, training_config)
                trainer.setup(env_name='strategic')
                trainer.train()
                
                st.success(f"✅ Modèle entraîné: `{training_config.model_save_path}`")
                
                # Ajouter à la liste
                st.session_state.trained_models.append({
                    'name': run_name_ex,
                    'path': training_config.model_save_path,
                    'timesteps': timesteps_ex,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                })
                
            except Exception as e:
                st.error(f"❌ Erreur: {e}")


# ============================
# FOOTER
# ============================
st.sidebar.divider()
st.sidebar.markdown("""
### 📚 À propos
**RLPlanif v2.0**  
Optimisation PDP par Reinforcement Learning

---
**Algorithme:** PPO  
**Framework:** Stable-Baselines3  
**Interface:** Streamlit

---
*Projet 4AS1 - RL*
""")
