# app.py - Interface Streamlit pour PDP RL

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path

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
    page_title="PDP RL Manager",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation de la session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'current_config' not in st.session_state:
    st.session_state.current_config = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Titre principal
st.markdown('<p class="main-header">🏭 PDP RL Manager - Gestion Intelligente de Production</p>', 
            unsafe_allow_html=True)

# Sidebar - Navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["🏠 Accueil", "⚙️ Configuration", "🏋️ Entraînement", 
     "📊 Évaluation", "📈 Visualisation", "🔬 Exemples Réels"]
)

# ============================
# PAGE 1: ACCUEIL
# ============================
if page == "🏠 Accueil":
    st.header("Bienvenue dans PDP RL Manager")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 À Propos</h3>
        <p>Application de gestion de production basée sur le Reinforcement Learning (PPO).</p>
        <ul>
        <li>Configurez votre environnement</li>
        <li>Entraînez des modèles RL</li>
        <li>Comparez avec des baselines</li>
        <li>Visualisez les résultats</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🚀 Démarrage Rapide</h3>
        <ol>
        <li>Créez une configuration</li>
        <li>Lancez l'entraînement</li>
        <li>Évaluez les performances</li>
        <li>Visualisez les résultats</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📚 Ressources</h3>
        <ul>
        <li>Exemples pré-configurés</li>
        <li>Stratégies baseline</li>
        <li>Comparaisons automatiques</li>
        <li>Export des résultats</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Statistiques
    st.subheader("📊 Statistiques")
    
    models_dir = Path("./models")
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*/")))
    else:
        model_count = 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modèles Entraînés", model_count)
    col2.metric("Stratégies Disponibles", len(BASELINE_STRATEGIES))
    col3.metric("Exemples Réels", 4)
    col4.metric("Configurations", "Illimité")

# ============================
# PAGE 2: CONFIGURATION
# ============================
elif page == "⚙️ Configuration":
    st.header("⚙️ Configuration de l'Environnement")
    
    # Choix du mode de configuration
    config_mode = st.radio(
        "Mode de configuration",
        ["🎯 Exemple Pré-configuré", "✏️ Configuration Personnalisée", "📁 Charger depuis Fichier"]
    )
    
    if config_mode == "🎯 Exemple Pré-configuré":
        st.subheader("Choisir un exemple")
        
        example = st.selectbox(
            "Exemple",
            ["rouleurs", "pdp_table", "compresseurs", "usinage"],
            help="Exemples basés sur des cas réels"
        )
        
        # Afficher les détails de l'exemple
        if example:
            config = get_example_config(example)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📋 Paramètres**")
                st.write(f"- Horizon: {config.horizon} périodes")
                st.write(f"- Produits: {config.n_products}")
                st.write(f"- Capacité régulière: {config.regular_capacity}")
                st.write(f"- Heures sup: {config.overtime_capacity}")
                st.write(f"- Sous-traitance: {config.subcontracting_capacity}")
            
            with col2:
                st.markdown("**💰 Coûts**")
                st.write(f"- Production régulière: {config.regular_cost}")
                st.write(f"- Heures sup: {config.overtime_cost}")
                st.write(f"- Sous-traitance: {config.subcontracting_cost}")
                st.write(f"- Stockage: {config.holding_cost}")
                st.write(f"- Rupture: {config.shortage_cost}")
            
            if st.button("✅ Utiliser cette configuration"):
                st.session_state.current_config = config
                st.success("✅ Configuration chargée avec succès!")
    
    elif config_mode == "✏️ Configuration Personnalisée":
        st.subheader("Créer une configuration personnalisée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏭 Paramètres Généraux**")
            n_products = st.number_input("Nombre de produits", min_value=1, max_value=5, value=1)
            horizon = st.number_input("Horizon (périodes)", min_value=3, max_value=24, value=12)
            
            st.markdown("**⚡ Capacités de Production**")
            regular_capacity = st.number_input("Capacité régulière", min_value=1.0, value=100.0)
            overtime_capacity = st.number_input("Capacité heures sup", min_value=0.0, value=30.0)
            subcontracting_capacity = st.number_input("Capacité sous-traitance", min_value=0.0, value=50.0)
            
            st.markdown("**📦 Stocks**")
            initial_stock = st.number_input("Stock initial", min_value=0.0, value=50.0)
            max_stock = st.number_input("Stock maximum", min_value=1.0, value=500.0)
        
        with col2:
            st.markdown("**💵 Coûts**")
            regular_cost = st.number_input("Coût régulier", min_value=0.1, value=10.0)
            overtime_cost = st.number_input("Coût heures sup", min_value=0.1, value=15.0)
            subcontracting_cost = st.number_input("Coût sous-traitance", min_value=0.1, value=20.0)
            holding_cost = st.number_input("Coût de stockage", min_value=0.0, value=2.0)
            shortage_cost = st.number_input("Coût de rupture", min_value=0.0, value=100.0)
            
            st.markdown("**🎯 Objectifs**")
            service_level_target = st.slider("Niveau de service cible", 0.0, 1.0, 0.95)
        
        if st.button("✅ Créer la configuration"):
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
                normalize_observations=True
            )
            st.session_state.current_config = config
            st.success("✅ Configuration créée avec succès!")
    
    else:  # Charger depuis fichier
        st.subheader("Charger une configuration depuis un fichier JSON")
        
        uploaded_file = st.file_uploader("Choisir un fichier JSON", type=['json'])
        
        if uploaded_file is not None:
            try:
                config_dict = json.load(uploaded_file)
                config = PDPEnvironmentConfig(**config_dict)
                st.session_state.current_config = config
                st.success("✅ Configuration chargée depuis le fichier!")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {e}")
    
    # Afficher la configuration actuelle
    if st.session_state.current_config:
        st.divider()
        st.subheader("📋 Configuration Actuelle")
        
        config = st.session_state.current_config
        config_df = pd.DataFrame({
            'Paramètre': ['Produits', 'Horizon', 'Capacité Régulière', 'Stock Initial', 'Coût Régulier'],
            'Valeur': [
                config.n_products,
                config.horizon,
                config.regular_capacity[0],
                config.initial_stock[0],
                config.regular_cost[0]
            ]
        })
        st.dataframe(config_df, use_container_width=True)

# ============================
# PAGE 3: ENTRAÎNEMENT
# ============================
elif page == "🏋️ Entraînement":
    st.header("🏋️ Entraînement du Modèle RL")
    
    if st.session_state.current_config is None:
        st.warning("⚠️ Veuillez d'abord créer une configuration dans la page Configuration")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Paramètres d'Entraînement")
            
            timesteps = st.selectbox(
                "Nombre de timesteps",
                [30000, 50000, 100000, 200000, 500000],
                index=2
            )
            
            env_type = st.selectbox(
                "Type d'environnement",
                ["strategic", "base"],
                help="Strategic: avec contraintes de stabilité, Base: environnement simple"
            )
            
            learning_rate = st.select_slider(
                "Learning rate",
                options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                value=3e-4,
                format_func=lambda x: f"{x:.0e}"
            )
            
            run_name = st.text_input(
                "Nom de l'entraînement",
                value=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        with col2:
            st.subheader("Architecture du Réseau")
            
            policy_layers = st.multiselect(
                "Couches du réseau",
                [64, 128, 256, 512],
                default=[256, 128, 64]
            )
            
            n_epochs = st.slider("Nombre d'epochs", 5, 20, 10)
            batch_size = st.slider("Batch size", 32, 256, 64, step=32)
            
            st.markdown("**Estimation du temps**")
            estimated_time = timesteps / 10000 * 3  # Approximation
            st.info(f"⏱️ Temps estimé: ~{estimated_time:.0f} minutes")
        
        st.divider()
        
        # Bouton d'entraînement
        if st.button("🚀 Lancer l'Entraînement", type="primary", use_container_width=True):
            st.session_state.training_in_progress = True
            
            # Créer la configuration d'entraînement
            training_config = PPOTrainingConfig(
                total_timesteps=timesteps,
                learning_rate=learning_rate,
                policy_arch=sorted(policy_layers, reverse=True),
                n_epochs=n_epochs,
                batch_size=batch_size,
                model_save_path=f"./models/{run_name}",
                tensorboard_log_path=f"./logs/tensorboard/{run_name}"
            )
            
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔧 Initialisation de l'environnement...")
                progress_bar.progress(10)
                
                # Créer et configurer le trainer
                trainer = PPOTrainer(st.session_state.current_config, training_config)
                trainer.setup(env_name=env_type)
                
                status_text.text("🏋️ Entraînement en cours...")
                progress_bar.progress(30)
                
                # Lancer l'entraînement
                trainer.train()
                
                progress_bar.progress(100)
                status_text.text("✅ Entraînement terminé!")
                
                # Sauvegarder le modèle dans la session
                st.session_state.trained_models.append({
                    'name': run_name,
                    'path': training_config.model_save_path,
                    'timesteps': timesteps,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                st.success(f"✅ Modèle entraîné avec succès!\n\nModèle sauvegardé: {training_config.model_save_path}")
                
                # Afficher le lien TensorBoard
                st.info(f"📊 Pour voir les logs d'entraînement:\n\n`tensorboard --logdir {training_config.tensorboard_log_path}`")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement: {e}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                st.session_state.training_in_progress = False
        
        # Afficher les modèles entraînés
        if st.session_state.trained_models:
            st.divider()
            st.subheader("📚 Modèles Entraînés")
            
            models_df = pd.DataFrame(st.session_state.trained_models)
            st.dataframe(models_df, use_container_width=True)

# ============================
# PAGE 4: ÉVALUATION
# ============================
elif page == "📊 Évaluation":
    st.header("📊 Évaluation et Comparaison")
    
    if st.session_state.current_config is None:
        st.warning("⚠️ Veuillez d'abord créer une configuration")
    else:
        # Sélection du modèle à évaluer
        st.subheader("Choisir un modèle à évaluer")
        
        # Lister les modèles disponibles
        models_dir = Path("./models")
        if models_dir.exists():
            available_models = [str(p) for p in models_dir.glob("*/best_model.zip")]
            if not available_models:
                available_models = [str(p) for p in models_dir.glob("*/final_model.zip")]
        else:
            available_models = []
        
        if not available_models:
            st.info("ℹ️ Aucun modèle disponible. Entraînez d'abord un modèle.")
        else:
            selected_model = st.selectbox("Modèle", available_models)
            
            col1, col2 = st.columns(2)
            with col1:
                n_episodes = st.slider("Nombre d'épisodes", 1, 50, 10)
            with col2:
                compare_baselines = st.checkbox("Comparer avec les baselines", value=True)
            
            if st.button("🎯 Lancer l'Évaluation", type="primary"):
                with st.spinner("Évaluation en cours..."):
                    # Import des fonctions d'évaluation
                    from stable_baselines3 import PPO
                    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
                    
                    # Évaluer le modèle RL
                    def make_env():
                        return EnvironmentRegistry.create('strategic', st.session_state.current_config)
                    
                    eval_env = DummyVecEnv([make_env])
                    
                    # Charger VecNormalize
                    vec_normalize_path = Path(selected_model).parent / "vec_normalize.pkl"
                    if vec_normalize_path.exists():
                        eval_env = VecNormalize.load(str(vec_normalize_path), eval_env)
                        eval_env.training = False
                        eval_env.norm_reward = False
                    
                    model = PPO.load(selected_model, env=eval_env)
                    
                    # Collecter les résultats
                    rl_results = {
                        'rewards': [],
                        'service_levels': [],
                        'costs': [],
                        'stocks': []
                    }
                    
                    for _ in range(n_episodes):
                        obs = eval_env.reset()
                        done = False
                        total_reward = 0
                        total_cost = 0
                        service_levels = []
                        
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, rewards, dones, infos = eval_env.step(action)
                            
                            reward = rewards[0]
                            info = infos[0]
                            done = dones[0]
                            
                            total_reward += reward
                            total_cost += sum(info['costs'].values())
                            service_levels.append(info['demand_fulfillment'])
                        
                        rl_results['rewards'].append(total_reward)
                        rl_results['costs'].append(total_cost)
                        rl_results['service_levels'].append(np.mean(service_levels))
                        rl_results['stocks'].append(info['inventory_level'][0])
                    
                    # Stocker les résultats
                    st.session_state.evaluation_results = {
                        'RL (PPO)': {
                            'reward': np.mean(rl_results['rewards']),
                            'cost': np.mean(rl_results['costs']),
                            'service': np.mean(rl_results['service_levels']),
                            'stock': np.mean(rl_results['stocks'])
                        }
                    }
                    
                    # Comparer avec baselines si demandé
                    if compare_baselines:
                        for strategy_name in BASELINE_STRATEGIES.keys():
                            env = EnvironmentRegistry.create('strategic', st.session_state.current_config)
                            StrategyClass = BASELINE_STRATEGIES[strategy_name]
                            strategy = StrategyClass(env)
                            
                            baseline_results = {
                                'rewards': [],
                                'costs': [],
                                'service_levels': []
                            }
                            
                            for _ in range(n_episodes):
                                total_reward, info = strategy.run_episode()
                                metrics = info['metrics']
                                
                                total_cost = sum(
                                    m['costs']['production_cost'] + 
                                    m['costs']['inventory_cost'] + 
                                    m['costs']['shortage_cost']
                                    for m in metrics
                                )
                                
                                baseline_results['rewards'].append(total_reward)
                                baseline_results['costs'].append(total_cost)
                                baseline_results['service_levels'].append(
                                    np.mean([m['demand_fulfillment'] for m in metrics])
                                )
                            
                            st.session_state.evaluation_results[strategy_name] = {
                                'reward': np.mean(baseline_results['rewards']),
                                'cost': np.mean(baseline_results['costs']),
                                'service': np.mean(baseline_results['service_levels']),
                                'stock': 0  # Non calculé pour baselines
                            }
                
                st.success("✅ Évaluation terminée!")
            
            # Afficher les résultats
            if st.session_state.evaluation_results:
                st.divider()
                st.subheader("📊 Résultats")
                
                results_df = pd.DataFrame(st.session_state.evaluation_results).T
                results_df = results_df.round(2)
                results_df.columns = ['Reward Moyen', 'Coût Total', 'Service Level', 'Stock Final']
                
                st.dataframe(results_df, use_container_width=True)
                
                # Identifier le meilleur
                best_service = results_df['Service Level'].idxmax()
                best_reward = results_df['Reward Moyen'].idxmax()
                
                col1, col2 = st.columns(2)
                col1.success(f"🏆 Meilleur Service: **{best_service}** ({results_df.loc[best_service, 'Service Level']:.3f})")
                col2.success(f"🏆 Meilleur Reward: **{best_reward}** ({results_df.loc[best_reward, 'Reward Moyen']:.2f})")

# ============================
# PAGE 5: VISUALISATION
# ============================
elif page == "📈 Visualisation":
    st.header("📈 Visualisation des Résultats")
    
    if st.session_state.evaluation_results is None:
        st.warning("⚠️ Veuillez d'abord effectuer une évaluation")
    else:
        # Graphique de comparaison
        st.subheader("Comparaison des Stratégies")
        
        results_df = pd.DataFrame(st.session_state.evaluation_results).T
        
        # Graphique en barres pour les métriques
        fig = go.Figure()
        
        metrics = ['reward', 'cost', 'service', 'stock']
        metric_names = ['Reward', 'Coût', 'Service Level', 'Stock Final']
        
        tab1, tab2, tab3, tab4 = st.tabs(metric_names)
        
        with tab1:
            fig1 = px.bar(
                x=results_df.index,
                y=results_df['reward'],
                labels={'x': 'Stratégie', 'y': 'Reward Moyen'},
                title='Comparaison des Rewards',
                color=results_df['reward'],
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            fig2 = px.bar(
                x=results_df.index,
                y=results_df['cost'],
                labels={'x': 'Stratégie', 'y': 'Coût Total'},
                title='Comparaison des Coûts',
                color=results_df['cost'],
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            fig3 = px.bar(
                x=results_df.index,
                y=results_df['service'],
                labels={'x': 'Stratégie', 'y': 'Service Level'},
                title='Comparaison du Service Level',
                color=results_df['service'],
                color_continuous_scale='RdYlGn'
            )
            fig3.add_hline(y=0.95, line_dash="dash", line_color="red", 
                          annotation_text="Cible: 95%")
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            fig4 = px.bar(
                x=results_df.index,
                y=results_df['stock'],
                labels={'x': 'Stratégie', 'y': 'Stock Final'},
                title='Comparaison du Stock Final',
                color=results_df['stock'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        st.divider()
        
        # Radar chart de comparaison
        st.subheader("Vue d'Ensemble Multi-Critères")
        
        # Normaliser les données pour le radar
        normalized_df = results_df.copy()
        normalized_df['reward'] = (normalized_df['reward'] - normalized_df['reward'].min()) / (normalized_df['reward'].max() - normalized_df['reward'].min())
        normalized_df['cost'] = 1 - (normalized_df['cost'] - normalized_df['cost'].min()) / (normalized_df['cost'].max() - normalized_df['cost'].min())
        
        fig_radar = go.Figure()
        
        for strategy in normalized_df.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=[normalized_df.loc[strategy, 'reward'],
                   normalized_df.loc[strategy, 'cost'],
                   normalized_df.loc[strategy, 'service']],
                theta=['Reward', 'Coût (inversé)', 'Service'],
                fill='toself',
                name=strategy
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Comparaison Multi-Critères (Normalisée)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Télécharger les résultats
        st.divider()
        st.subheader("💾 Exporter les Résultats")
        
        csv = results_df.to_csv()
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ============================
# PAGE 6: EXEMPLES RÉELS
# ============================
elif page == "🔬 Exemples Réels":
    st.header("🔬 Tester sur les Exemples Réels")
    
    example = st.selectbox(
        "Choisir un exemple",
        ["rouleurs", "pdp_table", "compresseurs", "usinage"]
    )
    
    # Charger la config de l'exemple
    config = get_example_config(example)
    
    # Afficher les détails
    st.subheader(f"📋 Détails: {example.upper()}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Horizon", f"{config.horizon} périodes")
    col2.metric("Produits", config.n_products)
    col3.metric("Capacité Régulière", config.regular_capacity[0])
    
    # Tester les stratégies baselines
    if st.button("🧪 Tester les Stratégies Baseline"):
        progress = st.progress(0)
        status = st.empty()
        
        results = {}
        strategies = list(BASELINE_STRATEGIES.keys())
        
        for i, strategy_name in enumerate(strategies):
            status.text(f"Test de {strategy_name}...")
            progress.progress((i + 1) / len(strategies))
            
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
                'Reward': total_reward,
                'Coût': total_cost,
                'Service': np.mean([m['demand_fulfillment'] for m in metrics]),
                'Stock Final': info['final_info']['inventory_level'][0]
            }
        
        status.text("✅ Tests terminés!")
        progress.progress(1.0)
        
        # Afficher les résultats
        st.divider()
        st.subheader("📊 Résultats des Baselines")
        
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df, use_container_width=True)
        
        # Graphique
        fig = px.bar(
            results_df,
            title=f"Comparaison des Stratégies - {example.upper()}",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Entraîner RL sur cet exemple
    st.subheader("🏋️ Entraîner RL sur cet Exemple")
    
    col1, col2 = st.columns(2)
    with col1:
        timesteps_example = st.selectbox(
            "Timesteps",
            [30000, 50000, 100000, 200000],
            index=1
        )
    with col2:
        run_name_example = st.text_input(
            "Nom",
            value=f"rl_{example}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
    
    if st.button("🚀 Lancer l'Entraînement"):
        with st.spinner("Entraînement en cours..."):
            training_config = PPOTrainingConfig(
                total_timesteps=timesteps_example,
                model_save_path=f"./models/{run_name_example}",
                tensorboard_log_path=f"./logs/tensorboard/{run_name_example}"
            )
            
            trainer = PPOTrainer(config, training_config)
            trainer.setup(env_name='strategic')
            trainer.train()
            
            st.success(f"✅ Entraînement terminé!\n\nModèle: {training_config.model_save_path}")

# ============================
# FOOTER
# ============================
st.sidebar.divider()
st.sidebar.markdown("""
---
### 📚 Aide
- [Documentation](https://github.com)
- [Guide d'utilisation](./GUIDE_EXEMPLES_REELS.md)
- [Exemples](./README_EXEMPLES_REELS.md)

### 🛠️ Informations
**Version:** 1.0.0  
**Framework:** Streamlit + Stable-Baselines3  
**Algorithme:** PPO (Proximal Policy Optimization)
""")