# agents/ppo_trainer.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import torch.nn as nn
import os

from config.training_configs import PPOTrainingConfig
from environments.env_registry import EnvironmentRegistry
from agents.rl_utils import SaveOnBestTrainingRewardCallback, EarlyStoppingCallback

class PPOTrainer:
    """Gestionnaire d'entraînement PPO avec améliorations"""
    
    def __init__(self, env_config, training_config: PPOTrainingConfig):
        self.env_config = env_config
        self.training_config = training_config
        self.model = None
        self.env = None
        
    def setup(self, env_name: str = 'strategic'):
        """Configure l'environnement et le modèle"""

        # Créer le répertoire AVANT de créer l'environnement
        os.makedirs(self.training_config.model_save_path, exist_ok=True)
        os.makedirs(self.training_config.tensorboard_log_path, exist_ok=True)
        
        def make_env():
            env = EnvironmentRegistry.create(env_name, self.env_config)
            return Monitor(env, self.training_config.model_save_path)
        
        # Création de l'environnement vectorisé
        self.env = DummyVecEnv([make_env])
        
        # Normalisation avec paramètres optimisés
        if self.env_config.normalize_observations:
            self.env = VecNormalize(
                self.env,
                norm_obs=True,      # Normalise les observations
                norm_reward=True,   # Normalise les rewards
                clip_obs=10.0,      # Clip les observations normalisées
                clip_reward=10.0,   # Clip les rewards normalisés
                gamma=self.training_config.gamma
            )
        
        # Modèle PPO avec MultiInputPolicy
        self.model = PPO(
            "MultiInputPolicy",
            self.env,
            learning_rate=self.training_config.learning_rate,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            n_epochs=self.training_config.n_epochs,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            clip_range=self.training_config.clip_range,
            ent_coef=self.training_config.ent_coef,
            vf_coef=self.training_config.vf_coef,
            max_grad_norm=self.training_config.max_grad_norm,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=self.training_config.policy_arch,  # Architecture pour la policy
                    vf=self.training_config.policy_arch   # Architecture pour la value function
                ),
                activation_fn=nn.ReLU,
                # Ajout de normalisation des features
                normalize_images=False
            ),
            tensorboard_log=self.training_config.tensorboard_log_path,
            verbose=1,
            seed=self.env_config.seed
        )
    
    def train(self):
        """Lance l'entraînement avec callbacks améliorés"""
        if self.model is None:
            self.setup()
        
        # Création du répertoire pour les logs
        os.makedirs(self.training_config.model_save_path, exist_ok=True)
        os.makedirs(self.training_config.tensorboard_log_path, exist_ok=True)
        
        # Callbacks
        callback_list = [
            SaveOnBestTrainingRewardCallback(
                check_freq=self.training_config.save_interval, 
                log_dir=self.training_config.model_save_path, 
                verbose=1
            ),
            # Décommenter l'early stopping si nécessaire
            # EarlyStoppingCallback(
            #     check_freq=self.training_config.save_interval * 2, 
            #     patience=5, 
            #     log_dir=self.training_config.model_save_path, 
            #     verbose=1
            # )
        ]
        
        print("🚀 Début de l'entraînement...")
        print(f"   Total timesteps: {self.training_config.total_timesteps}")
        print(f"   Learning rate: {self.training_config.learning_rate}")
        print(f"   N_steps: {self.training_config.n_steps}")
        print(f"   Batch size: {self.training_config.batch_size}")
        
        self.model.learn(
            total_timesteps=self.training_config.total_timesteps,
            tb_log_name="ppo_pdp_training",
            callback=CallbackList(callback_list),
            progress_bar=True  # Barre de progression
        )
        
        # Sauvegarde finale
        final_model_path = os.path.join(self.training_config.model_save_path, "final_model")
        self.model.save(final_model_path)
        print(f"✅ Modèle final sauvegardé: {final_model_path}")
        
        # Sauvegarde de VecNormalize si utilisé
        if self.env_config.normalize_observations:
            vec_normalize_path = os.path.join(self.training_config.model_save_path, "vec_normalize.pkl")
            self.env.save(vec_normalize_path)
            print(f"✅ VecNormalize sauvegardé: {vec_normalize_path}")
    
    def load_model(self, model_path: str):
        """Charge un modèle déjà entraîné"""
        self.model = PPO.load(model_path, env=self.env)
        print(f"✅ Modèle chargé depuis: {model_path}")
