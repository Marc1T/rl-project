# scripts/train.py

import argparse
import os
import sys
from datetime import datetime

# Ajout du chemin racine du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_configs import PDPEnvironmentConfig
from config.training_configs import PPOTrainingConfig
from agents.ppo_trainer import PPOTrainer

def main():
    parser = argparse.ArgumentParser(description="Entraînement PDP RL propre")
    parser.add_argument('--products', type=int, default=1, help='Nombre de produits')
    parser.add_argument('--timesteps', type=int, default=50000, help='Timesteps')
    parser.add_argument('--horizon', type=int, default=12, help='Horizon')
    parser.add_argument('--env_type', type=str, default='strategic', choices=['base', 'strategic'], help='Type d environnement à utiliser')
    
    args = parser.parse_args()
    
    print("🎯 ENTRAÎNEMENT PDP RL - CONFIGURATION PROPRE")
    print(f"Produits: {args.products}, Timesteps: {args.timesteps}, Horizon: {args.horizon}, Env: {args.env_type}")
    
    # Configuration explicite de l'environnement
    env_config = PDPEnvironmentConfig(
        n_products=args.products,
        horizon=args.horizon,
        regular_capacity=[100] * args.products,
        overtime_capacity=[30] * args.products,
        subcontracting_capacity=[50] * args.products,
        initial_stock=[150] * args.products
    )
    
    # Configuration entraînement
    run_name = f"ppo_pdp_{args.env_type}_{args.products}prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_config = PPOTrainingConfig(
        total_timesteps=args.timesteps,
        model_save_path=os.path.join("./models/", run_name)
    )
    
    os.makedirs(training_config.model_save_path, exist_ok=True)
    
    # Entraînement
    trainer = PPOTrainer(env_config, training_config)
    trainer.setup(env_name=args.env_type)
    trainer.train()
    
    print(f"✅ Entraînement terminé! Modèle dans: {training_config.model_save_path}")

if __name__ == "__main__":
    main()