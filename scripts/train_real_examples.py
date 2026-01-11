# scripts/train_real_examples.py
"""
Script pour entraîner un modèle RL sur les exemples réels d'exercices du cours de gestion de production
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.real_examples_configs import get_example_config
from config.training_configs import PPOTrainingConfig
from agents.ppo_trainer import PPOTrainer

def train_on_example(example_name: str, timesteps: int = 100000, 
                     env_type: str = 'strategic'):
    """
    Entraîne un modèle RL sur un exemple réel
    
    Args:
        example_name: Nom de l'exemple ('rouleurs', 'pdp_table', etc.)
        timesteps: Nombre de timesteps d'entraînement
        env_type: Type d'environnement ('base' ou 'strategic')
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 ENTRAÎNEMENT SUR EXEMPLE: {example_name.upper()}")
    print(f"{'='*80}\n")
    
    # Charger la configuration de l'exemple
    env_config = get_example_config(example_name)
    
    print(f"📋 Configuration:")
    print(f"   Exemple: {example_name}")
    print(f"   Horizon: {env_config.horizon} périodes")
    print(f"   Produits: {env_config.n_products}")
    print(f"   Capacités: R={env_config.regular_capacity}, "
          f"O={env_config.overtime_capacity}, S={env_config.subcontracting_capacity}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Environnement: {env_type}")
    print()
    
    # Configuration d'entraînement
    run_name = f"rl_{example_name}_{env_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_config = PPOTrainingConfig(
        total_timesteps=timesteps,
        model_save_path=os.path.join("./models/", run_name),
        tensorboard_log_path=os.path.join("./logs/tensorboard/", run_name)
    )
    
    os.makedirs(training_config.model_save_path, exist_ok=True)
    os.makedirs(training_config.tensorboard_log_path, exist_ok=True)
    
    print(f"💾 Modèle sera sauvegardé: {training_config.model_save_path}")
    print(f"📊 Logs TensorBoard: {training_config.tensorboard_log_path}")
    print()
    
    # Entraînement
    trainer = PPOTrainer(env_config, training_config)
    trainer.setup(env_name=env_type)
    
    print(f"🏋️  Début de l'entraînement...")
    trainer.train()
    
    print(f"\n✅ Entraînement terminé!")
    print(f"   Meilleur modèle: {training_config.model_save_path}/best_model.zip")
    print(f"   Modèle final: {training_config.model_save_path}/final_model.zip")
    print(f"\n💡 Pour évaluer:")
    print(f"   python scripts/evaluate_real_example.py \\")
    print(f"       --example {example_name} \\")
    print(f"       --model {training_config.model_save_path}/best_model.zip")
    
    return training_config.model_save_path

def main():
    parser = argparse.ArgumentParser(
        description="Entraîner un modèle RL sur un exemple réel"
    )
    parser.add_argument(
        '--example',
        type=str,
        required=True,
        choices=['rouleurs', 'pdp_table', 'compresseurs', 'usinage'],
        help='Exemple à utiliser pour l\'entraînement'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Nombre de timesteps d\'entraînement'
    )
    parser.add_argument(
        '--env_type',
        type=str,
        default='strategic',
        choices=['base', 'strategic'],
        help='Type d\'environnement'
    )
    
    args = parser.parse_args()
    
    # Lancer l'entraînement
    model_path = train_on_example(
        args.example,
        timesteps=args.timesteps,
        env_type=args.env_type
    )
    
    print(f"\n🎉 Entraînement terminé avec succès!")
    print(f"\n📝 Prochaines étapes:")
    print(f"   1. Vérifier les logs TensorBoard")
    print(f"   2. Évaluer le modèle")
    print(f"   3. Comparer avec les baselines")

if __name__ == "__main__":
    main()
