# scripts/test_env_diagnostic.py

"""
Script de diagnostic pour tester l'environnement et identifier les problèmes
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.environment_configs import PDPEnvironmentConfig
from environments.env_registry import EnvironmentRegistry

def test_basic_functionality():
    """Test des fonctionnalités de base"""
    print("=" * 60)
    print("TEST 1: FONCTIONNALITÉS DE BASE")
    print("=" * 60)
    
    config = PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        normalize_observations=False  # Désactivé pour voir les vraies valeurs
    )
    
    env = EnvironmentRegistry.create('strategic', config)
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"✅ Reset réussi")
    print(f"   Stock initial: {obs['current_stock']}")
    print(f"   Demandes futures shape: {obs['future_demands'].shape}")
    print(f"   Demande période 0: {info['demands'][:, 0]}")
    
    # Test d'un step
    action = np.array([0.8, 0.2, 0.1])  # Regular, Overtime, Subcontracting
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n✅ Step réussi")
    print(f"   Reward: {reward:.2f}")
    print(f"   Stock après step: {obs['current_stock']}")
    print(f"   Production totale: {info['total_production']:.1f}")
    print(f"   Niveau de service: {info['demand_fulfillment']:.3f}")
    print(f"   Coûts: {info['costs']}")
    
    return True

def test_reward_scale():
    """Test de l'échelle des rewards"""
    print("\n" + "=" * 60)
    print("TEST 2: ÉCHELLE DES REWARDS")
    print("=" * 60)
    
    config = PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        normalize_observations=False
    )
    
    env = EnvironmentRegistry.create('strategic', config)
    
    # Test avec différentes actions
    test_actions = [
        ("Faible production", np.array([0.2, 0.0, 0.0])),
        ("Production moyenne", np.array([0.7, 0.1, 0.0])),
        ("Production élevée", np.array([1.0, 0.5, 0.3])),
        ("Uniquement sous-traitance", np.array([0.0, 0.0, 1.0])),
    ]
    
    rewards = []
    
    for name, action in test_actions:
        env.reset(seed=42)
        obs, reward, _, _, info = env.step(action)
        rewards.append(reward)
        
        print(f"\n{name}:")
        print(f"   Action: R{action[0]:.1f}/O{action[1]:.1f}/S{action[2]:.1f}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Production: {info['total_production']:.1f}")
        print(f"   Stock final: {obs['current_stock'][0]:.1f}")
        print(f"   Service: {info['demand_fulfillment']:.3f}")
    
    print(f"\n📊 Statistiques des rewards:")
    print(f"   Min: {np.min(rewards):.4f}")
    print(f"   Max: {np.max(rewards):.4f}")
    print(f"   Moyenne: {np.mean(rewards):.4f}")
    print(f"   Std: {np.std(rewards):.4f}")
    
    if np.std(rewards) < 0.01:
        print("   ⚠️  ATTENTION: Les rewards varient très peu!")
    else:
        print("   ✅ Les rewards ont une variance acceptable")
    
    return True

def test_episode_consistency():
    """Test de cohérence sur un épisode complet"""
    print("\n" + "=" * 60)
    print("TEST 3: COHÉRENCE D'UN ÉPISODE COMPLET")
    print("=" * 60)
    
    config = PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        normalize_observations=False
    )
    
    env = EnvironmentRegistry.create('strategic', config)
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    stocks = [obs['current_stock'][0]]
    rewards_list = []
    
    print("\nDéroulement de l'épisode (action constante):")
    action = np.array([0.8, 0.1, 0.0])  # Action constante
    
    for period in range(config.horizon):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        stocks.append(obs['current_stock'][0])
        rewards_list.append(reward)
        
        if period < 3 or period >= config.horizon - 2:  # Afficher début et fin
            print(f"   Période {period}: Stock={obs['current_stock'][0]:6.1f}, "
                  f"Reward={reward:7.4f}, Service={info['demand_fulfillment']:.3f}")
    
    print(f"\n📊 Résumé épisode:")
    print(f"   Reward total: {total_reward:.2f}")
    print(f"   Reward moyen: {np.mean(rewards_list):.4f}")
    print(f"   Stock min/max: {np.min(stocks):.1f} / {np.max(stocks):.1f}")
    print(f"   Service moyen: {np.mean([info['demand_fulfillment']]):.3f}")
    
    # Vérifications
    if np.all(np.array(rewards_list) == rewards_list[0]):
        print("   ⚠️  PROBLÈME: Tous les rewards sont identiques!")
        return False
    else:
        print("   ✅ Les rewards varient pendant l'épisode")
        return True

def test_normalization():
    """Test de la normalisation.
    
    NOTE: La normalisation interne (ObservationNormalizer) est desactivee.
    La normalisation est geree par VecNormalize dans le trainer PPO.
    Ce test verifie que les observations brutes sont coherentes.
    """
    print("\n" + "=" * 60)
    print("TEST 4: OBSERVATIONS (VecNormalize gere la normalisation)")
    print("=" * 60)
    
    # L'environnement retourne maintenant des observations brutes
    # VecNormalize dans ppo_trainer.py s'occupe de la normalisation
    config = PDPEnvironmentConfig(
        n_products=1,
        horizon=12,
        normalize_observations=True  # Active VecNormalize dans le trainer
    )
    env = EnvironmentRegistry.create('strategic', config)
    obs, info = env.reset(seed=42)
    
    print("Observations brutes (normalisees par VecNormalize lors de l'entrainement):")
    print(f"   Stock initial: {obs['current_stock']}")
    print(f"   Demandes futures shape: {obs['future_demands'].shape}")
    print(f"   Demandes (min/max): {obs['future_demands'].min():.1f} / "
          f"{obs['future_demands'].max():.1f}")
    
    # Verifications de coherence des observations brutes
    stock = obs['current_stock']
    demands = obs['future_demands']
    
    # Le stock doit etre dans une plage raisonnable
    max_stock = np.array(config.max_stock)
    stock_valid = np.all(stock >= -max_stock) and np.all(stock <= max_stock)
    
    # Les demandes doivent etre positives
    demands_valid = np.all(demands >= 0)
    
    if stock_valid and demands_valid:
        print("   ✅ Observations brutes coherentes")
        print("   ℹ️  VecNormalize normalisera ces valeurs pendant l'entrainement")
        return True
    else:
        if not stock_valid:
            print("   ⚠️  Stock hors limites!")
        if not demands_valid:
            print("   ⚠️  Demandes negatives detectees!")
        return False

def main():
    """Lance tous les tests"""
    print("\n🔍 DIAGNOSTIC DE L'ENVIRONNEMENT PDP\n")
    
    tests = [
        ("Fonctionnalités de base", test_basic_functionality),
        ("Échelle des rewards", test_reward_scale),
        ("Cohérence d'épisode", test_episode_consistency),
        ("Normalisation", test_normalization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ ERREUR dans {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\n{passed}/{len(results)} tests réussis")
    
    if passed == len(results):
        print("\n🎉 Tous les tests sont passés! L'environnement est prêt.")
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les messages ci-dessus.")

if __name__ == "__main__":
    main()
