# agents/rl_utils.py

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback pour sauvegarder le meilleur modèle basé sur le reward moyen.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Créer le répertoire si nécessaire
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Récupérer les résultats
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          
          if len(x) > 0:
              # Calculer le reward moyen sur les 100 derniers épisodes
              mean_reward = np.mean(y[-100:])
              
              if self.verbose > 0:
                  print(f"Num timesteps: {self.num_timesteps}")
                  print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

              # Nouveau meilleur modèle, on sauvegarde
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                      print(f"Sauvegarde du nouveau meilleur modèle à {self.save_path} (reward: {self.best_mean_reward:.2f})")
                  self.model.save(self.save_path)

        return True

class EarlyStoppingCallback(BaseCallback):
    """
    Callback pour arrêter l'entraînement si le reward moyen ne s'améliore pas
    pendant un certain nombre de vérifications.
    """
    def __init__(self, check_freq: int, patience: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    
                if self.verbose > 0:
                    print(f"EarlyStopping - No improvement count: {self.no_improvement_count}/{self.patience}")

                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print("Arrêt prématuré: le reward moyen n'a pas progressé.")
                    return False # Arrête l'entraînement

        return True