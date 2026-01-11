# agents/agent_registry.py
"""
Registre centralise des agents RL pour le PDP.

Fournit un Factory pattern pour instancier les agents par leur nom.
Supporte aussi les agents de Stable-Baselines3.
"""

from typing import Dict, Type, Any, Optional, Union, List
from dataclasses import dataclass, field

from environments.base_pdp_env import BasePDPEnvironment


@dataclass
class AgentConfig:
    """Configuration generique pour un agent"""
    # Hyperparametres communs
    gamma: float = 0.99
    learning_rate: float = 3e-4
    seed: Optional[int] = None
    device: Optional[str] = None
    
    # Hyperparametres specifiques (passes en **kwargs)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Registre centralise pour tous les agents RL.
    
    Utilisation:
        # Lister les agents disponibles
        agents = AgentRegistry.list_available()
        
        # Creer un agent
        agent = AgentRegistry.create('sac', env, config)
        
        # Obtenir les infos sur un agent
        info = AgentRegistry.get_info('dqn')
    """
    
    _agents: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        agent_class: Type,
        description: str = "",
        category: str = "custom",
        supports_continuous: bool = True,
        supports_discrete: bool = False,
        requires_discretization: bool = False
    ):
        """
        Enregistre un nouvel agent.
        
        Args:
            name: Nom unique de l'agent
            agent_class: Classe de l'agent
            description: Description de l'agent
            category: Categorie (tabular, deep, actor-critic, etc.)
            supports_continuous: Supporte les actions continues
            supports_discrete: Supporte les actions discretes
            requires_discretization: Necessite une discretisation des actions
        """
        cls._agents[name] = {
            'class': agent_class,
            'description': description,
            'category': category,
            'supports_continuous': supports_continuous,
            'supports_discrete': supports_discrete,
            'requires_discretization': requires_discretization
        }
    
    @classmethod
    def create(
        cls, 
        name: str, 
        env: BasePDPEnvironment, 
        config: Optional[AgentConfig] = None,
        **kwargs
    ) -> Any:
        """
        Cree une instance d'agent.
        
        Args:
            name: Nom de l'agent
            env: Environnement PDP
            config: Configuration de l'agent
            **kwargs: Arguments supplementaires
            
        Returns:
            Instance de l'agent
        """
        if name not in cls._agents:
            raise ValueError(
                f"Agent '{name}' non trouve. "
                f"Disponibles: {list(cls._agents.keys())}"
            )
        
        agent_info = cls._agents[name]
        agent_class = agent_info['class']
        
        # Obtenir les parametres acceptes par le constructeur de l'agent
        import inspect
        sig = inspect.signature(agent_class.__init__)
        accepted_params = set(sig.parameters.keys()) - {'self'}
        
        # Fusionner la config avec les kwargs
        if config is not None:
            params = {
                'gamma': config.gamma,
                'seed': config.seed,
                **config.extra_params,
                **kwargs
            }
            
            # Ajouter device seulement si l'agent le supporte
            if 'device' in accepted_params and config.device is not None:
                params['device'] = config.device
            
            # Learning rate peut avoir un nom different selon l'agent
            if 'learning_rate' not in kwargs:
                if 'alpha' in accepted_params:
                    # Agents tabulaires utilisent 'alpha'
                    params['alpha'] = config.learning_rate
                elif 'learning_rate' in accepted_params:
                    params['learning_rate'] = config.learning_rate
        else:
            params = kwargs
        
        # Filtrer les parametres non acceptes
        filtered_params = {k: v for k, v in params.items() if k in accepted_params}
        
        return agent_class(env, **filtered_params)
    
    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Retourne les informations sur un agent"""
        if name not in cls._agents:
            raise ValueError(f"Agent '{name}' non trouve.")
        return cls._agents[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Liste les agents disponibles"""
        return list(cls._agents.keys())
    
    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """Liste les agents d'une categorie"""
        return [
            name for name, info in cls._agents.items() 
            if info['category'] == category
        ]
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Liste les categories disponibles"""
        return list(set(info['category'] for info in cls._agents.values()))
    
    @classmethod
    def print_summary(cls):
        """Affiche un resume de tous les agents"""
        print("\n" + "=" * 60)
        print("AGENTS RL DISPONIBLES")
        print("=" * 60)
        
        categories = cls.get_categories()
        for category in sorted(categories):
            print(f"\n📁 {category.upper()}")
            print("-" * 40)
            
            for name in cls.list_by_category(category):
                info = cls._agents[name]
                desc = info['description'][:50] + "..." if len(info['description']) > 50 else info['description']
                
                # Indicateurs
                indicators = []
                if info['supports_continuous']:
                    indicators.append("🔄")
                if info['supports_discrete']:
                    indicators.append("🔢")
                if info['requires_discretization']:
                    indicators.append("⚙️")
                
                print(f"  • {name}: {desc} {' '.join(indicators)}")
        
        print("\n" + "=" * 60)
        print("Legende: 🔄 Actions continues | 🔢 Actions discretes | ⚙️ Discretisation requise")
        print("=" * 60 + "\n")


def register_all_agents():
    """Enregistre tous les agents disponibles"""
    
    # ===== Agents Tabulaires =====
    from .tabular_agents import MonteCarloAgent, QLearningAgent, SARSAAgent
    
    AgentRegistry.register(
        'monte_carlo',
        MonteCarloAgent,
        description="Monte Carlo First-Visit avec discretisation",
        category="tabular",
        supports_continuous=False,
        supports_discrete=True,
        requires_discretization=True
    )
    
    AgentRegistry.register(
        'q_learning',
        QLearningAgent,
        description="Q-Learning (off-policy TD) avec discretisation",
        category="tabular",
        supports_continuous=False,
        supports_discrete=True,
        requires_discretization=True
    )
    
    AgentRegistry.register(
        'sarsa',
        SARSAAgent,
        description="SARSA (on-policy TD) avec discretisation",
        category="tabular",
        supports_continuous=False,
        supports_discrete=True,
        requires_discretization=True
    )
    
    # ===== Deep RL =====
    from .dqn_agent import DQNAgent
    
    AgentRegistry.register(
        'dqn',
        DQNAgent,
        description="Deep Q-Network avec replay buffer et target network",
        category="deep-q",
        supports_continuous=False,
        supports_discrete=True,
        requires_discretization=True
    )
    
    # ===== Actor-Critic =====
    from .a2c_agent import A2CAgent
    
    AgentRegistry.register(
        'a2c',
        A2CAgent,
        description="Advantage Actor-Critic (synchrone) pour actions continues",
        category="actor-critic",
        supports_continuous=True,
        supports_discrete=False,
        requires_discretization=False
    )
    
    from .sac_agent import SACAgent
    
    AgentRegistry.register(
        'sac',
        SACAgent,
        description="Soft Actor-Critic avec maximum entropy",
        category="actor-critic",
        supports_continuous=True,
        supports_discrete=False,
        requires_discretization=False
    )
    
    # ===== Stable-Baselines3 (wrapper) =====
    # On peut aussi integrer les agents SB3 via un wrapper
    try:
        from .ppo_trainer import PPOTrainer
        
        AgentRegistry.register(
            'ppo',
            PPOTrainer,
            description="Proximal Policy Optimization (via Stable-Baselines3)",
            category="stable-baselines3",
            supports_continuous=True,
            supports_discrete=True,
            requires_discretization=False
        )
    except ImportError:
        pass  # SB3 pas disponible


# Enregistrer automatiquement les agents au chargement du module
register_all_agents()
