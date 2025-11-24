# environments/env_registry.py

from typing import Dict, Type, List
from .base_pdp_env import BasePDPEnvironment
from .strategic_pdp_env import StrategicPDPEnvironment

class EnvironmentRegistry:
    """Registre des environnements disponibles"""
    
    _environments: Dict[str, Type[BasePDPEnvironment]] = {
        'base': BasePDPEnvironment,
        'strategic': StrategicPDPEnvironment,
    }
    
    @classmethod
    def register(cls, name: str, env_class: Type[BasePDPEnvironment]):
        """Enregistre un nouvel environnement"""
        cls._environments[name] = env_class
    
    @classmethod
    def create(cls, name: str, config) -> BasePDPEnvironment:
        """Crée une instance d'environnement"""
        if name not in cls._environments:
            raise ValueError(f"Environnement {name} non trouvé. Disponibles: {list(cls._environments.keys())}")
        
        return cls._environments[name](config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Liste les environnements disponibles"""
        return list(cls._environments.keys())