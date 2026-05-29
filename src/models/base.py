# src/models/base.py

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseCommandClassifier(ABC):
    """Abstract base class for all speech command classifiers."""
    
    @abstractmethod
    def train(self, train_data: list[dict[str, str]]) -> None:
        """Trains the model on a list of label-text dicts."""
        pass
        
    @abstractmethod
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predicts the command for the given text.
        
        Args:
            text: Input text string transcribed from ASR.
            
        Returns:
            A tuple of (predicted_class_name, confidence_score).
            If the query is Out-of-Scope (OOS), returns ("None", score).
        """
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the model state to a file."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the model state from a file."""
        pass
