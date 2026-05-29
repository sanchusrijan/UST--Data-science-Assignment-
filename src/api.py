# src/api.py

import os
from typing import Tuple, Dict, Any
from src.models.base import BaseCommandClassifier
from src.models.tfidf_classifier import TFIDFCommandClassifier
from src.models.onnx_transformer import ONNXTransformerClassifier


class VoiceCommandAPI:
    """
    Unified Pipeline API for Voice Command Classification.
    Integrates text normalization, classification, and OOS check.
    """
    
    def __init__(self, model_type: str = "onnx") -> None:
        """
        Initializes the API.
        
        Args:
            model_type: Either 'tfidf' or 'onnx'.
        """
        self.model_type: str = model_type.lower()
        self.classifier: BaseCommandClassifier
        
        # Resolve paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if self.model_type == "tfidf":
            model_path = os.path.join(project_root, "models", "tfidf_model.json")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"TF-IDF model file not found at {model_path}. "
                    "Run 'python scripts/train_tfidf.py' first."
                )
            self.classifier = TFIDFCommandClassifier()
            self.classifier.load(model_path)
            
        elif self.model_type == "onnx":
            model_dir = os.path.join(project_root, "models", "onnx_minilm")
            meta_path = os.path.join(model_dir, "onnx_metadata.json")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(
                    f"ONNX model directory not found at {model_dir}. "
                    "Run 'python scripts/export_onnx.py' first."
                )
            self.classifier = ONNXTransformerClassifier(model_dir=model_dir)
            self.classifier.load(meta_path)
            
        else:
            raise ValueError("Invalid model_type. Must be 'tfidf' or 'onnx'.")

    def _normalize(self, text: str) -> str:
        """Applies basic text normalization (cleaning whitespace, lowercasing)."""
        if not text:
            return ""
        # Lowers and cleans redundant spaces
        return " ".join(text.lower().strip().split())

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Processes a raw input text string and outputs a classification result.
        
        Args:
            text: Raw transcript string from the ASR engine.
            
        Returns:
            Dict containing the normalized text, predicted command, and confidence score.
        """
        normalized_text = self._normalize(text)
        if not normalized_text:
            return {
                "input": text,
                "normalized": "",
                "prediction": "None",
                "score": 0.0,
                "status": "rejected"
            }
            
        pred_label, score = self.classifier.predict(normalized_text)
        
        return {
            "input": text,
            "normalized": normalized_text,
            "prediction": pred_label,
            "score": round(score, 4),
            "status": "accepted" if pred_label != "None" else "rejected"
        }
