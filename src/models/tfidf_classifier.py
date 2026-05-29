# src/models/tfidf_classifier.py

import json
import math
import numpy as np
from typing import Tuple, Dict, Any, List
from src.models.base import BaseCommandClassifier
from src.config import ALL_COMMANDS, TFIDF_CONFIDENCE_THRESHOLD

class TFIDFCommandClassifier(BaseCommandClassifier):
    """
    TF-IDF + Logistic Regression command classifier.
    Supports standard training via scikit-learn and lightweight,
    dependency-free inference via pure Python/numpy.
    """
    
    def __init__(self, confidence_threshold: float = TFIDF_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold: float = confidence_threshold
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []
        self.coef: List[List[float]] = []
        self.intercept: List[float] = []
        self.classes: List[str] = []
        self.sublinear_tf: bool = True
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words and n-grams."""
        # Simple word tokenization (strip punctuation, convert to lowercase)
        clean_text = "".join([c.lower() if c.isalnum() or c.isspace() else " " for c in text])
        words = [w for w in clean_text.split() if w]
        
        # Extract word unigrams and bigrams (ngram_range=(1,2))
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]} {words[i+1]}")
        return tokens

    def train(self, train_data: List[Dict[str, str]]) -> None:
        """Trains the model using scikit-learn."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        texts = [d["text"] for d in train_data]
        labels = [d["label"] for d in train_data]
        
        # Fit vectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"\b\w+\b",
            sublinear_tf=self.sublinear_tf,
            smooth_idf=True
        )
        X = vectorizer.fit_transform(texts)
        
        # Fit Logistic Regression
        clf = LogisticRegression(max_iter=1000, C=2.0, class_weight='balanced')
        clf.fit(X, labels)
        
        # Export parameters
        self.vocabulary = vectorizer.vocabulary_
        self.idf = vectorizer.idf_.tolist()
        self.coef = clf.coef_.tolist()
        self.intercept = clf.intercept_.tolist()
        self.classes = clf.classes_.tolist()
        
    def _extract_tfidf_vector(self, text: str) -> np.ndarray:
        """Extracts a normalized TF-IDF vector in pure Python/numpy."""
        tokens = self._tokenize(text)
        vector = np.zeros(len(self.vocabulary))
        
        # Count term frequencies (TF)
        tf_counts: Dict[str, int] = {}
        for token in tokens:
            if token in self.vocabulary:
                tf_counts[token] = tf_counts.get(token, 0) + 1
                
        # Calculate TF-IDF
        for token, count in tf_counts.items():
            idx = self.vocabulary[token]
            # Apply sublinear TF scaling: 1 + log(tf)
            tf_val = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vector[idx] = tf_val * self.idf[idx]
            
        # L2 Normalization
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector

    def predict(self, text: str) -> Tuple[str, float]:
        """Runs fast inference using numpy softmax."""
        if not self.vocabulary:
            raise ValueError("Model has not been trained or loaded yet.")
            
        # 1. Extract TF-IDF vector
        x = self._extract_tfidf_vector(text)
        
        # 2. Compute decision function: z = x * W^T + b
        coef_matrix = np.array(self.coef)  # shape (num_classes, num_features)
        intercepts = np.array(self.intercept)  # shape (num_classes,)
        
        logits = np.dot(coef_matrix, x) + intercepts
        
        # 3. Softmax activation
        exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # 4. Find winning class
        best_idx = np.argmax(probs)
        pred_class = self.classes[best_idx]
        confidence = float(probs[best_idx])
        
        # 5. Out-of-Scope (OOS) rejection threshold
        if confidence < self.confidence_threshold:
            return "None", confidence
            
        return pred_class, confidence

    def save(self, path: str) -> None:
        """Saves weights and parameters to a JSON file."""
        model_data = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "coef": self.coef,
            "intercept": self.intercept,
            "classes": self.classes,
            "sublinear_tf": self.sublinear_tf,
            "confidence_threshold": self.confidence_threshold
        }
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

    def load(self, path: str) -> None:
        """Loads weights and parameters from a JSON file."""
        with open(path, "r") as f:
            model_data = json.load(f)
            
        self.vocabulary = model_data["vocabulary"]
        self.idf = model_data["idf"]
        self.coef = model_data["coef"]
        self.intercept = model_data["intercept"]
        self.classes = model_data["classes"]
        self.sublinear_tf = model_data.get("sublinear_tf", True)
        self.confidence_threshold = model_data.get("confidence_threshold", TFIDF_CONFIDENCE_THRESHOLD)
