# scripts/train_tfidf.py

import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import generate_splits
from src.models.tfidf_classifier import TFIDFCommandClassifier

def train_and_save() -> None:
    """Generates synthetic data, trains TF-IDF Logistic Regression, and serializes parameters."""
    print("Generating training and test splits...")
    train_data, test_clean, test_noisy = generate_splits(include_extension=True)
    
    print(f"Total training data size: {len(train_data)} samples.")
    print("Training TF-IDF model...")
    
    classifier = TFIDFCommandClassifier()
    classifier.train(train_data)
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_path = "models/tfidf_model.json"
    
    print(f"Saving trained model coefficients and vocabulary to {model_path}...")
    classifier.save(model_path)
    print("TF-IDF Model training and serialization complete!")

if __name__ == "__main__":
    train_and_save()
