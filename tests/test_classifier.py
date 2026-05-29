# tests/test_classifier.py

import os
import sys
import pytest

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import generate_splits, drop_articles, apply_indian_accent_phonetics
from src.api import VoiceCommandAPI

def test_dataset_generation() -> None:
    """Verifies that splits are generated correctly and have expected labels."""
    train_data, test_clean, test_noisy = generate_splits(include_extension=True)
    
    assert len(train_data) > 0
    assert len(test_clean) > 0
    assert len(test_noisy) > 0
    
    # Check that labels are valid
    valid_labels = {
        "activate do not disturb", "deactivate do not disturb", "decline the call",
        "pick up the call", "play the music", "pause the music",
        "play the next song", "play the previous song", "increase the volume",
        "decrease the volume", "increase the brightness", "decrease the brightness",
        "start the vehicle", "stop the vehicle", "None"
    }
    
    for item in train_data:
        assert item["label"] in valid_labels
        assert len(item["text"]) > 0

def test_drop_articles() -> None:
    """Tests the article dropping text augmenter."""
    assert drop_articles("play the music") == "play music"
    assert drop_articles("turn off a device") == "turn off device"
    assert drop_articles("ignore an incoming call") == "ignore incoming call"

def test_indian_accent_phonetics() -> None:
    """Tests that phonetic swaps are applied to known terms."""
    # Ensure swapping operates (may be probabilistic so we run multiple times or check vocabulary keys)
    text = "volume vehicle music"
    # Over many runs, at least some phonetic variations should be produced
    variants = set()
    for _ in range(50):
        variants.add(apply_indian_accent_phonetics(text))
    
    assert len(variants) > 1  # Verify variation is generated
    
def test_tfidf_classifier_flow() -> None:
    """Tests the full API flow using the TF-IDF baseline."""
    # Run API initialization (should load from models/tfidf_model.json)
    api = VoiceCommandAPI(model_type="tfidf")
    
    # Test a clean in-scope command
    res1 = api.classify("increase the volume")
    assert res1["normalized"] == "increase the volume"
    assert res1["prediction"] == "increase the volume"
    assert res1["score"] > 0.5
    assert res1["status"] == "accepted"
    
    # Test a clear out-of-scope command
    res2 = api.classify("what is the weather like today in Seattle")
    assert res2["prediction"] == "None"
    assert res2["status"] == "rejected"
