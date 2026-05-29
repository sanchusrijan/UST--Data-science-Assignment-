# scripts/interactive.py

import os
import sys
import time

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import VoiceCommandAPI

def main() -> None:
    """Interactive loop to run live classification tests on the console."""
    print("=" * 60)
    print("   UST SPEECH COMMAND CLASSIFIER - INTERACTIVE DEMO")
    print("=" * 60)
    print("Select Model Backend:")
    print("  1. TF-IDF + Logistic Regression (Baseline, ~0.1MB, extremely fast)")
    print("  2. Quantized ONNX Sentence Transformer (Advanced, ~22MB, highly semantic)")
    
    choice = input("\nEnter choice (1 or 2, default 2): ").strip()
    model_type = "tfidf" if choice == "1" else "onnx"
    
    print(f"\nInitializing {model_type.upper()} API...")
    try:
        api = VoiceCommandAPI(model_type=model_type)
        print("Model initialized successfully! Type your commands below.")
        print("Type 'exit' or 'quit' to stop.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have run the training/export scripts first:")
        print("  python scripts/train_tfidf.py")
        print("  python scripts/export_onnx.py")
        sys.exit(1)
        
    print(f"Confidence Threshold set to:")
    print(f"  - {api.classifier.confidence_threshold} (Rejects queries scoring lower)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nVoice Command Transcript > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
                
            # Run prediction and time it
            start = time.perf_counter()
            result = api.classify(user_input)
            end = time.perf_counter()
            
            elapsed_ms = (end - start) * 1000.0
            
            print(f"  Normalized Input : '{result['normalized']}'")
            print(f"  Prediction       : \033[1;32m{result['prediction']}\033[0m" if result['prediction'] != 'None' else f"  Prediction       : \033[1;31mREJECTED (None)\033[0m")
            print(f"  Confidence Score : {result['score']:.4f}")
            print(f"  Pipeline Latency : {elapsed_ms:.2f} ms")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing input: {e}")

if __name__ == "__main__":
    main()
