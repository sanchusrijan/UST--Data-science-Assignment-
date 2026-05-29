# scripts/benchmark.py

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Tuple

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import generate_splits
from src.api import VoiceCommandAPI
from src.utils import compute_metrics, compute_confusion_matrix, plot_confusion_matrix_text

def get_file_size_mb(path: str) -> float:
    """Returns file size in Megabytes."""
    if os.path.isdir(path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    elif os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

def benchmark_model(api: VoiceCommandAPI, test_data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Runs a complete evaluation benchmark on a given model and dataset."""
    y_true = [d["label"] for d in test_data]
    y_pred = []
    latencies = []
    
    # Warm up run
    _ = api.classify("test query")
    
    for item in test_data:
        text = item["text"]
        
        # Measure latency
        start = time.perf_counter()
        res = api.classify(text)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000.0) # in ms
        y_pred.append(res["prediction"])
        
    metrics = compute_metrics(y_true, y_pred)
    
    return {
        "metrics": metrics,
        "latencies": latencies,
        "y_true": y_true,
        "y_pred": y_pred
    }

def run_benchmarks() -> None:
    """Runs latency, accuracy, and size benchmarks for both models."""
    print("Loading test splits...")
    _, test_clean, test_noisy = generate_splits(include_extension=True)
    
    print(f"Clean test set size: {len(test_clean)}")
    print(f"Noisy test set size: {len(test_noisy)}")
    
    # Check paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfidf_path = os.path.join(project_root, "models", "tfidf_model.json")
    onnx_dir = os.path.join(project_root, "models", "onnx_minilm")
    
    if not os.path.exists(tfidf_path):
        print("TF-IDF model not found. Training baseline model first...")
        os.system(f"{sys.executable} scripts/train_tfidf.py")
        
    if not os.path.exists(os.path.join(onnx_dir, "model_quantized.onnx")):
        print("Quantized ONNX model not found. Exporting ONNX model first...")
        os.system(f"{sys.executable} scripts/export_onnx.py")
        
    # Load APIs
    print("\nInitializing TF-IDF Classifier...")
    tfidf_api = VoiceCommandAPI(model_type="tfidf")
    
    print("Initializing ONNX Sentence Transformer...")
    onnx_api = VoiceCommandAPI(model_type="onnx")
    
    results: Dict[str, Dict[str, Any]] = {}
    
    # Benchmark models
    for name, api in [("TF-IDF Baseline", tfidf_api), ("Quantized ONNX MiniLM", onnx_api)]:
        print(f"\nBenchmarking {name} on Clean Test Set...")
        clean_res = benchmark_model(api, test_clean, name)
        
        print(f"Benchmarking {name} on Noisy Test Set...")
        noisy_res = benchmark_model(api, test_noisy, name)
        
        # Calculate size
        if "TF-IDF" in name:
            model_size = get_file_size_mb(tfidf_path)
        else:
            model_size = get_file_size_mb(os.path.join(onnx_dir, "model_quantized.onnx"))
            
        results[name] = {
            "clean": clean_res,
            "noisy": noisy_res,
            "size_mb": model_size
        }
        
    # Generate MD Report
    print("\n" + "="*40 + "\n          BENCHMARK REPORT          \n" + "="*40)
    
    md_report = []
    md_report.append("# Benchmark Evaluation Report\n")
    md_report.append("| Model | Size (MB) | Avg Latency (ms) | P95 Latency (ms) | Clean Accuracy | Noisy Accuracy | OOS Rejection Rate | False Rejection Rate |")
    md_report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, res in results.items():
        clean_metrics = res["clean"]["metrics"]
        noisy_metrics = res["noisy"]["metrics"]
        latencies = res["clean"]["latencies"]
        
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        
        # OOS stats (True Rejection Rate: percentage of OOS correctly labeled None)
        # OOS samples have ground truth label 'None'
        clean_y_true = res["clean"]["y_true"]
        clean_y_pred = res["clean"]["y_pred"]
        
        oos_true = sum(1 for gt, pd in zip(clean_y_true, clean_y_pred) if gt == "None" and pd == "None")
        oos_total = sum(1 for gt in clean_y_true if gt == "None")
        oos_rejection_rate = oos_true / oos_total if oos_total > 0 else 1.0
        
        # False Rejection Rate (FRR: percentage of valid commands incorrectly rejected as None)
        valid_total = sum(1 for gt in clean_y_true if gt != "None")
        false_reject = sum(1 for gt, pd in zip(clean_y_true, clean_y_pred) if gt != "None" and pd == "None")
        false_rejection_rate = false_reject / valid_total if valid_total > 0 else 0.0
        
        md_line = (
            f"| **{name}** | {res['size_mb']:.2f} MB | {avg_lat:.2f} ms | {p95_lat:.2f} ms | "
            f"{clean_metrics['accuracy']:.2%} | {noisy_metrics['accuracy']:.2%} | "
            f"{oos_rejection_rate:.2%} | {false_rejection_rate:.2%} |"
        )
        md_report.append(md_line)
        print(md_line.replace("|", " "))
        
    print("\n" + "="*40)
    
    # Dynamic Noise Breakdown
    md_report.append("\n## Accuracy Breakdown by ASR Noise Type\n")
    md_report.append("| Model | Dropped Articles | Filler Words | Indian Phonetics | Indian Phrasing | Typo/Edits | Out of Scope |")
    md_report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, res in results.items():
        y_true = res["noisy"]["y_true"]
        y_pred = res["noisy"]["y_pred"]
        
        noise_types = [d.get("noise_type", "unknown") for d in test_noisy]
        
        breakdown: Dict[str, float] = {}
        unique_noises = set(noise_types)
        for noise in unique_noises:
            correct = sum(1 for gt, pd, nt in zip(y_true, y_pred, noise_types) if nt == noise and gt == pd)
            total = sum(1 for nt in noise_types if nt == noise)
            breakdown[noise] = correct / total if total > 0 else 0.0
            
        md_line = (
            f"| **{name}** | "
            f"{breakdown.get('dropped_articles', 0.0):.2%} | "
            f"{breakdown.get('filler_words', 0.0):.2%} | "
            f"{breakdown.get('indian_phonetics', 0.0):.2%} | "
            f"{breakdown.get('indian_phrasing', 0.0):.2%} | "
            f"{breakdown.get('phonetic_typos', 0.0):.2%} | "
            f"{breakdown.get('out_of_scope', 0.0):.2%} |"
        )
        md_report.append(md_line)
        
    # Write report file
    report_path = os.path.join(project_root, "models", "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md_report))
    print(f"\nFull benchmark markdown report saved to {report_path}")

    # Output Sample Confusion Matrix for the winning model (ONNX)
    print("\nConfusion Matrix for the Quantized ONNX Model:")
    onnx_res = results["Quantized ONNX MiniLM"]["clean"]
    matrix, classes = compute_confusion_matrix(onnx_res["y_true"], onnx_res["y_pred"])
    plot_confusion_matrix_text(matrix, classes)

if __name__ == "__main__":
    run_benchmarks()
