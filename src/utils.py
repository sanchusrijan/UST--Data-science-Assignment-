# src/utils.py

import time
from typing import Callable, Any, Tuple, List, Dict
import numpy as np

def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and return function execution time in milliseconds."""
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        return result, elapsed_ms
    return wrapper

def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes precision, recall, and F1-score for each class and overall micro/macro.
    
    Args:
        y_true: True labels list.
        y_pred: Predicted labels list.
        
    Returns:
        Dict containing class-wise and global metrics.
    """
    classes = sorted(list(set(y_true + y_pred)))
    metrics: Dict[str, Any] = {}
    
    # Class-wise metrics
    for c in classes:
        tp = sum(1 for gt, pd in zip(y_true, y_pred) if gt == c and pd == c)
        fp = sum(1 for gt, pd in zip(y_true, y_pred) if gt != c and pd == c)
        fn = sum(1 for gt, pd in zip(y_true, y_pred) if gt == c and pd != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": sum(1 for gt in y_true if gt == c)
        }
        
    # Global macro average
    macro_p = np.mean([metrics[c]["precision"] for c in classes])
    macro_r = np.mean([metrics[c]["recall"] for c in classes])
    macro_f1 = np.mean([metrics[c]["f1-score"] for c in classes])
    
    # Global accuracy
    correct = sum(1 for gt, pd in zip(y_true, y_pred) if gt == pd)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
    
    metrics["macro_avg"] = {
        "precision": float(macro_p),
        "recall": float(macro_r),
        "f1-score": float(macro_f1)
    }
    metrics["accuracy"] = float(accuracy)
    
    return metrics

def print_metrics_table(metrics: Dict[str, Any]) -> None:
    """Helper to print metrics dict in a beautiful text table."""
    print(f"{'Class':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for k, v in metrics.items():
        if k in {"macro_avg", "accuracy"}:
            continue
        print(f"{k:<30} | {v['precision']:10.4f} | {v['recall']:10.4f} | {v['f1-score']:10.4f} | {v['support']:8d}")
    print("-" * 75)
    print(f"{'Accuracy':<30} | {'':<10} | {'':<10} | {metrics['accuracy']:10.4f} | {sum(v['support'] for k, v in metrics.items() if k not in {'macro_avg', 'accuracy'}):8d}")
    macro = metrics["macro_avg"]
    print(f"{'Macro Avg':<30} | {macro['precision']:10.4f} | {macro['recall']:10.4f} | {macro['f1-score']:10.4f} |")

def compute_confusion_matrix(y_true: List[str], y_pred: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Computes a standard confusion matrix and list of classes."""
    classes = sorted(list(set(y_true + y_pred)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for gt, pd in zip(y_true, y_pred):
        matrix[class_to_idx[gt], class_to_idx[pd]] += 1
        
    return matrix, classes

def plot_confusion_matrix_text(matrix: np.ndarray, classes: List[str]) -> None:
    """Prints an ASCII text-based confusion matrix for quick console inspection."""
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    # Header
    header = f"{'True \\ Pred':<22} | " + " | ".join(f"{c[:4]:<4}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(matrix):
        row_str = f"{classes[i][:20]:<22} | " + " | ".join(f"{val:<4}" for val in row)
        print(row_str)
