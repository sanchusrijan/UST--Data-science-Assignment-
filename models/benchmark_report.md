# Benchmark Evaluation Report

| Model | Size (MB) | Avg Latency (ms) | P95 Latency (ms) | Clean Accuracy | Noisy Accuracy | OOS Rejection Rate | False Rejection Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.29 MB | 0.34 ms | 0.35 ms | 39.58% | 28.27% | 83.33% | 50.00% |
| **Quantized ONNX MiniLM** | 21.80 MB | 20.73 ms | 21.00 ms | 100.00% | 84.82% | 100.00% | 0.00% |

## Accuracy Breakdown by ASR Noise Type

| Model | Dropped Articles | Filler Words | Indian Phonetics | Indian Phrasing | Typo/Edits | Out of Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 41.18% | 16.67% | 14.29% | 40.48% | 28.57% | 83.33% |
| **Quantized ONNX MiniLM** | 100.00% | 100.00% | 69.05% | 100.00% | 61.90% | 100.00% |