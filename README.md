# Light Weight Speech Command Classifier for Edge Deployment

This repository contains the complete implementation for the **Light Weight Speech Command Classifier** assignment. The system is designed to classify text transcripts produced by an Automatic Speech Recognition (ASR) engine in an offline, low-latency automotive/IoT cockpit voice assistant.

The project evaluates and compares two distinct architectures:
1. **TF-IDF + Logistic Regression (Baseline):** A character/word n-gram baseline serialized to JSON and executed via a custom, dependency-free NumPy runtime.
2. **Quantized ONNX Sentence Transformer (Advanced):** An `all-MiniLM-L6-v2` transformer model exported to ONNX and dynamically quantized to INT8, utilizing few-shot template similarity.

---

## Project Structure

```
.
├── Approach_Document.md       # Full architecture explanation and design decisions
├── README.md                  # Setup and execution instructions
├── requirements.txt           # Python package dependencies
├── src/
│   ├── __init__.py
│   ├── config.py              # Single source of truth for command maps and templates
│   ├── dataset.py             # Synthetic data generation and ASR/Indian accent simulator
│   ├── api.py                 # Unified pipeline interface (normalization + classification)
│   ├── utils.py               # Evaluation helper metrics, timers, and matrix plotting
│   └── models/
│       ├── __init__.py
│       ├── base.py            # Abstract base class for classifiers
│       ├── tfidf_classifier.py# TF-IDF baseline with numpy-only inference
│       └── onnx_transformer.py# Quantized ONNX MiniLM similarity classifier
├── scripts/
│   ├── train_tfidf.py         # Trains and serializes the TF-IDF model
│   ├── export_onnx.py         # Downloads, exports, and quantizes the MiniLM model
│   ├── benchmark.py           # Runs complete accuracy, latency, and size benchmarks
│   └── interactive.py         # Real-time interactive command-line interface
└── tests/
    └── test_classifier.py     # Automated unit test suite
```

---

## Requirements & Environment Setup

This project requires **Python 3.11+** (fully compatible with Python 3.14+).

1. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage Guide

To run the pipeline from end-to-end, execute the following commands in order:

### 1. Train the Baseline TF-IDF Classifier
Generates the synthetic dataset (incorporating ASR noise and Indian English accents) and trains the Logistic Regression baseline:
```bash
python scripts/train_tfidf.py
```
*Output weights are saved to `models/tfidf_model.json` (~300 KB).*

### 2. Download, Export, and Quantize the Transformer
Downloads the pre-trained `all-MiniLM-L6-v2` Sentence-Transformer, converts it to ONNX, and applies dynamic INT8 quantization:
```bash
python scripts/export_onnx.py
```
*Outputs are saved to `models/onnx_minilm/` (`model_quantized.onnx` is ~22 MB).*

### 3. Run the Performance & Robustness Benchmark
Runs a comprehensive evaluation of both models on clean and noisy inputs, benchmarks CPU latency, and outputs a summary report:
```bash
python scripts/benchmark.py
```
*A Markdown summary is written to `models/benchmark_report.md`.*

### 4. Run the Real-Time Interactive Demo
Test the models live using the console. You can input clean commands, noisy commands with typos, or out-of-scope text:
```bash
python scripts/interactive.py
```

### 5. Run the Unit Tests
Execute the automated test suite to ensure the data pipelines and classifiers are working as expected:
```bash
pytest tests/
```

---

## Deliverables & Documentation
*   **Approach Document:** Found at [Approach_Document.md](file:///Users/sasikala/Desktop/UST-Data%20science/Approach_Document.md). Explains the design logic, ASR noise simulation, Indian accent modeling, and the architecture trade-offs.
*   **Self-Assessment:** Found at [Self_Assessment.md](file:///Users/sasikala/Desktop/UST-Data%20science/Self_Assessment.md). Details what works well, what does not, and paths to production deployment.
*   **Evaluation Summary:** Generated dynamically by `scripts/benchmark.py` and saved to `models/benchmark_report.md`.
