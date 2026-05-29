# scripts/export_onnx.py

import os
import sys
import shutil
from typing import Any

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def export_and_quantize() -> None:
    """Downloads, exports, and quantizes MiniLM to ONNX format."""
    print("Starting ONNX export and quantization process...")
    
    # Import optimum inside function to ensure environment is fully loaded
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error importing Optimum/Transformers: {e}")
        print("Please ensure you have activated your virtual environment and installed requirements.txt.")
        sys.exit(1)
        
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "models/onnx_minilm"
    
    # 1. Download and Export standard ONNX model
    print(f"Downloading and exporting '{model_id}' to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Save base ONNX model and tokenizer config
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Base ONNX model exported to: {output_dir}")
    
    # Verify the unquantized model size
    base_onnx_path = os.path.join(output_dir, "model.onnx")
    base_size_mb = os.path.getsize(base_onnx_path) / (1024 * 1024)
    print(f"Base ONNX model size: {base_size_mb:.2f} MB")
    
    # 2. Apply INT8 Dynamic Quantization
    print("Applying INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(output_dir)
    
    # Use standard dynamic quantization config (ideal for CPU edge runtimes)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    # Perform quantization and save in the same directory
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=dqconfig
    )
    
    # The default output name for optimum is 'model_quantized.onnx'
    quantized_path = os.path.join(output_dir, "model_quantized.onnx")
    if os.path.exists(quantized_path):
        quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"Quantized ONNX model size: {quant_size_mb:.2f} MB")
        
        # Clean up unquantized model to save workspace size (optional but keeps edge package clean)
        # We will keep it for benchmarking size differences, then delete or keep.
        # Let's keep both for benchmarking, so the user can show original vs quantized comparison!
        print("Export and quantization complete. Ready for benchmarking!")
    else:
        print("Error: Quantization failed to produce model_quantized.onnx.")
        sys.exit(1)

if __name__ == "__main__":
    export_and_quantize()
