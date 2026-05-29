# src/models/onnx_transformer.py

import os
import json
import numpy as np
from typing import Tuple, Dict, List, Any
from src.models.base import BaseCommandClassifier
from src.config import ALL_COMMANDS, COMMAND_TEMPLATES, COSINE_SIMILARITY_THRESHOLD

class ONNXTransformerClassifier(BaseCommandClassifier):
    """
    Quantized ONNX Sentence Transformer command classifier.
    Computes sentence embeddings via ONNX Runtime and matches inputs
    to predefined command templates using Cosine Similarity.
    Requires no neural network training, making it highly extensible.
    """
    
    def __init__(
        self, 
        model_dir: str, 
        similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD,
        include_extension: bool = True
    ) -> None:
        self.model_dir: str = model_dir
        self.similarity_threshold: float = similarity_threshold
        self.include_extension: bool = include_extension
        self.session = None
        self.tokenizer = None
        self.template_embeddings: Dict[int, np.ndarray] = {}  # cmd_id -> matrix of template embeddings
        self.commands: Dict[int, str] = {}
        
    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Applies mean pooling to token embeddings to get sentence embedding."""
        token_embeddings = model_output  # First output from the transformer model (batch_size, seq_len, hidden_dim)
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        mean_pooled = sum_embeddings / sum_mask
        
        # L2 normalize embeddings
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return mean_pooled / norms

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Tokenizes inputs and runs ONNX inference to generate sentence embeddings."""
        if not self.session or not self.tokenizer:
            raise ValueError("Model is not loaded. Call load() first.")
            
        # Encode texts using the Rust-backed Tokenizer
        encoded = [self.tokenizer.encode(t) for t in texts]
        
        # Padding to max length in batch
        max_len = max(len(enc.ids) for enc in encoded)
        
        input_ids = []
        attention_mask = []
        
        for enc in encoded:
            ids = enc.ids + [0] * (max_len - len(enc.ids))
            mask = enc.attention_mask + [0] * (max_len - len(enc.attention_mask))
            input_ids.append(ids)
            attention_mask.append(mask)
            
        input_ids_np = np.array(input_ids, dtype=np.int64)
        attention_mask_np = np.array(attention_mask, dtype=np.int64)
        
        # ONNX input dictionary
        # MiniLM usually accepts input_ids, attention_mask, token_type_ids
        inputs = {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np
        }
        
        # Check if the model expects token_type_ids
        model_inputs = [i.name for i in self.session.get_inputs()]
        if "token_type_ids" in model_inputs:
            inputs["token_type_ids"] = np.zeros_like(input_ids_np, dtype=np.int64)
            
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Apply mean pooling (outputs[0] contains last_hidden_state)
        embeddings = self._mean_pooling(outputs[0], attention_mask_np)
        return embeddings

    def train(self, train_data: List[Dict[str, str]]) -> None:
        """
        Since this is a Few-Shot / Zero-Shot similarity model, 'training'
        simply involves caching the embeddings of the target command templates.
        """
        # Populate the command mapping based on config
        from src.config import CORE_COMMANDS, ALL_COMMANDS
        self.commands = ALL_COMMANDS if self.include_extension else CORE_COMMANDS
        
        # Pre-embed all templates
        print("Pre-embedding command templates...")
        for cmd_id, cmd_name in self.commands.items():
            templates = COMMAND_TEMPLATES[cmd_id]
            # Embed all templates for this command
            embeddings = self._embed(templates)
            self.template_embeddings[cmd_id] = embeddings
        print("Command templates embedded successfully.")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Embeds the input text and computes maximum cosine similarity
        against pre-cached templates.
        """
        if not self.template_embeddings:
            raise ValueError("Model has not been trained or template embeddings not cached.")
            
        # 1. Embed input text
        input_emb = self._embed([text])  # shape: (1, embedding_dim)
        
        best_cmd_id = -1
        best_score = -1.0
        
        # 2. Compute similarity to cached templates
        # Since embeddings are L2 normalized, cosine similarity is just the dot product
        for cmd_id, templates_emb in self.template_embeddings.items():
            # templates_emb shape: (num_templates_for_cmd, embedding_dim)
            # input_emb shape: (1, embedding_dim)
            similarities = np.dot(templates_emb, input_emb.T).squeeze(axis=1)
            max_sim = float(np.max(similarities))
            
            if max_sim > best_score:
                best_score = max_sim
                best_cmd_id = cmd_id
                
        # 3. Handle OOS rejection
        if best_score < self.similarity_threshold:
            return "None", best_score
            
        pred_label = self.commands.get(best_cmd_id, "None")
        return pred_label, best_score

    def save(self, path: str) -> None:
        """
        Saves the pre-computed template embeddings and metadata.
        This is saved to a JSON file alongside the ONNX model files.
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {
            str(cmd_id): emb.tolist() for cmd_id, emb in self.template_embeddings.items()
        }
        
        meta = {
            "include_extension": self.include_extension,
            "similarity_threshold": self.similarity_threshold,
            "embeddings": serializable_embeddings,
            "commands": {str(k): v for k, v in self.commands.items()}
        }
        
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Loads ONNX session, tokenizer, and cached template embeddings."""
        # 1. Load ONNX Runtime session
        import onnxruntime as ort
        
        model_path = os.path.join(self.model_dir, "model_quantized.onnx")
        if not os.path.exists(model_path):
            # Fallback to standard ONNX model if quantized isn't available yet
            model_path = os.path.join(self.model_dir, "model.onnx")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found in {self.model_dir}")
            
        # Configure session options for fast CPU edge execution
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        
        # 2. Load tokenizer (using tokenizers library for speed and zero torch dependency)
        from tokenizers import Tokenizer
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_dir}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 3. Load pre-cached template embeddings if metadata file exists
        if os.path.exists(path):
            with open(path, "r") as f:
                meta = json.load(f)
            self.include_extension = meta.get("include_extension", self.include_extension)
            self.similarity_threshold = meta.get("similarity_threshold", self.similarity_threshold)
            self.commands = {int(k): v for k, v in meta["commands"].items()}
            self.template_embeddings = {
                int(cmd_id): np.array(emb_list) for cmd_id, emb_list in meta["embeddings"].items()
            }
        else:
            # If no cached file, pre-embed templates on-the-fly (dynamic initialization)
            self.train([])
