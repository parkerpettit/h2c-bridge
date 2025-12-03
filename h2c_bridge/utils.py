"""Project utilities."""

import gc
import os
import random

import numpy as np
import torch
from transformers import AutoConfig


def set_seed(seed=42):
    """Sets random seeds.
    
    Args:
        seed: Random seed (default: 42)
    """
    # 1. Python and NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. PyTorch (CPU + GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # 3. Force Deterministic Algorithms
    # This slows down training slightly but ensures convolution/matmul operations
    # are reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def get_model_dims(model_id):
    """Extracts model dimensions from AutoConfig.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        dict: Model dimensions (hidden_dim, num_layers, num_heads, kv_heads, head_dim)
    """
    config = AutoConfig.from_pretrained(model_id)
    hidden_dim = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    kv_heads = getattr(config, "num_key_value_heads", num_heads)  # Fallback to num_heads if no GQA
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))

    # Calculate head_dim if not explicitly stated
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden_dim and num_heads:
        head_dim = hidden_dim // num_heads

    print(f"[{model_id}] Dims -> Hidden: {hidden_dim}, Layers: {num_layers}, "
          f"KV Heads: {kv_heads}, Head Dim: {head_dim}")

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "config": config
    }


def clear_gpu():
    """Clears GPU memory."""
    # All potential globals that hold GPU tensors
    names_to_delete = [
        "engine", "factory", "dm", "trainer", "evaluator",
        "sharer", "receiver", "bridge",
        "tok_sharer", "tok_receiver",
        "train_loader", "val_loader", "mmlu_loader"
    ]

    g = globals()
    for name in names_to_delete:
        if name in g:
            try:
                obj = g[name]
                # If it has a .to() method, try moving to CPU first
                if hasattr(obj, 'to'):
                    try:
                        obj.to('cpu')
                    except:
                        pass
                del g[name]
            except Exception as e:
                print(f"Warning: Could not delete {name}: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Cleared GPU cache.")
