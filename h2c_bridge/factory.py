"""Model loading and bridge initialization."""

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from h2c_bridge.models.projector import H2CProjector


class H2CModelFactory:
    """Loads models and initializes the H2C Bridge.
    
    Loads tokenizers, quantized LLMs, and initializes the bridge projector.
    """
    
    def __init__(self, sharer_id, receiver_id, device="cuda", dtype=torch.bfloat16):
        """Initialize the factory.
        
        Args:
            sharer_id: HuggingFace model ID for the sharer model
            receiver_id: HuggingFace model ID for the receiver model
            device: Device to use (default: "cuda")
            dtype: Data type for the bridge (default: torch.bfloat16)
        """
        self.sharer_id = sharer_id
        self.receiver_id = receiver_id
        self.device = device
        self.dtype = dtype

        # Placeholders
        self.sharer = None
        self.receiver = None
        self.tok_sharer = None
        self.tok_receiver = None
        self.bridge = None

    def load_tokenizers(self):
        """Loads tokenizers for sharer and receiver.
        
        Returns:
            (tok_sharer, tok_receiver)
        """
        print("--- [ModelFactory] Loading Tokenizers...")
        self.tok_sharer = AutoTokenizer.from_pretrained(self.sharer_id)
        self.tok_receiver = AutoTokenizer.from_pretrained(self.receiver_id)

        # Use left-padding for decoder-only models (required for correct generation)
        self.tok_sharer.padding_side = 'left'
        self.tok_receiver.padding_side = 'left'

        # Ensure pad token is set (some models don't have one by default)
        if self.tok_sharer.pad_token is None:
            self.tok_sharer.pad_token = self.tok_sharer.eos_token
        if self.tok_receiver.pad_token is None:
            self.tok_receiver.pad_token = self.tok_receiver.eos_token

        return self.tok_sharer, self.tok_receiver

    def load_llms(self):
        """Loads frozen LLMs with 4-bit quantization.
        
        Returns:
            (sharer, receiver)
        """
        print("--- [ModelFactory] Loading LLMs (Frozen + Quantized)...")

        # 4-bit quantization config for frozen models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # nested quantization
        )

        self.sharer = AutoModelForCausalLM.from_pretrained(
            self.sharer_id,
            quantization_config=bnb_config,
            device_map="auto",  # required for quantized models
        )

        self.receiver = AutoModelForCausalLM.from_pretrained(
            self.receiver_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Attach tokenizers for convenience
        self.sharer.tokenizer = self.tok_sharer
        self.receiver.tokenizer = self.tok_receiver

        return self.sharer, self.receiver

    def create_bridge(self):
        """Initializes the H2C Projector.
        
        Returns:
            H2CProjector instance
        """
        print("--- [ModelFactory] Initializing Bridge...")

        s_dims = self._get_model_dims(self.sharer_id)
        r_dims = self._get_model_dims(self.receiver_id)

        self.bridge = H2CProjector(
            sharer_dim=s_dims["hidden_dim"],
            receiver_head_dim=r_dims["head_dim"],
            receiver_num_heads=r_dims["kv_heads"],  # Uses KV heads for GQA compatibility
            num_receiver_layers=r_dims["num_layers"],
            sharer_num_layers=s_dims["num_layers"],
            proj_num_heads=16,  # Increased from 4 for more expressive attention (56 dim/head)
            dtype=self.dtype
        ).to(self.device)
        
        # Print parameter count
        total = sum(p.numel() for p in self.bridge.parameters())
        trainable = sum(p.numel() for p in self.bridge.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Size (MB): {total * 4 / 1024 / 1024:.1f} MB (float32)")

        return self.bridge

    def _get_model_dims(self, model_id):
        """Helper to extract dimensions safely.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Dictionary of model dimensions
        """
        config = AutoConfig.from_pretrained(model_id)
        hidden_dim = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
        kv_heads = getattr(config, "num_key_value_heads", num_heads)
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and hidden_dim and num_heads:
            head_dim = hidden_dim // num_heads

        return {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim
        }
