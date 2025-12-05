"""Cross-attention block."""

import torch
import torch.nn as nn
from torch import Tensor


class H2CAttentionBlock(nn.Module):
    """Performs cross-attention from Receiver KV to Sharer hidden states.
    
    Full transformer-style block: CrossAttention + FFN with single output gate.
    """
    
    def __init__(
        self,
        sharer_dim: int,          # Dim of sharer hidden states
        receiver_dim: int,        # Dim of receiver hidden states
        num_attention_heads: int, # Number of heads for projection attention
        dropout: float = 0.1,
        ffn_expansion: int = 4,   # FFN hidden dim = receiver_dim * expansion
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Pre-norms for stability
        self.norm_receiver = nn.LayerNorm(receiver_dim, dtype=dtype)
        self.norm_sharer = nn.LayerNorm(sharer_dim, dtype=dtype)

        # Cross attention
        self.attn = nn.MultiheadAttention(
            embed_dim=receiver_dim,
            num_heads=num_attention_heads,
            kdim=sharer_dim,
            vdim=sharer_dim,
            dropout=dropout,
            batch_first=True,
            dtype=dtype,
        )

        # FFN: 2-layer MLP with GELU activation
        ffn_hidden = receiver_dim * ffn_expansion
        self.norm_ffn = nn.LayerNorm(receiver_dim, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(receiver_dim, ffn_hidden, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, receiver_dim, dtype=dtype),
            nn.Dropout(dropout),
        )

        # Single gate controls overall block contribution
        self.gate = nn.Parameter(torch.ones(1, dtype=dtype) * 1.0 + 0.01 * torch.randn(1, dtype=dtype))
        self.dropout = nn.Dropout(dropout)

    def forward(self, receiver_state: Tensor, sharer_hidden: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            receiver_state: (B, Seq, receiver_num_heads * receiver_head_dim)
            sharer_hidden: (B, Seq, sharer_dim)
            
        Returns:
            Modified receiver state
        """
        # 1. Cross-Attention
        q = self.norm_receiver(receiver_state)
        k = v = self.norm_sharer(sharer_hidden)
        attn_output, _ = self.attn(query=q, key=k, value=v, need_weights=False)
        attn_output = self.dropout(attn_output)
        
        # 2. FFN on attention output
        ffn_output = self.ffn(self.norm_ffn(attn_output))
        
        # 3. Single gated residual
        gate = torch.sigmoid(self.gate)
        return receiver_state + gate * ffn_output
