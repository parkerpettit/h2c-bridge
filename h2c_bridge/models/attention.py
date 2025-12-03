"""Cross-attention block."""

import torch
import torch.nn as nn
from torch import Tensor


class H2CAttentionBlock(nn.Module):
    """Performs cross-attention from Receiver KV to Sharer hidden states."""
    
    def __init__(
        self,
        sharer_dim: int,          # Dim of sharer hidden states
        receiver_dim: int,        # Dim of receiver hidden states
        num_attention_heads: int, # Number of heads for projection attention (not model heads)
        dropout: float = 0.1,
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

        # Zero-init output projection for identity start
        nn.init.normal_(self.attn.out_proj.weight, std=0.01)
        nn.init.zeros_(self.attn.out_proj.bias)

        # Initialize gate near 0
        init = 0.1 * torch.randn(1, dtype=dtype)
        self.gate = nn.Parameter(init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, receiver_state: Tensor, sharer_hidden: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            receiver_state: (B, Seq, receiver_num_heads * receiver_head_dim)
            sharer_hidden: (B, Seq, sharer_dim)
            
        Returns:
            Modified receiver state
        """
        # Apply norms
        q = self.norm_receiver(receiver_state)
        k = v = self.norm_sharer(sharer_hidden)

        # Cross attention: Query=Receiver, Key/Val = sharer
        attn_output, _ = self.attn(query=q, key=k, value=v, need_weights=False)

        # Dropout on attention output
        output = self.dropout(attn_output)

        # Simple learned gate in (0, 1)
        gate = torch.sigmoid(self.gate)

        return receiver_state + gate * output
