"""Main bridge architecture."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from transformers import DynamicCache

from h2c_bridge.models.attention import H2CAttentionBlock


class H2CProjector(nn.Module):
    """Projects hidden states into KV cache.
    
    Core bridge module. Projects sharer hidden states into the 
    receiver's key-value cache.
    """
    
    def __init__(
        self,
        sharer_dim: int,
        receiver_head_dim: int,
        receiver_num_heads: int,
        num_receiver_layers: int,
        sharer_num_layers: int,
        proj_num_heads: int = 4,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.sharer_dim = sharer_dim
        self.receiver_head_dim = receiver_head_dim
        self.receiver_num_heads = receiver_num_heads
        self.flat_receiver_dim = receiver_head_dim * receiver_num_heads

        # Store layer counts for alignment logic
        self.sharer_num_layers = sharer_num_layers
        self.receiver_num_layers = num_receiver_layers

        # Top-Down Alignment: Bridge the top min(N, M) layers
        self.num_bridged_layers = min(self.sharer_num_layers, self.receiver_num_layers)
        print(f"--- [Bridge] Aligning Top {self.num_bridged_layers} layers "
              f"(Sharer: {self.sharer_num_layers}, Receiver: {self.receiver_num_layers})")

        # One block per bridged layer
        self.key_modifiers = nn.ModuleList([
            H2CAttentionBlock(
                sharer_dim=sharer_dim,
                receiver_dim=self.flat_receiver_dim,
                num_attention_heads=proj_num_heads,
                dropout=dropout,
                dtype=dtype
            ) for _ in range(self.num_bridged_layers)
        ])

        self.value_modifiers = nn.ModuleList([
            H2CAttentionBlock(
                sharer_dim=sharer_dim,
                receiver_dim=self.flat_receiver_dim,
                num_attention_heads=proj_num_heads,
                dropout=dropout,
                dtype=dtype
            ) for _ in range(self.num_bridged_layers)
        ])

    def forward(self,
                source_hidden: Tensor,
                target_kv: Tuple[Tensor, Tensor],
                layer_idx: int) -> Tuple[Tensor, Tensor]:
        """Layer-specific projection.
        
        Args:
            source_hidden: Sharer hidden states
            target_kv: Receiver cache tuple (key, value)
            layer_idx: Bridge block index
            
        Returns:
            Modified (key, value) tuple
        """
        if layer_idx >= len(self.key_modifiers):
            raise ValueError(f"Layer index {layer_idx} out of bounds for Projector "
                           f"initialized with {len(self.key_modifiers)} layers")

        target_k, target_v = target_kv
        B, H, N, D = target_k.shape

        # Clone immediately to avoid any reference to original cache
        target_k = target_k.clone()
        target_v = target_v.clone()

        # Flatten
        target_k_flat = target_k.transpose(1, 2).reshape(B, N, H*D)
        target_v_flat = target_v.transpose(1, 2).reshape(B, N, H*D)

        # Select the specific blocks for this layer
        key_block = self.key_modifiers[layer_idx]
        value_block = self.value_modifiers[layer_idx]

        # Apply Cross Attention
        modified_k_flat = key_block(target_k_flat, source_hidden)
        modified_v_flat = value_block(target_v_flat, source_hidden)

        # Unflatten
        modified_k = modified_k_flat.view(B, N, H, D).transpose(1, 2)
        modified_v = modified_v_flat.view(B, N, H, D).transpose(1, 2)

        return modified_k, modified_v

    def cache_project(self, source_hidden_states: Tuple[Tensor], 
                     target_kv_cache: DynamicCache) -> DynamicCache:
        """Projects sharer states to receiver cache.
        
        Args:
            source_hidden_states: Tuple of sharer hidden states
            target_kv_cache: Receiver DynamicCache
            
        Returns:
            Modified DynamicCache
        """
        if not isinstance(target_kv_cache, DynamicCache):
            raise ValueError(f"Expected DynamicCache, got {type(target_kv_cache)}")

        # Shape Assertions
        assert isinstance(source_hidden_states, (tuple, list)), \
            f"Expected tuple/list of hidden states, got {type(source_hidden_states)}"
        assert len(source_hidden_states) >= self.sharer_num_layers, \
            f"Expected at least {self.sharer_num_layers} sharer layers, got {len(source_hidden_states)}"

        projected_cache = DynamicCache()

        # Iterate through Receiver layers (0 to N_R - 1)
        for r_layer_idx in range(len(target_kv_cache)):
            target_kv = target_kv_cache[r_layer_idx]

            # Top-Down Alignment Logic
            # We want to bridge the TOP 'num_bridged_layers'
            # The 'bridge index' 0 corresponds to the TOPMOST bridged layer

            # Calculate how far from the top we are
            layers_from_top = (self.receiver_num_layers - 1) - r_layer_idx

            if layers_from_top < self.num_bridged_layers:
                # This layer IS bridged

                # Determine which Sharer layer to use
                s_layer_idx = (self.sharer_num_layers - 1) - layers_from_top

                # Determine which Bridge Block to use (0 is top, N-1 is bottom of bridged stack)
                bridge_idx = layers_from_top

                source_hidden = source_hidden_states[s_layer_idx]

                # Pass bridge_idx to forward to pick the correct weights
                proj_key, proj_value = self.forward(source_hidden, target_kv, bridge_idx)

                # MUST clone to break reference chains
                projected_cache.update(proj_key.clone().contiguous(), 
                                     proj_value.clone().contiguous(), r_layer_idx)

            else:
                # This layer is NOT bridged (too deep / bottom layers)
                # Just copy the original cache
                k, v = target_kv
                projected_cache.update(k.clone().contiguous(), 
                                     v.clone().contiguous(), r_layer_idx)

        return projected_cache

    def get_gate_stats(self) -> dict:
        """Returns gate statistics.
        
        Returns:
            Dict with average and per-layer gate values
        """
        key_gates = [torch.sigmoid(block.gate).item() for block in self.key_modifiers]
        value_gates = [torch.sigmoid(block.gate).item() for block in self.value_modifiers]

        return {
            "key_gates": key_gates,
            "value_gates": value_gates,
            "key_avg": sum(key_gates) / len(key_gates),
            "value_avg": sum(value_gates) / len(value_gates),
        }
