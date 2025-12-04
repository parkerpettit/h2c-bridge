"""Base class for training and eval."""

import gc

import torch
from transformers import DynamicCache


class H2CBase:
    """Shared logic for bridge operations.
    
    Implements common forward pass: Sharer -> Receiver Cache -> Bridge.
    Inherited by Trainer and Evaluator.
    """
    
    def __init__(self, sharer, receiver, bridge, config, device="cuda"):
        """Initialize the base class.
        
        Args:
            sharer: Sharer model
            receiver: Receiver model
            bridge: H2C Bridge (projector)
            config: Configuration dictionary
            device: Device to use (default: "cuda")
        """
        self.sharer = sharer
        self.receiver = receiver
        self.bridge = bridge
        self.device = device
        self.config = config

    def get_bridged_cache(self, sharer_ids, sharer_mask, rec_prompt_ids, rec_prompt_mask=None):
        """Computes bridged cache.
        
        Steps:
            1. Extract Sharer Hidden States (Frozen)
            2. Extract Receiver Initial Cache (Frozen)
            3. Project/Modify Cache (Trainable)
        
        Args:
            sharer_ids: Sharer input IDs
            sharer_mask: Sharer attention mask
            rec_prompt_ids: Receiver prompt IDs (no last token)
            rec_prompt_mask: Receiver attention mask
            
        Returns:
            Modified DynamicCache
        """
        # A. Run Frozen Models (No Grad)
        receiver_cache = DynamicCache()
        with torch.no_grad():
            # 1. Sharer
            sharer_out = self.sharer(
                input_ids=sharer_ids,
                attention_mask=sharer_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # Clone ALL hidden states to detach them
            # This allows us to delete sharer_out and free VRAM
            # IMPORTANT: detach() is crucial to ensure no graph history is kept
            sharer_hidden = tuple(h.detach().clone() for h in sharer_out.hidden_states)
            
            # CRITICAL MEMORY FIX: Explicitly delete the huge KV cache
            if hasattr(sharer_out, 'past_key_values'):
                del sharer_out.past_key_values
                
            del sharer_out

            # 2. Receiver (Pre-fill)
            # This populates receiver_cache in-place.
            # We capture the output only to delete it immediately.
            rec_out = self.receiver(
                input_ids=rec_prompt_ids,
                attention_mask=rec_prompt_mask,  # Use mask if provided
                past_key_values=receiver_cache,
                use_cache=True
            )
            del rec_out

        # Force Python GC to reclaim the deleted tensors before allocating more VRAM
        gc.collect()

        # B. Run Bridge (Grads allowed if context permits)
        # This is outside no_grad() so the Trainer can track gradients here.
        modified_cache = self.bridge.cache_project(sharer_hidden, receiver_cache)

        # Cleanup intermediate tensors (modified_cache has its own copies)
        del sharer_hidden

        return modified_cache
