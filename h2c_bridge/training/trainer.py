"""H2C Trainer."""

import torch
import wandb

from h2c_bridge.evaluation.base import H2CBase


class H2CTrainer(H2CBase):
    """Handles training steps.
    
    Manages forward pass, loss, and optimization. Sharer/Receiver are frozen.
    """
    
    def __init__(self, sharer, receiver, bridge, optimizer, config, device="cuda"):
        """Initialize the trainer.
        
        Args:
            sharer: Sharer model (will be frozen)
            receiver: Receiver model (will be frozen)
            bridge: H2C Bridge (trainable)
            optimizer: Optimizer for bridge parameters
            config: Configuration dictionary
            device: Device to use (default: "cuda")
        """
        super().__init__(sharer, receiver, bridge, config, device)
        self.optimizer = optimizer

        # Freeze Models
        self.sharer.eval()
        self.sharer.requires_grad_(False)
        self.receiver.eval()
        self.receiver.requires_grad_(False)

        # Bridge is Trainable
        self.bridge.train()
        self.bridge.requires_grad_(True)

    def train_step(self, batch):
        """Executes one training step.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            float: Loss value
        """
        # 1. Unpack Batch
        sharer_ids = batch['sharer_input_ids'].to(self.device)
        sharer_mask = batch['sharer_mask'].to(self.device)
        rec_prompt_ids = batch['receiver_prompt_ids'].to(self.device)
        rec_prompt_mask = batch['receiver_prompt_mask'].to(self.device)
        rec_kickstart_ids = batch['receiver_kickstart_ids'].to(self.device)
        rec_target_ids = batch['receiver_target_ids'].to(self.device)
        rec_target_mask = batch['receiver_target_mask'].to(self.device)

        # 2. Get Modified Cache (sharer/receiver inputs only needed here)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            modified_cache = self.get_bridged_cache(sharer_ids, sharer_mask, rec_prompt_ids, rec_prompt_mask)

        # Free sharer inputs immediately after cache creation
        del sharer_ids, sharer_mask, rec_prompt_ids, rec_prompt_mask

        # 3. Calculate Loss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            combined_input = torch.cat([rec_kickstart_ids, rec_target_ids], dim=1)
            # Build attention mask: kickstart is always attended, target uses its mask
            combined_mask = torch.cat([torch.ones_like(rec_kickstart_ids), rec_target_mask], dim=1)

            # Build full attention mask including cache positions
            cache_len = modified_cache[0][0].shape[2]
            batch_size = combined_input.shape[0]
            cache_mask = torch.ones(batch_size, cache_len, device=self.device, dtype=combined_mask.dtype)
            full_attention_mask = torch.cat([cache_mask, combined_mask], dim=1)

            # Create labels: -100 for kickstart (ignore), actual tokens for targets
            # Build target labels with padding masked first
            target_labels = rec_target_ids.clone()
            target_labels[rec_target_mask == 0] = -100  # Mask padding in targets

            # Concatenate kickstart (-100) with masked target labels
            labels = torch.cat([
                torch.full_like(rec_kickstart_ids, -100),  # Ignore kickstart in loss
                target_labels  # Target tokens with padding already masked
            ], dim=1)

            del rec_kickstart_ids, rec_target_ids, rec_target_mask  # No longer needed

            outputs = self.receiver(
                input_ids=combined_input,
                past_key_values=modified_cache,
                attention_mask=full_attention_mask,
                labels=labels,
            )
            del combined_input, combined_mask, labels  # No longer needed

            loss = outputs.loss

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Free computation graph after backward (gradients computed)
        # CRITICAL: Delete the grown cache from receiver output first
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            del outputs.past_key_values
        del modified_cache, outputs
        torch.cuda.empty_cache()

        # Collect gradient norms for diagnostics
        key_attn_grad = 0.0
        value_attn_grad = 0.0
        gate_grad = 0.0
        for block in self.bridge.key_modifiers:
            if block.attn.out_proj.weight.grad is not None:
                key_attn_grad += block.attn.out_proj.weight.grad.norm().item()
            if block.gate.grad is not None:
                gate_grad += block.gate.grad.norm().item()
        for block in self.bridge.value_modifiers:
            if block.attn.out_proj.weight.grad is not None:
                value_attn_grad += block.attn.out_proj.weight.grad.norm().item()
            if block.gate.grad is not None:
                gate_grad += block.gate.grad.norm().item()

        # Clip to 10.0 instead of 1.0 - with grad norms spiking to 3000+,
        # clipping to 1.0 makes effective updates ~1e-8, appearing as flat plots
        grad_norm = torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), 10.0)

        # Log metrics (need .item() before deleting)
        loss_value = loss.item()
        del loss

        wandb.log({
            "Training/Grad Norm": grad_norm.item(),
            "Training/Key Attn Grad Norm": key_attn_grad,
            "Training/Value Attn Grad Norm": value_attn_grad,
            "Training/Gate Grad Norm": gate_grad,
        })

        self.optimizer.step()

        return loss_value
