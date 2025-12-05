"""OpenHermes evaluation."""

import gc

import torch
from tqdm.auto import tqdm

from h2c_bridge.evaluation.base import H2CBase


class H2CEvaluator(H2CBase):
    """Evaluates on OpenHermes validation set.
    
    Calculates validation loss and generates demos.
    """
    
    def __init__(self, sharer, receiver, bridge, tok_receiver, config, device="cuda"):
        """Initialize the evaluator.
        
        Args:
            sharer: Sharer model
            receiver: Receiver model
            bridge: H2C Bridge
            tok_receiver: Receiver tokenizer
            config: Configuration dictionary
            device: Device to use (default: "cuda")
        """
        super().__init__(sharer, receiver, bridge, config, device)
        self.tok_receiver = tok_receiver

    @torch.no_grad()
    def evaluate_loss(self, dataloader):
        """Evaluate validation loss on a dataloader.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            float: Average validation loss
        """
        self.bridge.eval()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Validation Loop", leave=False)

        for batch in progress_bar:
            try:
                # 1. Unpack
                sharer_ids = batch['sharer_input_ids'].to(self.device)
                sharer_mask = batch['sharer_mask'].to(self.device)
                rec_prompt_ids = batch['receiver_prompt_ids'].to(self.device)
                rec_prompt_mask = batch['receiver_prompt_mask'].to(self.device)
                rec_kickstart_ids = batch['receiver_kickstart_ids'].to(self.device)
                rec_target_ids = batch['receiver_target_ids'].to(self.device)
                rec_target_mask = batch['receiver_target_mask'].to(self.device)

                # Use autocast to match training dtype
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    modified_cache = self.get_bridged_cache(sharer_ids, sharer_mask, rec_prompt_ids, rec_prompt_mask)
                    del sharer_ids, sharer_mask, rec_prompt_ids, rec_prompt_mask

                    combined_input = torch.cat([rec_kickstart_ids, rec_target_ids], dim=1)
                    combined_mask = torch.cat([torch.ones_like(rec_kickstart_ids), rec_target_mask], dim=1)

                    # Shape Assertions
                    assert modified_cache is not None
                    assert len(modified_cache) > 0

                    # Build full attention mask including cache positions
                    cache_len = modified_cache[0][0].shape[2]
                    batch_size = combined_input.shape[0]

                    # Assertions
                    assert combined_input.shape[0] == batch_size
                    assert combined_mask.shape[0] == batch_size

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

                    del rec_kickstart_ids, rec_target_ids, rec_target_mask

                    outputs = self.receiver(
                        input_ids=combined_input,
                        past_key_values=modified_cache,
                        attention_mask=full_attention_mask,
                        labels=labels
                    )
                    del combined_input, combined_mask, full_attention_mask, labels, modified_cache

                total_loss += outputs.loss.item()
                del outputs
                num_batches += 1

            except torch.cuda.OutOfMemoryError:
                print(f"\n[WARNING] OOM during validation, skipping batch...")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        # Cleanup after eval loop
        torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.bridge.train()
        return avg_loss

    @torch.no_grad()
    def generate_demo(self, prompt_text, max_new_tokens=500):
        """Generates demo comparison.
        
        Args:
            prompt_text: Input prompt
            max_new_tokens: Max tokens to generate
        """
        self.bridge.eval()
        print(f"\nPROMPT: {prompt_text}\n" + "-"*20)

        # --- A. Manual Tokenization ---
        # Sharer
        s_raw = self.sharer.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True, return_dict=False
        )
        s_ids = torch.tensor([s_raw[:-1]]).to(self.device)  # Strip Last
        s_mask = torch.ones_like(s_ids).to(self.device)     # Mask is all 1s

        # Receiver
        r_raw = self.tok_receiver.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True, return_dict=False
        )

        r_input_full = torch.tensor([r_raw]).to(self.device)            # For Vanilla
        r_input_stripped = torch.tensor([r_raw[:-1]]).to(self.device)   # For Bridge (Context)
        r_kickstart = torch.tensor([r_raw[-1:]]).to(self.device)        # For Bridge (Input)

        # --- B. Vanilla Control Group ---
        vanilla_out = self.receiver.generate(
            r_input_full,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok_receiver.pad_token_id
        )
        v_text = self.tok_receiver.decode(vanilla_out[0], skip_special_tokens=False)
        print(f"[Vanilla]:\n{v_text.replace(prompt_text, '').strip()}")

        # --- C. Bridged Generation ---
        # 1. Use Helper Function (with autocast to match training dtype)
        r_mask = torch.ones_like(r_input_stripped).to(self.device)  # No padding in single sample
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            modified_cache = self.get_bridged_cache(s_ids, s_mask, r_input_stripped, r_mask)

            # 2. Generate
            bridged_out = self.receiver.generate(
                input_ids=r_input_full,  # Pass FULL input (not just kickstart)
                past_key_values=modified_cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok_receiver.pad_token_id
            )

        b_text = self.tok_receiver.decode(bridged_out[0], skip_special_tokens=False)
        print(f"\n[Bridged]:\n{b_text.strip()}")
        print("-" * 20 + "\n")

        # Cleanup
        del s_ids, s_mask, r_input_full, r_input_stripped, r_kickstart, r_mask
        del vanilla_out, modified_cache, bridged_out
        torch.cuda.empty_cache()

        self.bridge.train()
