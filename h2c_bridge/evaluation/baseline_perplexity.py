"""Baseline perplexity evaluation."""

import torch
from tqdm.auto import tqdm


class BaselinePerplexityEvaluator:
    """Evaluates perplexity for baseline models.
    
    Computes validation loss (and derived perplexity) for:
    - Receiver-only (no knowledge transfer)
    - Sharer-only (full sharer model)
    - Text-to-text (sharer generates hint, receiver uses it)
    """
    
    def __init__(self, sharer, receiver, tok_sharer, tok_receiver, device="cuda"):
        """Initialize the evaluator.
        
        Args:
            sharer: Sharer model
            receiver: Receiver model
            tok_sharer: Sharer tokenizer
            tok_receiver: Receiver tokenizer
            device: Device to use (default: "cuda")
        """
        self.sharer = sharer
        self.receiver = receiver
        self.tok_sharer = tok_sharer
        self.tok_receiver = tok_receiver
        self.device = device
    
    @torch.no_grad()
    def evaluate_receiver_only_loss(self, dataloader):
        """Evaluate receiver-only loss.
        
        Args:
            dataloader: Validation dataloader (OpenHermes val set)
            
        Returns:
            dict: {"loss": float, "perplexity": float}
        """
        self.receiver.eval()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Receiver-Only Perplexity", leave=False)
        
        for batch in progress_bar:
            # Receiver processes full prompt + target
            rec_prompt_ids = batch['receiver_prompt_ids'].to(self.device)
            rec_prompt_mask = batch['receiver_prompt_mask'].to(self.device)
            rec_kickstart_ids = batch['receiver_kickstart_ids'].to(self.device)
            rec_target_ids = batch['receiver_target_ids'].to(self.device)
            rec_target_mask = batch['receiver_target_mask'].to(self.device)
            
            # Combine prompt + kickstart + target
            combined_input = torch.cat([rec_prompt_ids, rec_kickstart_ids, rec_target_ids], dim=1)
            combined_mask = torch.cat([rec_prompt_mask, torch.ones_like(rec_kickstart_ids), rec_target_mask], dim=1)
            
            # Labels: ignore prompt and kickstart, only predict target
            labels = torch.cat([
                torch.full_like(rec_prompt_ids, -100),
                torch.full_like(rec_kickstart_ids, -100),
                rec_target_ids.clone()
            ], dim=1)
            labels[labels == self.tok_receiver.pad_token_id] = -100  # Mask padding
            
            outputs = self.receiver(
                input_ids=combined_input,
                attention_mask=combined_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            del combined_input, combined_mask, labels, outputs
        
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    @torch.no_grad()
    def evaluate_sharer_only_loss(self, dataloader):
        """Evaluate sharer-only loss on TARGET generation.
        
        Measures how well the sharer predicts the target response given the prompt.
        This is a fair comparison to receiver-only.
        
        Args:
            dataloader: Validation dataloader (OpenHermes val set)
            
        Returns:
            dict: {"loss": float, "perplexity": float}
        """
        self.sharer.eval()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Sharer-Only Perplexity", leave=False)
        
        for batch in progress_bar:
            # Get raw prompts and targets from batch
            raw_prompts = batch.get('raw_context', None)
            if raw_prompts is None:
                # Fall back to decoding receiver prompt
                raw_prompts = self.tok_receiver.batch_decode(
                    batch['receiver_prompt_ids'], skip_special_tokens=True
                )
            
            # Get raw targets
            raw_targets = self.tok_receiver.batch_decode(
                batch['receiver_target_ids'], skip_special_tokens=True
            )
            
            # Format for sharer: prompt -> target
            sharer_inputs_formatted = []
            for prompt, target in zip(raw_prompts, raw_targets):
                sharer_inputs_formatted.append([
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": target.strip()}
                ])
            
            # Tokenize full conversation (prompt + target)
            encoded = self.tok_sharer.apply_chat_template(
                sharer_inputs_formatted,
                tokenize=True,
                add_generation_prompt=False,  # Include assistant response
                padding=True,
                return_tensors="pt",
                return_dict=True
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Get prompt-only length to know where targets start
            prompt_only_formatted = []
            for prompt in raw_prompts:
                prompt_only_formatted.append([{"role": "user", "content": prompt.strip()}])
            
            prompt_encoded = self.tok_sharer.apply_chat_template(
                prompt_only_formatted,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True
            )
            prompt_len = prompt_encoded["input_ids"].shape[1]
            
            # Create labels: -100 for prompt, actual tokens for target
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100  # Ignore prompt tokens
            labels[attention_mask == 0] = -100  # Ignore padding
            
            outputs = self.sharer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            del input_ids, attention_mask, labels, outputs
        
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    @torch.no_grad()
    def evaluate_text_to_text_loss(self, dataloader, max_hint_tokens=48):
        """Evaluate text-to-text loss.
        
        Sharer generates hint, receiver predicts target given hint.
        This measures how well the receiver performs with textual hints.
        
        Args:
            dataloader: Validation dataloader (OpenHermes val set)
            max_hint_tokens: Max tokens for sharer hint generation
            
        Returns:
            dict: {"loss": float, "perplexity": float}
        """
        self.sharer.eval()
        self.receiver.eval()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Text-to-Text Perplexity", leave=False)
        
        for batch in progress_bar:
            # 1. Get raw prompts and targets
            raw_prompts = batch.get('raw_prompt', None)
            if raw_prompts is None:
                # Fall back to decoding receiver prompt if raw_prompt not available
                raw_prompts = self.tok_receiver.batch_decode(
                    batch['receiver_prompt_ids'], skip_special_tokens=True
                )
            
            rec_target_ids = batch['receiver_target_ids'].to(self.device)
            rec_target_mask = batch['receiver_target_mask'].to(self.device)
            
            # 2. Sharer generates hints
            sharer_inputs_formatted = []
            for prompt in raw_prompts:
                s_prompt = f"Give a concise hint to help answer: {prompt.strip()} Do NOT give the answer directly."
                sharer_inputs_formatted.append([{"role": "user", "content": s_prompt}])
            
            s_encoded = self.tok_sharer.apply_chat_template(
                sharer_inputs_formatted,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True
            )
            s_inputs = s_encoded["input_ids"].to(self.device)
            s_attn_mask = s_encoded["attention_mask"].to(self.device)
            
            s_out = self.sharer.generate(
                s_inputs,
                attention_mask=s_attn_mask,
                max_new_tokens=max_hint_tokens,
                pad_token_id=self.tok_sharer.pad_token_id,
                eos_token_id=self.tok_sharer.eos_token_id,
                do_sample=False
            )
            
            # Decode hints (skip special tokens to avoid leaking into receiver)
            s_hints = self.tok_sharer.batch_decode(
                s_out[:, s_inputs.shape[1]:],
                skip_special_tokens=True
            )
            
            # 3. Receiver processes prompt + hint + target
            receiver_inputs_formatted = []
            for prompt, hint in zip(raw_prompts, s_hints):
                clean_hint = hint.replace("\n", " ").strip()
                final_content = f"{prompt.strip()}\n[Hint: {clean_hint}]"
                receiver_inputs_formatted.append([{"role": "user", "content": final_content}])
            
            r_encoded = self.tok_receiver.apply_chat_template(
                receiver_inputs_formatted,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True
            )
            r_prompt_ids = r_encoded["input_ids"].to(self.device)
            r_prompt_mask = r_encoded["attention_mask"].to(self.device)
            
            # 4. Compute loss on target
            # Combine prompt (with hint) + target
            combined_input = torch.cat([r_prompt_ids, rec_target_ids], dim=1)
            combined_mask = torch.cat([r_prompt_mask, rec_target_mask], dim=1)
            
            # Labels: ignore prompt, only predict target
            labels = torch.cat([
                torch.full_like(r_prompt_ids, -100),
                rec_target_ids.clone()
            ], dim=1)
            labels[labels == self.tok_receiver.pad_token_id] = -100  # Mask padding
            
            outputs = self.receiver(
                input_ids=combined_input,
                attention_mask=combined_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            del s_inputs, s_attn_mask, s_out, r_prompt_ids, r_prompt_mask
            del combined_input, combined_mask, labels, outputs
        
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    def evaluate_all_baselines(self, dataloader):
        """Evaluate all baseline perplexities.
        
        Args:
            dataloader: Validation dataloader (OpenHermes val set)
            
        Returns:
            dict: Results for each baseline
        """
        results = {}
        
        print("\n--- Evaluating Baseline Perplexities (OpenHermes Val Set) ---")
        
        # Receiver-only
        rec_results = self.evaluate_receiver_only_loss(dataloader)
        results['receiver_only'] = rec_results
        print(f"Receiver-Only: Loss={rec_results['loss']:.4f}, PPL={rec_results['perplexity']:.2f}")
        
        # Sharer-only
        sharer_results = self.evaluate_sharer_only_loss(dataloader)
        results['sharer_only'] = sharer_results
        print(f"Sharer-Only: Loss={sharer_results['loss']:.4f}, PPL={sharer_results['perplexity']:.2f}")
        
        # Text-to-text (sharer hint → receiver)
        t2t_results = self.evaluate_text_to_text_loss(dataloader)
        results['text_to_text'] = t2t_results
        print(f"Text-to-Text: Loss={t2t_results['loss']:.4f}, PPL={t2t_results['perplexity']:.2f}")
        
        return results
