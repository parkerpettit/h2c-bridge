"""MMLU evaluation."""

import gc
import re
import time

import torch
import wandb
from tqdm.auto import tqdm

from h2c_bridge.evaluation.base import H2CBase


class H2CMMLUEvaluator(H2CBase):
    """Evaluates on MMLU benchmark.
    
    Evaluates bridge and baselines (receiver-only, sharer-only, text-to-text).
    """
    
    def __init__(self, sharer, receiver, bridge, tok_receiver, tok_sharer, config, device="cuda"):
        """Initialize the evaluator.
        
        Args:
            sharer: Sharer model
            receiver: Receiver model
            bridge: H2C Bridge
            tok_receiver: Receiver tokenizer
            tok_sharer: Sharer tokenizer
            config: Configuration dictionary
            device: Device to use (default: "cuda")
        """
        super().__init__(sharer, receiver, bridge, config, device)
        self.tok_receiver = tok_receiver
        self.tok_sharer = tok_sharer
        
        # Cache answer token IDs for logprob-based evaluation
        # Use " A", " B", etc. to match the expected format after generation prompt
        self.answer_tokens_receiver = ["A", "B", "C", "D"]
        self.answer_ids_receiver = [
            tok_receiver.encode(" " + a, add_special_tokens=False)[-1] 
            for a in self.answer_tokens_receiver
        ]
        self.answer_ids_sharer = [
            tok_sharer.encode(" " + a, add_special_tokens=False)[-1] 
            for a in self.answer_tokens_receiver
        ]

    @torch.no_grad()
    def evaluate_baselines(self, dataloader, max_new_tokens=256, debug_mode=False):
        """Runs static baselines.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            debug_mode: If True, stop after 5 samples for quick testing
            
        Returns:
            dict: Results for each mode
        """
        print("\n" + "="*30)
        print(">>> RUNNING BASELINES (Deterministic)")
        if debug_mode:
            print("[DEBUG MODE] Will stop after 25 samples")
        print("="*30 + "\n")

        results = {}
        modes = ["receiver_only", "sharer_only", "text_to_text"]

        for mode in modes:
            print(f"--- Running Baseline: {mode} ---")
            acc, err, lat = self._eval_loop(dataloader, mode=mode, max_new_tokens=max_new_tokens, debug_mode=debug_mode)
            results[mode] = {"acc": acc, "err": err, "latency_s": lat}

        return results

    @torch.no_grad()
    def evaluate_baselines_detailed(self, dataloader, max_new_tokens=256, include_text_to_text=True, debug_mode=False):
        """Runs baselines with detailed logging.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            include_text_to_text: Include text-to-text baseline
            debug_mode: If True, stop after 25 samples for quick testing
            
        Returns:
            dict: Results with details
        """
        print("\n" + "="*30)
        print(">>> RUNNING BASELINES (Detailed)")
        if debug_mode:
            print("[DEBUG MODE] Will stop after 25 samples")
        print("="*30 + "\n")

        results = {}
        modes = ["receiver_only", "sharer_only"]
        if include_text_to_text:
            modes.append("text_to_text")

        for mode in modes:
            print(f"--- Running Baseline (Detailed): {mode} ---")
            acc, err, lat, details = self._eval_loop(
                dataloader, mode=mode, max_new_tokens=max_new_tokens, collect_details=True, debug_mode=debug_mode
            )
            results[mode] = {
                "acc": acc,
                "err": err,
                "latency_s": lat,
                "details": details
            }

        return results

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader, max_new_tokens=256, debug_mode=False, step=None):
        """Evaluates bridge accuracy.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            debug_mode: If True, stop after 5 samples for quick testing
            step: Current training step (for WandB logging)
            
        Returns:
            (accuracy, error_rate, latency_s)
        """
        self.bridge.eval()
        return self._eval_loop(dataloader, mode="bridge", max_new_tokens=max_new_tokens, debug_mode=debug_mode, step=step)

    @torch.no_grad()
    def evaluate_accuracy_detailed(self, dataloader, max_new_tokens=256, debug_mode=False, step=None):
        """Evaluates bridge with detailed logging.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            debug_mode: If True, stop after 25 samples for quick testing
            step: Current training step (for WandB logging)
            
        Returns:
            (accuracy, error_rate, latency, detailed_results)
        """
        self.bridge.eval()
        return self._eval_loop(dataloader, mode="bridge", max_new_tokens=max_new_tokens, collect_details=True, debug_mode=debug_mode, step=step)

    def _eval_loop(self, dataloader, mode, max_new_tokens, collect_details=False, debug_mode=False, step=None):
        """Main evaluation loop.
        
        Args:
            dataloader: Dataloader
            mode: Evaluation mode ("bridge", "receiver_only", "sharer_only", "text_to_text")
            max_new_tokens: Maximum tokens to generate
            collect_details: If True, collect detailed results for each sample
            debug_mode: If True, stop after 25 samples for quick testing
            step: Current training step (for WandB logging)
            
        Returns:
            Tuple of (accuracy, error_rate, latency_s) or (accuracy, error_rate, latency_s, detailed_results) if collect_details=True
        """
        stats = {"correct": 0, "total": 0, "format_errors": 0, "total_time": 0.0, "total_tokens_generated": 0}
        detailed_results = [] if collect_details else None
        
        # WandB example logging
        wandb_log_examples = self.config.get("wandb_log_examples", 10)
        wandb_examples = [] if wandb_log_examples > 0 else None

        desc = f"Eval [{mode}]" + (" [DEBUG]" if debug_mode else "")
        progress_bar = tqdm(dataloader, desc=desc, leave=True, mininterval=1.0)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 1. Measure Generation Time (with CUDA sync for accurate timing)
                torch.cuda.synchronize()  # Ensure previous ops complete
                start_time = time.time()
                prompt_texts, gen_texts, labels, tokens_generated = self._generate_batch(batch, mode, max_new_tokens)
                torch.cuda.synchronize()  # Ensure all GPU ops complete before timing
                end_time = time.time()

                # Get subjects if available (for detailed analysis)
                subjects = batch.get('subjects', ['unknown'] * len(labels))

                # Update Stats
                stats["total_time"] += (end_time - start_time)
                stats["total_tokens_generated"] += tokens_generated

                # 2. Score & Log
                batch_details = self._score_batch(
                    prompt_texts, gen_texts, labels, stats, subjects, 
                    collect_details=collect_details,
                    wandb_examples=wandb_examples,
                    wandb_max=wandb_log_examples,
                    mode=mode
                )
                if collect_details and batch_details:
                    detailed_results.extend(batch_details)

                # 3. Debug Mode Early Exit
                if debug_mode and stats["total"] >= 25:
                    print(f"\n[DEBUG MODE] Stopping early after {stats['total']} samples")
                    break


            except torch.cuda.OutOfMemoryError:
                print(f"\n[WARNING] OOM during {mode} evaluation, skipping batch...")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        # Calculate Metrics
        total = stats["total"] if stats["total"] > 0 else 1
        accuracy = stats["correct"] / total
        error_rate = stats["format_errors"] / total
        avg_latency_s = (stats["total_time"] / total)
        avg_tokens = stats["total_tokens_generated"] / total
        # Calculate per-token latency (only if we generated tokens)
        total_tokens = stats["total_tokens_generated"]
        latency_per_token = stats["total_time"] / total_tokens if total_tokens > 0 else 0

        print(f"[{mode}] Acc: {accuracy:.2%} | Err: {error_rate:.2%} | Latency: {avg_latency_s:.4f}s | Avg tokens: {avg_tokens:.1f} | Per-token: {latency_per_token:.4f}s")

        # Log examples to WandB
        if wandb_examples and len(wandb_examples) > 0:
            table = wandb.Table(
                columns=["mode", "subject", "prompt_preview", "response", "prediction", "label", "correct", "status"],
                data=wandb_examples
            )
            step_suffix = f"_step{step}" if step is not None else ""
            wandb.log({f"eval_examples/{mode}{step_suffix}": table})
            print(f"[WandB] Logged {len(wandb_examples)} examples to eval_examples/{mode}{step_suffix}")

        if mode == "bridge":
            self.bridge.train()

        if collect_details:
            return accuracy, error_rate, avg_latency_s, detailed_results
        return accuracy, error_rate, avg_latency_s

    def _generate_batch(self, batch, mode, max_new_tokens):
        """Score a batch using logprob-based evaluation (single forward pass).
        
        Instead of generating tokens, we do a single forward pass and check
        which answer token (A, B, C, D) has the highest probability.
        This ensures fair latency comparison across all modes.
        
        Args:
            batch: Batch from dataloader
            mode: Evaluation mode
            max_new_tokens: Unused (kept for API compatibility)
            
        Returns:
            Tuple of (prompt_texts, predictions, labels, tokens_generated)
            predictions is a list of predicted answer letters (A/B/C/D)
        """
        import torch.nn.functional as F
        
        # Standard Tensors
        sharer_ids = batch['sharer_input_ids'].to(self.device)
        sharer_mask = batch['sharer_mask'].to(self.device)
        rec_full_ids = batch['receiver_prompt_ids'].to(self.device)
        rec_mask = batch['receiver_prompt_mask'].to(self.device)
        rec_kickstart = batch['receiver_kickstart_ids'].to(self.device)
        batch_size = sharer_ids.shape[0]

        prompt_texts = []
        predictions = []  # Changed from gen_texts - now stores predicted answers

        if mode == "bridge":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Build cache from receiver_prompt_ids (N-1 tokens)
                modified_cache = self.get_bridged_cache(sharer_ids, sharer_mask, rec_full_ids, rec_mask)

                # Full input for forward pass
                full_input_ids = torch.cat([rec_full_ids, rec_kickstart], dim=1)
                full_mask = torch.cat([rec_mask, torch.ones_like(rec_kickstart)], dim=1)

                # Single forward pass to get logits
                outputs = self.receiver(
                    input_ids=full_input_ids,
                    past_key_values=modified_cache,
                    attention_mask=full_mask
                )
                del modified_cache
                
            # Get logits for last position, extract answer token probs
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            answer_logits = logits[:, self.answer_ids_receiver]  # [batch_size, 4]
            pred_indices = answer_logits.argmax(dim=-1)  # [batch_size]
            predictions = [self.answer_tokens_receiver[idx] for idx in pred_indices.tolist()]
            prompt_texts = self.tok_receiver.batch_decode(full_input_ids, skip_special_tokens=False)
            del outputs

        elif mode == "receiver_only":
            full_input = torch.cat([rec_full_ids, rec_kickstart], dim=1)
            full_mask = torch.cat([rec_mask, torch.ones_like(rec_kickstart)], dim=1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.receiver(
                    input_ids=full_input,
                    attention_mask=full_mask
                )
            
            logits = outputs.logits[:, -1, :]
            answer_logits = logits[:, self.answer_ids_receiver]
            pred_indices = answer_logits.argmax(dim=-1)
            predictions = [self.answer_tokens_receiver[idx] for idx in pred_indices.tolist()]
            prompt_texts = self.tok_receiver.batch_decode(full_input, skip_special_tokens=False)
            del outputs, full_input, full_mask

        elif mode == "sharer_only":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.sharer(
                    input_ids=sharer_ids,
                    attention_mask=sharer_mask
                )
            
            logits = outputs.logits[:, -1, :]
            answer_logits = logits[:, self.answer_ids_sharer]
            pred_indices = answer_logits.argmax(dim=-1)
            predictions = [self.answer_tokens_receiver[idx] for idx in pred_indices.tolist()]
            prompt_texts = self.tok_sharer.batch_decode(sharer_ids, skip_special_tokens=False)
            del outputs

        elif mode == "text_to_text":
            # Text-to-text still needs generation for the hint, but we score with logprobs
            raw_contexts = batch['raw_context']

            # 1. Sharer generates hint
            sharer_inputs_formatted = []
            for ctx in raw_contexts:
                clean_ctx = ctx.replace("Question: ", "").strip()
                s_prompt = f"In one clear sentence, describe the most essential background knowledge needed to answer the question: {clean_ctx} Do NOT directly solve or give answer to the question."
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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                s_out = self.sharer.generate(
                    s_inputs,
                    attention_mask=s_attn_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tok_sharer.pad_token_id,
                    eos_token_id=self.tok_sharer.eos_token_id,
                    do_sample=False
                )

            s_hints = self.tok_sharer.batch_decode(
                s_out[:, s_inputs.shape[1]:],
                skip_special_tokens=True
            )

            # 2. Receiver scores with logprobs (using hint)
            receiver_inputs_formatted = []
            for ctx, hint in zip(raw_contexts, s_hints):
                clean_hint = hint.replace("\n", " ").strip()
                final_content = f"{ctx}\n[Hint: {clean_hint}]"
                receiver_inputs_formatted.append([{"role": "user", "content": final_content}])

            r_encoded = self.tok_receiver.apply_chat_template(
                receiver_inputs_formatted,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True
            )
            r_inputs = r_encoded["input_ids"].to(self.device)
            r_attn_mask = r_encoded["attention_mask"].to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.receiver(
                    input_ids=r_inputs,
                    attention_mask=r_attn_mask
                )

            logits = outputs.logits[:, -1, :]
            answer_logits = logits[:, self.answer_ids_receiver]
            pred_indices = answer_logits.argmax(dim=-1)
            predictions = [self.answer_tokens_receiver[idx] for idx in pred_indices.tolist()]
            prompt_texts = [m[0]['content'] for m in receiver_inputs_formatted]
            del s_inputs, s_attn_mask, s_out, r_inputs, r_attn_mask, outputs

        # Cleanup
        del sharer_ids, sharer_mask, rec_full_ids, rec_mask

        # tokens_generated is always 0 for logprob-based eval (we don't generate)
        # We return predictions as gen_texts for compatibility with _score_batch
        return prompt_texts, predictions, batch['labels'], 0

    def _score_batch(self, prompt_texts, gen_texts, labels, stats, subjects=None, collect_details=False, wandb_examples=None, wandb_max=10, mode="unknown"):
        """Scores the batch and collects examples for WandB if enabled.
        
        Args:
            prompt_texts: List of prompts
            gen_texts: List of predicted answers (A/B/C/D from logprobs) or generated texts
            labels: List of correct labels
            stats: Dictionary of statistics to update
            subjects: List of subjects (optional)
            collect_details: If True, collect detailed results
            wandb_examples: List to append WandB examples to (optional)
            wandb_max: Maximum number of examples to collect for WandB
            mode: Evaluation mode name
            
        Returns:
            List of detailed results if collect_details=True, else None
        """
        if subjects is None:
            subjects = ['unknown'] * len(labels)

        batch_details = [] if collect_details else None

        for prompt, pred, truth, subject in zip(prompt_texts, gen_texts, labels, subjects):
            # With logprob-based eval, predictions are always valid (A/B/C/D)
            # No INVALID cases possible since we always pick the highest probability answer
            
            is_correct = (pred == truth)
            if is_correct:
                stats["correct"] += 1

            stats["total"] += 1

            # Collect detailed results if requested
            if collect_details:
                batch_details.append({
                    "pred": pred,
                    "label": truth,
                    "subject": subject,
                    "correct": is_correct
                })

            # Collect examples for WandB logging (limit to wandb_max)
            if wandb_examples is not None and len(wandb_examples) < wandb_max:
                if is_correct:
                    status = "correct"
                    correct_symbol = "✓"
                else:
                    status = "wrong"
                    correct_symbol = "✗"
                
                prompt_preview = prompt.strip()
                
                wandb_examples.append([
                    mode,  # mode
                    subject,  # subject
                    prompt_preview,  # prompt_preview
                    pred,  # response (predicted letter)
                    pred,  # prediction
                    truth,  # label
                    correct_symbol,  # correct
                    status  # status
                ])

        return batch_details



    def _extract_json_answer(self, text):
        """Extracts answer letter from text.
        
        Args:
            text: Generated text
            
        Returns:
            str: Answer letter (A/B/C/D) or "INVALID"
        """
        text = text.strip()

        # "answer: A", "answer is A", "answer A", etc.
        match = re.search(r'answer[\s:\"\']*([A-D])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # "the answer is A"
        match = re.search(r'answer\s*is\s*([A-D])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # "option A", "option A)", "option: A"
        match = re.search(r'option[\s:]*([A-D])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Starts with "A)" or "A." or "A " followed by text (e.g., "C) Hurler syndrome")
        match = re.match(r'^([A-D])[\)\.\s]', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Just "(A)" or "A" alone with optional parens/whitespace
        match = re.match(r'^[\(\s]*([A-D])[\)\s]*$', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Repeated letter like "C C" or "A A"
        match = re.match(r'^([A-D])\s+\1', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Last resort: find the first standalone A/B/C/D in the text
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1).upper()

        return "INVALID"
