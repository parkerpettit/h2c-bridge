"""MMLU evaluation."""

import re
import time

import torch
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

    @torch.no_grad()
    def evaluate_baselines(self, dataloader, max_new_tokens=15):
        """Runs static baselines.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            
        Returns:
            dict: Results for each mode
        """
        print("\n" + "="*30)
        print(">>> RUNNING BASELINES (Deterministic)")
        print("="*30 + "\n")

        results = {}
        modes = ["receiver_only", "sharer_only", "text_to_text"]

        for mode in modes:
            print(f"--- Running Baseline: {mode} ---")
            acc, err, lat = self._eval_loop(dataloader, mode=mode, max_new_tokens=max_new_tokens)
            results[mode] = {"acc": acc, "err": err, "latency_ms": lat}

        return results

    @torch.no_grad()
    def evaluate_baselines_detailed(self, dataloader, max_new_tokens=15, include_text_to_text=True):
        """Runs baselines with detailed logging.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            include_text_to_text: Include text-to-text baseline
            
        Returns:
            dict: Results with details
        """
        print("\n" + "="*30)
        print(">>> RUNNING BASELINES (Detailed)")
        print("="*30 + "\n")

        results = {}
        modes = ["receiver_only", "sharer_only"]
        if include_text_to_text:
            modes.append("text_to_text")

        for mode in modes:
            print(f"--- Running Baseline (Detailed): {mode} ---")
            acc, err, lat, details = self._eval_loop(
                dataloader, mode=mode, max_new_tokens=max_new_tokens, collect_details=True
            )
            results[mode] = {
                "acc": acc,
                "err": err,
                "latency_ms": lat,
                "details": details
            }

        return results

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader, max_new_tokens=15):
        """Evaluates bridge accuracy.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            
        Returns:
            (accuracy, error_rate, latency_ms)
        """
        self.bridge.eval()
        return self._eval_loop(dataloader, mode="bridge", max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def evaluate_accuracy_detailed(self, dataloader, max_new_tokens=15):
        """Evaluates bridge with detailed logging.
        
        Args:
            dataloader: MMLU dataloader
            max_new_tokens: Max tokens to generate
            
        Returns:
            (accuracy, error_rate, latency, detailed_results)
        """
        self.bridge.eval()
        return self._eval_loop(dataloader, mode="bridge", max_new_tokens=max_new_tokens, collect_details=True)

    def _eval_loop(self, dataloader, mode, max_new_tokens, collect_details=False):
        """Main evaluation loop.
        
        Args:
            dataloader: Dataloader
            mode: Evaluation mode ("bridge", "receiver_only", "sharer_only", "text_to_text")
            max_new_tokens: Maximum tokens to generate
            collect_details: If True, collect detailed results for each sample
            
        Returns:
            Tuple of (accuracy, error_rate, latency_ms) or (accuracy, error_rate, latency_ms, detailed_results) if collect_details=True
        """
        stats = {"correct": 0, "total": 0, "format_errors": 0, "total_time": 0.0}
        detailed_results = [] if collect_details else None

        progress_bar = tqdm(dataloader, desc=f"Eval [{mode}]", leave=False, mininterval=3.0)

        for batch_idx, batch in enumerate(progress_bar):
            # 1. Measure Generation Time
            start_time = time.time()
            prompt_texts, gen_texts, labels = self._generate_batch(batch, mode, max_new_tokens)
            end_time = time.time()

            # Get subjects if available (for detailed analysis)
            subjects = batch.get('subjects', ['unknown'] * len(labels))

            # Update Time Stats
            stats["total_time"] += (end_time - start_time)

            # 2. Score & Log
            batch_details = self._score_batch(prompt_texts, gen_texts, labels, stats, subjects, collect_details)
            if collect_details and batch_details:
                detailed_results.extend(batch_details)

            # Periodic memory cleanup (every 10 batches)
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Calculate Metrics
        total = stats["total"] if stats["total"] > 0 else 1
        accuracy = stats["correct"] / total
        error_rate = stats["format_errors"] / total
        avg_latency_ms = (stats["total_time"] / total)

        print(f"[{mode}] Acc: {accuracy:.2%} | Err: {error_rate:.2%} | Latency: {avg_latency_ms:.1f}ms")

        if mode == "bridge":
            self.bridge.train()

        if collect_details:
            return accuracy, error_rate, avg_latency_ms, detailed_results
        return accuracy, error_rate, avg_latency_ms

    def _generate_batch(self, batch, mode, max_new_tokens):
        """Generate outputs for a batch using the specified mode.
        
        Args:
            batch: Batch from dataloader
            mode: Generation mode
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (prompt_texts, gen_texts, labels)
        """
        # Standard Tensors
        sharer_ids = batch['sharer_input_ids'].to(self.device)
        sharer_mask = batch['sharer_mask'].to(self.device)
        rec_full_ids = batch['receiver_prompt_ids'].to(self.device)
        rec_mask = batch['receiver_prompt_mask'].to(self.device)

        prompt_texts = []
        gen_texts = []

        if mode == "bridge":
            # Bridge Logic (Standard) - use autocast to match training dtype
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Create cache from all but last token
                rec_context = rec_full_ids[:, :-1]
                rec_context_mask = rec_mask[:, :-1]  # Slice mask to match context
                modified_cache = self.get_bridged_cache(sharer_ids, sharer_mask, rec_context, rec_context_mask)
                del rec_context, rec_context_mask

                outputs = self.receiver.generate(
                    input_ids=rec_full_ids,  # Pass FULL input (model handles cache internally)
                    past_key_values=modified_cache,
                    attention_mask=rec_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tok_receiver.pad_token_id,
                    do_sample=False
                )
                del modified_cache
            prompt_texts = self.tok_receiver.batch_decode(rec_full_ids, skip_special_tokens=False)
            gen_texts = self.tok_receiver.batch_decode(outputs[:, rec_full_ids.shape[1]:], skip_special_tokens=False)
            del outputs

        elif mode == "receiver_only":
            # Reconstruct full input (Prompt + Kickstart)
            rec_kickstart = batch['receiver_kickstart_ids'].to(self.device)
            full_input = torch.cat([rec_full_ids, rec_kickstart], dim=1)
            full_mask = torch.cat([rec_mask, torch.ones_like(rec_kickstart)], dim=1)

            outputs = self.receiver.generate(
                input_ids=full_input, attention_mask=full_mask,
                max_new_tokens=max_new_tokens, pad_token_id=self.tok_receiver.pad_token_id, do_sample=False
            )
            prompt_texts = self.tok_receiver.batch_decode(full_input, skip_special_tokens=False)
            gen_texts = self.tok_receiver.batch_decode(outputs[:, full_input.shape[1]:], skip_special_tokens=False)
            del outputs, full_input, full_mask

        elif mode == "sharer_only":
            # Sharer input is already full (padded) in the batch
            outputs = self.sharer.generate(
                input_ids=sharer_ids, attention_mask=sharer_mask,
                max_new_tokens=max_new_tokens, pad_token_id=self.tok_sharer.pad_token_id, do_sample=False
            )
            prompt_texts = self.tok_sharer.batch_decode(sharer_ids, skip_special_tokens=False)
            gen_texts = self.tok_sharer.batch_decode(outputs[:, sharer_ids.shape[1]:], skip_special_tokens=False)
            del outputs

        elif mode == "text_to_text":
            # Text-to-text knowledge transfer: Sharer generates hint, Receiver uses it
            
            # 1. Get Raw Components
            raw_contexts = batch['raw_context']
            raw_instructions = batch['raw_instruction']

            # 2. Sharer Step (Generate Hint)
            sharer_inputs_formatted = []
            for ctx in raw_contexts:
                s_prompt = f"{ctx}\n\nIn one sentence, give a hint on how to solve the problem. Do NOT explicitly say the answer. Focus on the key logic and approach needed."
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
                max_new_tokens=48,
                pad_token_id=self.tok_sharer.pad_token_id,
                do_sample=False
            )

            # Decode only the new tokens (the hint)
            s_hints = self.tok_sharer.batch_decode(
                s_out[:, s_inputs.shape[1]:],
                skip_special_tokens=False
            )

            # 3. Receiver Step (Context + Hint + Instruction)
            receiver_inputs_formatted = []
            for ctx, hint, instr in zip(raw_contexts, s_hints, raw_instructions):
                # Clean the hint to prevent formatting issues
                clean_hint = hint.replace("\n", " ").strip()

                # Construct the final prompt
                final_content = f"{ctx}\n\n[Hint: {clean_hint}]\n\n{instr}"
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

            r_out = self.receiver.generate(
                r_inputs,
                attention_mask=r_attn_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok_receiver.pad_token_id,
                do_sample=False
            )

            # Store prompts for logging (we use the raw text version for readability)
            prompt_texts = [m[0]['content'] for m in receiver_inputs_formatted]
            gen_texts = self.tok_receiver.batch_decode(
                r_out[:, r_inputs.shape[1]:],
                skip_special_tokens=False
            )
            # Cleanup T2T tensors
            del s_inputs, s_attn_mask, s_out, r_inputs, r_attn_mask, r_out

        # Cleanup - tensors created at function start
        del sharer_ids, sharer_mask, rec_full_ids, rec_mask

        return prompt_texts, gen_texts, batch['labels']

    def _score_batch(self, prompt_texts, gen_texts, labels, stats, subjects=None, collect_details=False):
        """Scores the batch and prints verbose logs if enabled.
        
        Args:
            prompt_texts: List of prompts
            gen_texts: List of generated texts
            labels: List of correct labels
            stats: Dictionary of statistics to update
            subjects: List of subjects (optional)
            collect_details: If True, collect detailed results
            
        Returns:
            List of detailed results if collect_details=True, else None
        """
        if subjects is None:
            subjects = ['unknown'] * len(labels)

        batch_details = [] if collect_details else None

        for prompt, gen_text, truth, subject in zip(prompt_texts, gen_texts, labels, subjects):
            pred = self._extract_json_answer(gen_text)

            is_correct = False
            is_invalid = False

            if pred == "INVALID":
                stats["format_errors"] += 1
                is_invalid = True
            elif pred == truth:
                stats["correct"] += 1
                is_correct = True

            stats["total"] += 1

            # Collect detailed results if requested
            if collect_details:
                batch_details.append({
                    "pred": pred,
                    "label": truth,
                    "subject": subject,
                    "correct": is_correct,
                    "invalid": is_invalid
                })

            # --- VERBOSE LOGGING ---
            if self.config.get("verbose", False):
                self._log_sample(prompt, gen_text, truth, pred, is_correct, is_invalid)

        return batch_details

    def _log_sample(self, prompt, gen_text, truth, pred, is_correct, is_invalid):
        """Log a single sample (verbose mode)."""
        if is_invalid:
            status = "⚠️ FORMAT ERROR"
        else:
            status = "✅ CORRECT" if is_correct else "❌ WRONG"

        print("\n" + "="*50)
        # Only show last 300 chars of prompt to keep log clean
        print(f"--- PROMPT ---\n...{prompt.strip()[-300:]}")
        print(f"--- RESPONSE ---\n{gen_text.strip()}")
        print(f"--- EVAL ---")
        print(f"Truth: {truth} | Pred: {pred} | Result: {status}")
        print("="*50 + "\n")

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
