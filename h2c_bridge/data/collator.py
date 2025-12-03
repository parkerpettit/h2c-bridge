"""Data collation."""

import torch


class H2CDataCollator:
    """Batches training data.
    
    Tokenizes and pads inputs for bridge training.
    """
    
    def __init__(self, tokenizer_sharer, tokenizer_receiver, max_len=1024):
        """Initialize the collator.
        
        Args:
            tokenizer_sharer: Tokenizer for the sharer model
            tokenizer_receiver: Tokenizer for the receiver model
            max_len: Maximum sequence length (default: 1024)
        """
        self.tok_sharer = tokenizer_sharer
        self.tok_receiver = tokenizer_receiver
        self.max_len = max_len

        # Ensure pad tokens
        if self.tok_sharer.pad_token is None:
            self.tok_sharer.pad_token = self.tok_sharer.eos_token
        if self.tok_receiver.pad_token is None:
            self.tok_receiver.pad_token = self.tok_receiver.eos_token

    def _get_ids(self, tokenizer, prompt, add_gen):
        """Helper to tokenize prompts with chat template.
        
        Args:
            tokenizer: The tokenizer to use
            prompt: Text prompt
            add_gen: Whether to add generation prompt
            
        Returns:
            List of token IDs
        """
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=add_gen,
            return_dict=False
        )

    def __call__(self, batch):
        """Collate a batch of samples.
        
        Args:
            batch: List of dictionaries with 'prompt' and 'target' keys
            
        Returns:
            Dictionary with batched tensors for training
        """
        prompts = [x['prompt'] for x in batch]
        targets = [x['target'] for x in batch]

        # Lists for padding later
        sharer_ids_list = []
        rec_prompt_ids_list = []  # The prompt MINUS the last token
        rec_kickstart_list = []   # The extracted last token
        rec_target_ids_list = []

        for p, t in zip(prompts, targets):
            # 1. Sharer: Tokenize & Strip Last & Truncate
            s_raw = self._get_ids(self.tok_sharer, p, add_gen=True)
            s_raw = s_raw[:self.max_len]  # Truncate to max_len
            sharer_ids_list.append(s_raw[:-1])  # Strip last

            # 2. Receiver: Tokenize & Strip Last & Truncate
            r_raw = self._get_ids(self.tok_receiver, p, add_gen=True)
            r_raw = r_raw[:self.max_len]  # Truncate to max_len
            rec_prompt_ids_list.append(r_raw[:-1])  # Strip last (The 'Prompt')
            rec_kickstart_list.append([r_raw[-1]])  # Keep last (The 'Kickstart')

            # 3. Target: Tokenize & Truncate
            t_ids = self.tok_receiver(t, add_special_tokens=False)["input_ids"]
            t_ids = t_ids[:self.max_len]  # Truncate to max_len
            if len(t_ids) > 0 and t_ids[-1] != self.tok_receiver.eos_token_id:
                t_ids.append(self.tok_receiver.eos_token_id)
            rec_target_ids_list.append(t_ids)

        # --- Padding ---
        # 1. Sharer Inputs
        sharer_batch = self.tok_sharer.pad(
            {"input_ids": sharer_ids_list}, padding=True, return_tensors="pt"
        )

        # 2. Receiver Inputs (Stripped Prompt)
        rec_prompt_batch = self.tok_receiver.pad(
            {"input_ids": rec_prompt_ids_list}, padding=True, return_tensors="pt"
        )

        # 3. Receiver Kickstart (It's just 1 token, but we tensor-ize it)
        rec_kickstart_batch = torch.tensor(rec_kickstart_list)

        # 4. Receiver Targets
        rec_target_batch = self.tok_receiver.pad(
            {"input_ids": rec_target_ids_list}, padding=True, return_tensors="pt"
        )

        # 5. Metadata (Subjects & Labels for Eval)
        subjects = [x.get('subject', 'unknown') for x in batch]
        # 'target' in batch is the raw string (e.g. "A"). Use that as label.
        labels = [x.get('target', 'unknown') for x in batch]
        
        # 6. Raw text for text-to-text baseline
        raw_contexts = []
        raw_instructions = []
        for p in prompts:
            split_marker = "\nThink carefully"
            if split_marker in p:
                context, instruction = p.rsplit(split_marker, 1)
                raw_contexts.append(context)
                raw_instructions.append(split_marker + instruction)
            else:
                raw_contexts.append(p)
                raw_instructions.append("")

        return {
            "sharer_input_ids": sharer_batch['input_ids'],
            "sharer_mask": sharer_batch['attention_mask'],

            "receiver_prompt_ids": rec_prompt_batch['input_ids'],      # The prefill input
            "receiver_prompt_mask": rec_prompt_batch['attention_mask'], # Mask for prefill
            "receiver_kickstart_ids": rec_kickstart_batch,              # The \n token

            "receiver_target_ids": rec_target_batch['input_ids'],
            "receiver_target_mask": rec_target_batch['attention_mask'],  # Mask for targets

            "raw_context": raw_contexts,
            "raw_instruction": raw_instructions,

            "subjects": subjects,
            "labels": labels
        }
