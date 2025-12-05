"""Dataset classes."""

from datasets import load_dataset
from torch.utils.data import Dataset


class H2CDatasetWrapper(Dataset):
    """OpenHermes-2.5 wrapper.
    
    Parses OpenHermes-2.5 into prompt/target pairs.
    """
    
    def __init__(self, split: str = "train", max_samples: int = None, tokenizer=None, max_len=2048):
        """Initialize the dataset.
        
        Args:
            split: Dataset split to load (default: "train")
            max_samples: Maximum number of samples to load (default: None, load all)
            tokenizer: Tokenizer for length filtering (optional)
            max_len: Maximum sequence length for filtering (default: 2048)
        """
        print(f"Loading OpenHermes-2.5 ({split})...")
        # Load the dataset from HuggingFace
        self.dataset = load_dataset("teknium/OpenHermes-2.5", split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))
        
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data = []
        self._process_data()

    def _process_data(self):
        """Iterates through the dataset and extracts the first instruction/response pair."""
        valid_count = 0
        skipped_count = 0
        
        for entry in self.dataset:
            convs = entry.get('conversations', [])

            # We look for the first 'human' (or 'user') and 'gpt' (or 'assistant') pair
            first_input = None
            first_target = None

            # Simple logic: Find the first human -> assistant pair (standalone, no context needed)
            for i in range(len(convs) - 1):
                msg = convs[i]
                next_msg = convs[i+1]

                if msg['from'] in ['human', 'user'] and next_msg['from'] in ['gpt', 'assistant']:
                    first_input = msg['value']      # human message -> input (prompt)
                    first_target = next_msg['value'] # assistant message -> target (response)
                    break

            if first_input and first_target:
                # Filter by length if tokenizer is provided
                if self.tokenizer:
                    # Check combined length (approximate check for sharer input)
                    # We add generation prompt in collator, so we simulate it here
                    full_text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": first_input}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # Rough check: tokenize just the prompt part since that's what hits the sharer limit most often
                    # Ideally we check prompt + target but prompt is the main constraint for the sharer input
                    ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
                    
                    if len(ids) > self.max_len:
                        skipped_count += 1
                        continue

                self.data.append({
                    "prompt": first_input,
                    "target": first_target
                })
                valid_count += 1

        print(f"Processed {valid_count} valid conversation pairs (Skipped {skipped_count} > {self.max_len} tokens).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We return raw text here. Tokenization happens in the Collator.
        # This allows for dynamic padding (much faster training).
        return self.data[idx]


class MMLUDataset(Dataset):
    """MMLU wrapper.
    
    Parses MMLU into prompt/target pairs.
    Used for both Auxiliary Training and Validation.
    """
    
    def __init__(self, split: str = "auxiliary_train", samples_per_subject: int = None, max_samples: int = None):
        """Initialize the dataset.
        
        Args:
            split: Dataset split to load (options: "auxiliary_train", "validation", "test")
            samples_per_subject: Number of samples per subject/category (for validation/test splits)
            max_samples: Maximum total samples (for auxiliary_train which has no subjects)
        """
        print(f"--- [MMLU] Loading {split} split...")
        # 'all' loads all subjects
        self.dataset = load_dataset("cais/mmlu", "all", split=split)
        self.samples_per_subject = samples_per_subject
        self.max_samples = max_samples

        self.data = []
        self._process_data()
        print(f"--- [MMLU] Processed {len(self.data)} examples ({split}).")

    def _process_data(self):
        """Process the raw MMLU data into prompt-target pairs.
        
        For auxiliary_train: uses max_samples (no subject grouping)
        For validation/test: uses samples_per_subject
        """
        # Group examples by subject
        subject_groups = {}
        for entry in self.dataset:
            subject = entry.get("subject", "unknown")
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(entry)
        
        num_subjects = len(subject_groups)
        print(f"--- [MMLU] Found {num_subjects} subjects")
        
        # Determine which entries to process
        if num_subjects == 1 and self.max_samples is not None:
            # auxiliary_train has no subject field - just use max_samples
            entries_to_process = list(self.dataset)[:self.max_samples]
        else:
            # validation/test has subjects - sample per subject
            entries_to_process = []
            for subject, entries in subject_groups.items():
                if self.samples_per_subject is not None:
                    entries = entries[:self.samples_per_subject]
                entries_to_process.extend(entries)
        
        # Process all selected entries
        for entry in entries_to_process:
            q = entry["question"]
            choices = entry["choices"]
            answer_idx = entry["answer"]  # 0..3
            answer_letter = "ABCD"[answer_idx]
            subject = entry.get("subject", "unknown")

            # Build the prompt
            context = "Question: " + q + "\n"
            for i, choice in enumerate(choices):
                context += f"{'ABCD'[i]}) {choice}\n"
            
            instruction = (
                "Carefully read the question and all options.\n"
                "Respond with only the letter of the correct answer (A, B, C, or D)."
            )

            full_prompt = instruction + "\n" + context

            self.data.append({
                "prompt": full_prompt,
                "target": answer_letter,
                "subject": subject
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
