"""Dataset classes."""

from datasets import load_dataset
from torch.utils.data import Dataset


class H2CDatasetWrapper(Dataset):
    """OpenHermes-2.5 wrapper.
    
    Parses OpenHermes-2.5 into prompt/target pairs.
    """
    
    def __init__(self, split: str = "train", max_samples: int = None):
        """Initialize the dataset.
        
        Args:
            split: Dataset split to load (default: "train")
            max_samples: Maximum number of samples to load (default: None, load all)
        """
        print(f"Loading OpenHermes-2.5 ({split})...")
        # Load the dataset from HuggingFace
        self.dataset = load_dataset("teknium/OpenHermes-2.5", split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

        self.data = []
        self._process_data()

    def _process_data(self):
        """Iterates through the dataset and extracts the first instruction/response pair."""
        valid_count = 0
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
                self.data.append({
                    "prompt": first_input,
                    "target": first_target
                })
                valid_count += 1

        print(f"Processed {valid_count} valid conversation pairs.")

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
    
    def __init__(self, split: str = "auxiliary_train", max_samples: int = None):
        """Initialize the dataset.
        
        Args:
            split: Dataset split to load (options: "auxiliary_train", "validation", "test")
            max_samples: Maximum number of samples to load (default: None, load all)
        """
        print(f"--- [MMLU] Loading {split} split...")
        # 'all' loads all subjects
        self.dataset = load_dataset("cais/mmlu", "all", split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

        self.data = []
        self._process_data()
        print(f"--- [MMLU] Processed {len(self.data)} examples ({split}).")

    def _process_data(self):
        """Process the raw MMLU data into prompt-target pairs."""
        for entry in self.dataset:
            q = entry["question"]
            choices = entry["choices"]
            answer_idx = entry["answer"]  # 0..3
            answer_letter = "ABCD"[answer_idx]

            # Build the prompt
            context = "Question: " + q + "\n"
            for i, choice in enumerate(choices):
                context += f"{'ABCD'[i]}) {choice}\n"

            instruction = (
                "\nThink carefully, then provide your answer. "
                "Output your final answer as a single letter (A, B, C, or D) on the last line in the format 'Answer': <letter>\n"
            )

            full_prompt = context + instruction

            self.data.append({
                "prompt": full_prompt,
                "target": answer_letter,
                "subject": entry.get("subject", "unknown")
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
