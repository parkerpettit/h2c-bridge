"""Data module."""

from torch.utils.data import DataLoader, random_split, ConcatDataset

from h2c_bridge.data.datasets import H2CDatasetWrapper, MMLUDataset
from h2c_bridge.data.collator import H2CDataCollator


class H2CDataModule:
    """Manages datasets and loaders.
    
    Loads, splits, and batches data for training and evaluation.
    """
    
    def __init__(self, tok_sharer, tok_receiver, config):
        """Initialize the data module.
        
        Args:
            tok_sharer: Tokenizer for the sharer model
            tok_receiver: Tokenizer for the receiver model
            config: Configuration dictionary
        """
        self.tok_sharer = tok_sharer
        self.tok_receiver = tok_receiver
        self.batch_size = config["BATCH_SIZE"]
        self.max_samples = config["MAX_SAMPLES"]
        self.samples_per_subject = config["mmlu_sample_size"]

        self.train_loader = None
        self.val_loader = None
        self.mmlu_loader = None

    def setup(self):
        """Prepares datasets and splits."""
        print(f"--- [DataModule] Loading Datasets (Max {self.max_samples})...")

        # 1. Base OpenHermes dataset
        oh_dataset = H2CDatasetWrapper(split="train", max_samples=self.max_samples)

        # 2. MMLU auxiliary train dataset (5% mix)
        #    We want MMLU to be ~5% of the total training data
        #    OpenHermes is max_samples. So MMLU should be max_samples * 0.05
        #    Use 5 samples per subject to get a reasonable mix across categories
        mmlu_samples_per_subject = 5
        mmlu_aux_dataset = MMLUDataset(split="auxiliary_train", samples_per_subject=mmlu_samples_per_subject)

        # 3. Combine them for training/validation
        full_dataset = ConcatDataset([oh_dataset, mmlu_aux_dataset])
        print(f"--- [DataModule] Combined Train Source Sizes: "
              f"{len(oh_dataset)} OpenHermes + {len(mmlu_aux_dataset)} MMLU Aux "
              f"= {len(full_dataset)} total")

        # 4. Train/Val split on the combined dataset
        train_size = int(0.99 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_set, self.val_set = random_split(full_dataset, [train_size, val_size])

        print(f"--- [DataModule] Split: {len(self.train_set)} Train | {len(self.val_set)} Val")

        # 5. Shared collator for train/val (same (prompt, target) interface)
        self.collator = H2CDataCollator(self.tok_sharer, self.tok_receiver)

        # 6. Train / Val loaders
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        # 7. MMLU EVAL (Validation Split)
        print(f"--- [DataModule] Setting up MMLU Eval (Validation Split)...")
        # Use validation split for evaluation with samples_per_subject from config
        mmlu_eval_dataset = MMLUDataset(split="validation", samples_per_subject=self.samples_per_subject)

        eval_batch_size = max(1, self.batch_size // 2)
        self.mmlu_loader = DataLoader(
            mmlu_eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

    def get_mmlu_loader(self):
        """Get the MMLU evaluation data loader.
        
        Returns:
            DataLoader for MMLU evaluation
        """
        if not self.train_loader:
            self.setup()
        return self.mmlu_loader

    def get_loaders(self):
        """Get the training and validation data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not self.train_loader:
            self.setup()
        return self.train_loader, self.val_loader
