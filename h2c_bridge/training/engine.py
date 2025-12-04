"""Training engine."""

import os

import torch
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from h2c_bridge.training.trainer import H2CTrainer
from h2c_bridge.evaluation.evaluator import H2CEvaluator
from h2c_bridge.evaluation.mmlu_evaluator import H2CMMLUEvaluator
from h2c_bridge.evaluation.baseline_perplexity import BaselinePerplexityEvaluator


class H2CEngine:
    """Orchestrates training.
    
    Manages models, data, training, evaluation, and logging.
    """
    
    def __init__(self, factory, data_module, config, lr=1e-4, eval_every=50, checkpoint_path=None, wandb_run_id=None):
        """Initialize the engine.
        
        Args:
            factory: H2CModelFactory instance
            data_module: H2CDataModule instance
            config: Configuration dictionary
            lr: Learning rate (default: 1e-4)
            eval_every: Evaluate every N steps (default: 50)
            checkpoint_path: Optional path to checkpoint to resume from
            wandb_run_id: Optional WandB run ID to resume
        """
        self.factory = factory
        self.dm = data_module
        self.config = config
        self.lr = lr
        self.eval_every = eval_every
        self.global_step = 0
        self.best_accuracy = 0.0  # Track best accuracy for "save best" logic

        # Create checkpoint naming based on model names
        sharer_name = config['SHARER_ID'].split('/')[-1].replace('.', '-')
        receiver_name = config['RECEIVER_ID'].split('/')[-1].replace('.', '-')
        self.checkpoint_prefix = f"bridge_{sharer_name}_TO_{receiver_name}"

        # Store wandb run ID for resuming
        self.wandb_run_id = wandb_run_id

        # Component Setup
        self._setup_wandb()
        self._setup_models()
        self._setup_optimization()
        self._setup_evaluators()

        # Load from checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _setup_wandb(self):
        """Initializes WandB."""
        run_name = f"{self.config['SHARER_ID'].split('/')[-1]}_TO_{self.config['RECEIVER_ID'].split('/')[-1]}"

        if self.wandb_run_id:
            # Resume existing run
            self.wandb_run = wandb.init(
                id=self.wandb_run_id,
                project="nlp_project",
                resume="must",  # Will error if run doesn't exist
                reinit=True
            )
            print(f"Resumed wandb run: {self.wandb_run_id}")
        else:
            # Start new run
            self.wandb_run = wandb.init(
                name=run_name.replace("\\", "-"),
                project="nlp_project",
                config=self.config,
                save_code=True,
                reinit=True
            )

    def _setup_models(self):
        """Loads models."""
        self.sharer, self.receiver = self.factory.load_llms()
        self.bridge = self.factory.create_bridge()

    def _setup_optimization(self):
        """Sets up optimization."""
        self.train_loader, self.val_loader = self.dm.get_loaders()
        self.mmlu_loader = self.dm.get_mmlu_loader()
        self.optimizer = optim.AdamW(self.bridge.parameters(), lr=self.lr)

        # Setup cosine scheduler with warmup
        epochs = self.config.get('epochs', 1)  # Default to 1 if not specified
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"--- [Scheduler] Cosine schedule: {total_steps} total steps, {warmup_steps} warmup steps")

        self.trainer = H2CTrainer(
            self.sharer, self.receiver, self.bridge, self.optimizer, self.config, device=self.factory.device
        )
        
        # CRITICAL FIX: Ensure engine uses the same bridge instance as the trainer
        # The trainer might have a different instance if optimization was re-setup
        if self.bridge is not self.trainer.bridge:
            print(f"--- [Engine] Syncing bridge reference from Trainer ({id(self.trainer.bridge)}) to Engine")
            self.bridge = self.trainer.bridge

    def _setup_evaluators(self):
        """Sets up evaluators."""
        # Ensure we use the bridge that is actually being trained (if trainer exists)
        bridge_to_use = self.trainer.bridge if hasattr(self, 'trainer') else self.bridge
        
        self.evaluator = H2CEvaluator(
            self.sharer, self.receiver, bridge_to_use, self.factory.tok_receiver, self.config, device=self.factory.device
        )
        self.mmlu_evaluator = H2CMMLUEvaluator(
            self.sharer, self.receiver, bridge_to_use, self.factory.tok_receiver, self.factory.tok_sharer, self.config, device=self.factory.device
        )
        self.baseline_ppl_evaluator = BaselinePerplexityEvaluator(
            self.sharer, self.receiver, self.factory.tok_sharer, self.factory.tok_receiver, device=self.factory.device
        )

    def run(self, epochs=3):
        """Runs training loop.
        
        Args:
            epochs: Number of epochs
        """
        print(f"--- [Engine] Starting Training for {epochs} epochs...")
        for epoch in range(epochs):
            self._run_epoch(epoch)

        # Save final checkpoint at end of training
        print("\n--- Training Complete ---")
        self._save_checkpoint(checkpoint_type="final")
        print(f"Best accuracy achieved: {self.best_accuracy:.2%}")

    def _run_epoch(self, epoch_idx):
        """Runs one epoch.
        
        Args:
            epoch_idx: Current epoch index
        """
        self.bridge.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx + 1}")
        log_bridge_every = self.config.get("log_bridge_every", 100)

        for batch in progress_bar:
            # Proactive memory check - skip batch if memory is critically low
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if allocated / total > 0.90:  # >90% used
                    print(f"\n[WARNING] Memory at {allocated:.1f}/{total:.1f}GB ({allocated/total:.0%}), skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            
            try:
                loss = self.trainer.train_step(batch)
            except torch.cuda.OutOfMemoryError:
                # Handle OOM by skipping this batch (likely very long sequences)
                print(f"\n[WARNING] OOM at step {self.global_step}, clearing memory and skipping...")
                # Aggressive cleanup
                self.optimizer.zero_grad(set_to_none=True)  # More aggressive than zero_grad()
                if hasattr(self.trainer, 'bridge'):
                    self.trainer.bridge.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()  # Call again after gc
                continue  # Skip to next batch
            
            self.global_step += 1

            # Step the scheduler
            self.scheduler.step()

            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']

            progress_bar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{current_lr:.2e}"})
            wandb.log({
                "Training/Loss": loss,
                "Training/Learning Rate": current_lr,
                "epoch": epoch_idx + 1
            })

            # Log bridge stats periodically (without printing)
            if self.global_step % log_bridge_every == 0:
                self._log_bridge_stats(print_output=False)

            if self.global_step % self.eval_every == 0:
                self._perform_eval(progress_bar)

    def _log_bridge_stats(self, print_output=True):
        """Logs bridge stats.
        
        Args:
            print_output: Whether to print to console
        """
        stats = self.bridge.get_gate_stats()

        key_gates = stats["key_gates"]
        value_gates = stats["value_gates"]
        avg_key_gate = stats["key_avg"]
        avg_value_gate = stats["value_avg"]

        # Count how many gates are "on" (>0.5) vs "off" (<0.5)
        key_on = sum(1 for g in key_gates if g > 0.5)
        value_on = sum(1 for g in value_gates if g > 0.5)

        # Get raw gate values (before sigmoid)
        key_raw_gates = [block.gate.item() for block in self.bridge.key_modifiers]
        value_raw_gates = [block.gate.item() for block in self.bridge.value_modifiers]
        avg_key_raw = sum(key_raw_gates) / len(key_raw_gates)
        avg_value_raw = sum(value_raw_gates) / len(value_raw_gates)

        # Helper functions
        def _mean(values):
            return sum(values) / len(values) if values else 0.0

        def _std(values, mean):
            return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0

        # Get attention weight norms
        key_attn_norms = [block.attn.out_proj.weight.norm().item() for block in self.bridge.key_modifiers]
        value_attn_norms = [block.attn.out_proj.weight.norm().item() for block in self.bridge.value_modifiers]
        avg_key_attn_norm = _mean(key_attn_norms)
        avg_value_attn_norm = _mean(value_attn_norms)
        std_key_attn_norm = _std(key_attn_norms, avg_key_attn_norm)
        std_value_attn_norm = _std(value_attn_norms, avg_value_attn_norm)

        if print_output:
            print(f"Bridge Gates: Key avg={avg_key_gate:.3f} ({key_on}/{len(key_gates)} on) | "
                  f"Value avg={avg_value_gate:.3f} ({value_on}/{len(value_gates)} on)")
            print(f"  raw gates:      Key={avg_key_raw:.4f} | Value={avg_value_raw:.4f}")
            print(f"  attn norms:     Key={avg_key_attn_norm:.4f} | Value={avg_value_attn_norm:.4f}")

        # Log to wandb
        wandb.log({
            # Gate values (clamped to 0-1)
            "Bridge/Key Gate Avg": avg_key_gate,
            "Bridge/Value Gate Avg": avg_value_gate,
            "Bridge/Key Gates On": key_on,
            "Bridge/Value Gates On": value_on,
            # Raw gate values (before sigmoid)
            "Bridge/Key Raw Gate Avg": avg_key_raw,
            "Bridge/Value Raw Gate Avg": avg_value_raw,
            # Attention weight norms
            "Bridge/Key Attn Norm": avg_key_attn_norm,
            "Bridge/Value Attn Norm": avg_value_attn_norm,
            "Bridge/Key Attn Norm Min": min(key_attn_norms),
            "Bridge/Key Attn Norm Max": max(key_attn_norms),
            "Bridge/Value Attn Norm Min": min(value_attn_norms),
            "Bridge/Value Attn Norm Max": max(value_attn_norms),
            "Bridge/Key Attn Norm Std": std_key_attn_norm,
            "Bridge/Value Attn Norm Std": std_value_attn_norm,
        })

    def _perform_eval(self, pbar_ref=None):
        """Runs evaluation.
        
        Args:
            pbar_ref: Optional progress bar to clear
        """
        if pbar_ref:
            pbar_ref.clear()
        print(f"\n--- Evaluation at Step {self.global_step} ---")

        # 1. Standard Metrics
        val_loss = self.evaluator.evaluate_loss(self.val_loader)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()  # Perplexity = exp(loss)
        
        # Use max_new_tokens=30 to allow natural EOS stopping while preventing runaway generation
        mmlu_acc, mmlu_err, mmlu_lat = self.mmlu_evaluator.evaluate_accuracy(
            self.mmlu_loader, max_new_tokens=30
        )

        print(f"Validation/Loss: {val_loss:.4f}")
        print(f"Validation/Perplexity: {val_perplexity:.2f}")
        print(f"MMLU Accuracy: {mmlu_acc:.2%}")
        print(f"MMLU Latency:  {mmlu_lat:.1f}ms")
        
        # 2. Baseline Perplexities - use cached values from config (they never change)
        baselines = self.config.get("BASELINES", {})
        baseline_ppls = {
            "receiver_only_ppl": baselines.get("receiver_only", {}).get("ppl", 2.69),
            "sharer_only_ppl": baselines.get("sharer_only", {}).get("ppl", 21.15),
            "text_to_text_ppl": baselines.get("text_to_text", {}).get("ppl", 2.72),
        }

        # Log bridge gate values
        self._log_bridge_stats()

        # 3. Calculate Deltas and Organize Logs
        logs = {
            "Validation/Loss": val_loss,
            "Validation/Perplexity": val_perplexity,
            "MMLU/Accuracy": mmlu_acc,
            "MMLU/Error Rate": mmlu_err,
            "MMLU/Latency (s)": mmlu_lat,
        }
        
        # Add baseline perplexities if computed
        if baseline_ppls:
            logs.update({
                "Baselines/Receiver-Only Perplexity": baseline_ppls["receiver_only_ppl"],
                "Baselines/Sharer-Only Perplexity": baseline_ppls["sharer_only_ppl"],
                "Baselines/Text-to-Text Perplexity": baseline_ppls["text_to_text_ppl"],
            })

        if "BASELINES" in self.config:
            for name, stats in self.config["BASELINES"].items():
                clean_name = name.replace("_", " ").title()

                # Accuracy Delta
                acc_delta = mmlu_acc - stats["acc"]
                logs[f"Deltas/Accuracy Delta vs {clean_name}"] = acc_delta

                # Latency Delta
                lat_delta = mmlu_lat - stats["latency_ms"]
                logs[f"Deltas/Latency Delta (s) vs {clean_name}"] = lat_delta

                print(f"vs {name}: Acc {acc_delta:+.2%} | Lat {lat_delta:+.1f}ms")

        # 3. Log & Save
        wandb.log(logs)
        self.evaluator.generate_demo("Explain quantum entanglement like I'm five.", max_new_tokens=256)
        self._save_checkpoint(checkpoint_type="step", accuracy=mmlu_acc)

    def _save_checkpoint(self, checkpoint_type="step", accuracy=None):
        """Saves checkpoint as WandB Artifact.
        
        Args:
            checkpoint_type: "step", "best", or "final"
            accuracy: Current accuracy
        """
        import tempfile
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': self.global_step,
            'bridge_state_dict': self.bridge.state_dict(),
            'config': self.config
        }
        
        # Add accuracy to step/best checkpoints
        if checkpoint_type in ["step", "best"]:
            checkpoint_data['accuracy'] = accuracy
        elif checkpoint_type == "final":
            checkpoint_data['best_accuracy'] = self.best_accuracy
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as tmp:
            torch.save(checkpoint_data, tmp.name)
            tmp_path = tmp.name
        
        try:
            # Create WandB artifact
            artifact = wandb.Artifact(
                name=f"{self.checkpoint_prefix}_checkpoint",
                type="model",
                description=f"H2C Bridge checkpoint at step {self.global_step}",
                metadata={
                    'step': self.global_step,
                    'accuracy': accuracy if accuracy is not None else self.best_accuracy,
                    'checkpoint_type': checkpoint_type
                }
            )
            
            # Add checkpoint file
            artifact.add_file(tmp_path, name="checkpoint.pt")
            
            # Determine aliases
            aliases = [f"v{self.global_step}"]
            
            if checkpoint_type == "step":
                aliases.append("latest")
                print(f"Saved checkpoint: step {self.global_step}")
                
                # Check if this is the best accuracy
                if accuracy is not None and accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    aliases.append("best")
                    print(f"New best accuracy ({accuracy:.2%})!")
                    
            elif checkpoint_type == "best":
                aliases.append("best")
                print(f"Saved best checkpoint: accuracy {accuracy:.2%}")
                
            elif checkpoint_type == "final":
                aliases.append("final")
                print(f"Saved final checkpoint: best accuracy {self.best_accuracy:.2%}")
            
            # Log artifact with aliases
            self.wandb_run.log_artifact(artifact, aliases=aliases)
            print(f"Uploaded to WandB with aliases: {', '.join(aliases)}")
            
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        print("-" * 30)


    def load_checkpoint(self, checkpoint_path):
        """Loads checkpoint from WandB artifact or local file.
        
        Args:
            checkpoint_path: WandB artifact path (e.g., "entity/project/run-id/model:best")
                           or local file path for legacy checkpoints
            
        Returns:
            dict: Checkpoint metadata
        """
        # Check if it's a WandB artifact path (contains ":" for alias)
        if ":" in checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from WandB artifact: {checkpoint_path}")
            
            # Download artifact
            artifact = self.wandb_run.use_artifact(checkpoint_path, type="model")
            artifact_dir = artifact.download()
            
            # Load checkpoint from downloaded directory
            local_path = os.path.join(artifact_dir, "checkpoint.pt")
            checkpoint = torch.load(local_path, map_location=self.factory.device)
            
        else:
            # Legacy: load from local file path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print(f"Loading checkpoint from local file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.factory.device)
        
        # Handle both old format (just state_dict) and new format (dict with metadata)
        if isinstance(checkpoint, dict) and 'bridge_state_dict' in checkpoint:
            self.bridge.load_state_dict(checkpoint['bridge_state_dict'])
            self.global_step = checkpoint.get('step', 0)
            loaded_acc = checkpoint.get('accuracy') or checkpoint.get('best_accuracy', 0)
            self.best_accuracy = loaded_acc if loaded_acc else 0.0
            print(f"Loaded from step {self.global_step}, accuracy: {self.best_accuracy:.2%}")
            return checkpoint
        else:
            # Old format: just the state dict
            self.bridge.load_state_dict(checkpoint)
            print("Loaded checkpoint (legacy format, no metadata)")
            return {'bridge_state_dict': checkpoint}

