"""Training configuration."""

# Full list of MMLU subjects
MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_studies', 'sociology',
    'us_foreign_policy', 'virology', 'world_religions'
]


def get_default_config():
    """Returns default configuration.
    
    Returns:
        dict: Default config values
    """
    return {
        "SHARER_ID": "meta-llama/Llama-3.1-8B-Instruct",
        "RECEIVER_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        
        # Dataset size
        "MAX_SAMPLES": 100_000,  # max samples of OpenHermes to pretrain bridge on
        "MAX_LEN": 2048,  # Max sequence length (filter longer examples)
        "BATCH_SIZE": 8,
        "lr": 1e-4,
        
        # Evaluation frequency (in steps)
        "eval_every": 1000,
        "log_bridge_every": 50,  # log bridge gate stats to wandb
        
        # Training
        "epochs": 1,
        "gate_warmup_steps": 0,
        
        # MMLU evaluation
        "mmlu_sample_size": 5,  # samples per category
        
        # Logging
        "verbose": False,  # True prints debug output in eval
        
        # Baseline results (run calculate_baselines.py to generate)
        "BASELINES": {
            "receiver_only": {
                "acc": 0.3590994371482176,
                "err": 0.013696060037523453,
                "latency_ms": 0.022912156872633028
            },
            "sharer_only": {
                "acc": 0.6872420262664165,
                "err": 0.00075046904315197,
                "latency_ms": 0.08787205532388884
            },
            "text_to_text": {
                "acc": 0.38574108818011255,
                "err": 0.0031894934333958724,
                "latency_ms": 0.8592565691269808
            }
        }
    }


def merge_config(user_config=None):
    """Merges user config with defaults.
    
    Args:
        user_config: User-specified values
        
    Returns:
        dict: Merged configuration
    """
    config = get_default_config()
    if user_config:
        config.update(user_config)
    return config
