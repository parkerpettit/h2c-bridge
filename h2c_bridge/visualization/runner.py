"""Visualization runner."""

import wandb
from h2c_bridge.visualization.theme import Theme, apply_theme
from h2c_bridge.visualization.charts import (
    log_performance_charts, log_gate_dynamics, log_bridge_heatmap,
    log_probability_shift, log_category_breakdown, log_confusion_matrix,
    log_layer_radar, log_attention_flow, log_qualitative_table,
    log_training_summary
)


def _generate_all_charts(engine, config, eval_cache, detailed_results, baseline_results, theme_suffix=""):
    """Generates all charts."""
    # Temporarily modify wandb.log to add suffix
    original_wandb_log = wandb.log
    def suffixed_log(data, *args, **kwargs):
        if theme_suffix:
            data = {f"{k}{theme_suffix}": v for k, v in data.items()}
        return original_wandb_log(data, *args, **kwargs)
    wandb.log = suffixed_log

    try:
        # --- Performance Charts ---
        log_performance_charts(engine, config, eval_cache=eval_cache, baseline_results=baseline_results)

        # --- Gate Dynamics ---
        log_gate_dynamics(engine)

        # --- Injection Heatmap ---
        log_bridge_heatmap(engine, prompt_text="Explain why the sky is blue.")

        # --- Probability Shift ---
        log_probability_shift(
            engine,
            prompt_text="Question: If x + 2 = 10, what is x?\nA) 5\nB) 8\nC) 12\nD) 2",
            true_label="B"
        )

        # --- Category Breakdown (with baseline comparison) ---
        log_category_breakdown(engine, config,
                               detailed_results=detailed_results,
                               baseline_results=baseline_results)

        # --- Confusion Matrix (with baseline comparison) ---
        log_confusion_matrix(engine,
                             detailed_results=detailed_results,
                             baseline_results=baseline_results)

        # --- Layer Radar Chart ---
        log_layer_radar(engine)

        # --- Attention Flow Sankey ---
        log_attention_flow(engine)

        # --- Qualitative Table (only once, theme doesn't affect it) ---
        # if not theme_suffix:  # Only log table once
        log_qualitative_table(engine)

        # --- Training Summary ---
        log_training_summary(engine, config, eval_cache=eval_cache, baseline_results=baseline_results)
    finally:
        # Restore original wandb.log
        wandb.log = original_wandb_log


def run_all_visualizations(engine, config, themes=("dark", "light"), include_baseline_comparison=True):
    """Runs all visualizations.
    
    Args:
        engine: Trained engine
        config: Config dict
        themes: Themes to use
        include_baseline_comparison: Whether to run baselines
    """
    print("\n" + "="*60)
    print("VISUALIZATION SUITE - Publication Ready")
    print("="*60 + "\n")

    # Validate themes
    valid_themes = {"dark", "light"}
    themes = [t for t in themes if t in valid_themes]
    if not themes:
        themes = ["dark"]

    print(f"[Viz] Generating charts for theme(s): {', '.join(themes)}")

    # ==========================================================================
    # STEP 1: Run H2C Bridge evaluation (detailed)
    # ==========================================================================
    print("\n[Viz] Running H2C Bridge evaluation...")
    try:
        h2c_acc, h2c_err, h2c_lat, detailed_results = engine.mmlu_evaluator.evaluate_accuracy_detailed(
            engine.mmlu_loader
        )
        eval_cache = {'acc': h2c_acc, 'err': h2c_err, 'lat': h2c_lat}
        print(f"[Viz] H2C Bridge: Accuracy={h2c_acc:.2%}, Latency={h2c_lat:.3f}s")
    except Exception as e:
        print(f"[Viz] Warning: Detailed evaluation failed ({e}), falling back to basic eval")
        h2c_acc, h2c_err, h2c_lat = engine.mmlu_evaluator.evaluate_accuracy(engine.mmlu_loader)
        eval_cache = {'acc': h2c_acc, 'err': h2c_err, 'lat': h2c_lat}
        detailed_results = None

    # ==========================================================================
    # STEP 2: Run baseline evaluations for comparison (optional)
    # ==========================================================================
    baseline_results = None
    if include_baseline_comparison:
        print("\n[Viz] Running baseline evaluations for comparison...")
        try:
            baseline_results = engine.mmlu_evaluator.evaluate_baselines_detailed(engine.mmlu_loader)
            for mode, results in baseline_results.items():
                print(f"[Viz] {mode}: Accuracy={results['acc']:.2%}")
        except Exception as e:
            print(f"[Viz] Warning: Baseline evaluation failed ({e}), skipping comparisons")
            baseline_results = None

    # ==========================================================================
    # STEP 3: Generate charts for each theme
    # ==========================================================================
    for i, theme_name in enumerate(themes):
        print(f"\n{'='*40}")
        print(f"Generating {theme_name.upper()} theme charts ({i+1}/{len(themes)})")
        print(f"{'='*40}")

        # Switch theme
        Theme.set_theme(theme_name)

        # Determine suffix for WandB keys
        # If only one theme, no suffix. If both, add _dark or _light
        if len(themes) == 1:
            suffix = ""
        else:
            suffix = f"_{theme_name}"

        # Generate all charts
        _generate_all_charts(engine, config, eval_cache, detailed_results, baseline_results, suffix)

    # ==========================================================================
    # DONE
    # ==========================================================================
    print("\n" + "="*60)
    print("All visualizations logged to WandB successfully!")
    print(f"  - Themes generated: {', '.join(themes)}")
    print("  - 10 chart types per theme")
    if include_baseline_comparison:
        print("  - Includes baseline comparisons in confusion matrix & category breakdown")
    if len(themes) > 1:
        print("  - Use '_dark' or '_light' suffix in WandB to filter")
    print("="*60 + "\n")


def run_visualizations_light_only(engine, config):
    """Runs light theme only."""
    run_all_visualizations(engine, config, themes=("light",))


def run_visualizations_dark_only(engine, config):
    """Runs dark theme only."""
    run_all_visualizations(engine, config, themes=("dark",))
