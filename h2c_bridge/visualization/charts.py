"""Visualization charts."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import wandb

# Plotly for Sankey diagrams (optional - graceful fallback if not installed)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[Viz] Warning: plotly not installed. Sankey diagrams will be skipped.")

from h2c_bridge.visualization.theme import Theme, apply_theme
from h2c_bridge.visualization.utils import (
    create_custom_cmap, format_card, categorize_subject, safe_viz,
    add_glow, add_text_stroke
)


@safe_viz
def log_performance_charts(engine, config, eval_cache=None, baseline_results=None):
    """Generates performance charts.
    
    Creates scatter plot (accuracy vs latency) and bar chart.

    Args:
        eval_cache: Optional cached metrics
        baseline_results: Optional baseline metrics
    """
    print("[Viz] Generating performance charts...")
    apply_theme()

    # --- Data Preparation ---
    data = []

    # Prefer freshly calculated baseline_results - compute if not provided
    if baseline_results:
        for name, metrics in baseline_results.items():
            clean_name = name.replace("_", " ").title()
            lat = metrics.get("latency_s", 0)
            data.append({
                "Method": clean_name,
                "Accuracy": metrics["acc"],
                "Latency": lat,
                "Type": "baseline"
            })
    else:
        # Compute baselines fresh (don't use precomputed config values)
        print("[Viz] Computing baselines fresh...")
        fresh_baselines = engine.mmlu_evaluator.evaluate_baselines(engine.mmlu_loader)
        for name, metrics in fresh_baselines.items():
            clean_name = name.replace("_", " ").title()
            lat = metrics.get("latency_s", 0)
            data.append({
                "Method": clean_name,
                "Accuracy": metrics["acc"],
                "Latency": lat,
                "Type": "baseline"
            })

    # Add our method - use cache if available
    if eval_cache:
        h2c_acc, h2c_lat = eval_cache['acc'], eval_cache['lat']
    else:
        h2c_acc, _, h2c_lat = engine.mmlu_evaluator.evaluate_accuracy(engine.mmlu_loader)

    data.append({
        "Method": "H2C Bridge (Ours)",
        "Accuracy": h2c_acc,
        "Latency": h2c_lat,
        "Type": "ours"
    })
    df = pd.DataFrame(data)

    # ==========================================================================
    # CHART 1: Accuracy vs Latency Scatter
    # ==========================================================================
    fig1, ax = plt.subplots(figsize=(10, 6))
    format_card(ax, title="Performance Trade-off",
                 subtitle="Accuracy vs. Inference Latency")

    # Plot baselines first (smaller, muted)
    baselines = df[df["Type"] == "baseline"]
    ax.scatter(
        baselines["Latency"], baselines["Accuracy"],
        s=120, c=Theme.TEXT_MUTED, alpha=0.7,
        edgecolors=Theme.TEXT_SECONDARY, linewidth=1.5,
        zorder=2, label="Baselines"
    )

    # Plot our method (larger, glowing)
    ours = df[df["Type"] == "ours"]
    scatter_ours = ax.scatter(
        ours["Latency"], ours["Accuracy"],
        s=300, c=Theme.ACCENT_PRIMARY, alpha=1.0,
        edgecolors='white', linewidth=2,
        zorder=3, marker='D', label="H2C Bridge (Ours)"
    )

    # Add glow effect to our point
    ax.scatter(
        ours["Latency"], ours["Accuracy"],
        s=600, c=Theme.ACCENT_PRIMARY, alpha=0.15,
        zorder=1
    )

    # Annotations
    for _, row in df.iterrows():
        is_ours = row["Type"] == "ours"
        offset = (-12, 18) if is_ours else (8, -12)
        color = Theme.ACCENT_PRIMARY if is_ours else Theme.TEXT_SECONDARY
        weight = 'bold' if is_ours else 'normal'

        ax.annotate(
            row["Method"],
            (row["Latency"], row["Accuracy"]),
            xytext=offset, textcoords='offset points',
            fontsize=9, fontweight=weight, color=color,
            ha='center' if is_ours else 'left',
            path_effects=[path_effects.withStroke(linewidth=3, foreground=Theme.BG_CARD)]
        )

    # Ideal region indicator (top-left corner)
    ax.annotate(
        "IDEAL",
        xy=(0.02, 0.98), xycoords='axes fraction',
        fontsize=8, color=Theme.ACCENT_SUCCESS, alpha=0.6,
        ha='left', va='top', fontweight='bold'
    )
    ax.annotate(
        "",
        xy=(0.02, 0.95), xycoords='axes fraction',
        xytext=(0.15, 0.85), textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", color=Theme.ACCENT_SUCCESS, alpha=0.3, lw=1.5)
    )

    ax.set_xlabel("Latency (seconds) - lower is better", labelpad=10)
    ax.set_ylabel("Accuracy - higher is better", labelpad=10)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    wandb.log({"Evaluation/Performance_Scatter": wandb.Image(fig1)})
    plt.close(fig1)

    # ==========================================================================
    # CHART 2: Horizontal Bar Chart
    # ==========================================================================
    fig2, ax = plt.subplots(figsize=(9, 5))
    format_card(ax, title="Accuracy Comparison")

    # Sort by accuracy
    df_sorted = df.sort_values("Accuracy", ascending=True)

    # Create bars with conditional coloring
    colors = [Theme.ACCENT_PRIMARY if t == "ours" else Theme.TEXT_MUTED
              for t in df_sorted["Type"]]

    bars = ax.barh(
        df_sorted["Method"], df_sorted["Accuracy"],
        color=colors, edgecolor=Theme.BG_ELEVATED, linewidth=1,
        height=0.6, alpha=0.9
    )

    # Add value labels
    for bar, acc in zip(bars, df_sorted["Accuracy"]):
        width = bar.get_width()
        is_ours = width == ours["Accuracy"].values[0]

        # Label inside or outside based on bar width
        if width > 0.15:
            ax.text(
                width - 0.02, bar.get_y() + bar.get_height()/2,
                f"{acc:.1%}",
                ha='right', va='center',
                fontsize=11, fontweight='bold',
                color='white' if is_ours else Theme.BG_CARD
            )
        else:
            ax.text(
                width + 0.01, bar.get_y() + bar.get_height()/2,
                f"{acc:.1%}",
                ha='left', va='center',
                fontsize=11, fontweight='bold',
                color=Theme.TEXT_PRIMARY
            )

    # Add highlight for our method
    for bar, method_type in zip(bars, df_sorted["Type"]):
        if method_type == "ours":
            # Glow effect
            ax.barh(
                bar.get_y() + bar.get_height()/2,
                bar.get_width(),
                height=bar.get_height() * 1.3,
                color=Theme.ACCENT_PRIMARY, alpha=0.15,
                zorder=0
            )

    ax.set_xlabel("Accuracy", labelpad=10)
    ax.set_xlim(0, max(df["Accuracy"]) * 1.15)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    wandb.log({"Evaluation/Accuracy_Bars": wandb.Image(fig2)})
    plt.close(fig2)

    print("[Viz] Performance charts logged.")


@safe_viz
def log_gate_dynamics(engine):
    """Visualizes learned gates."""
    print("[Viz] Generating gate dynamics...")
    apply_theme()

    bridge = engine.bridge
    k_gates = [torch.sigmoid(layer.gate).detach().cpu().item()
               for layer in bridge.key_modifiers]
    v_gates = [torch.sigmoid(layer.gate).detach().cpu().item()
               for layer in bridge.value_modifiers]
    layers = np.arange(len(k_gates))

    fig, ax = plt.subplots(figsize=(11, 5))
    format_card(ax, title="Learned Gate Strengths by Layer",
                 subtitle="0 = no contribution, 1 = full contribution from bridge")

    # Calculate dynamic y-axis limits based on actual values
    all_gates = k_gates + v_gates
    gate_min = min(all_gates)
    gate_max = max(all_gates)
    gate_range = gate_max - gate_min
    
    # Add padding (15% of range or minimum 0.05)
    padding = max(gate_range * 0.15, 0.05)
    y_min = max(0, gate_min - padding)  # Don't go below 0
    y_max = min(1, gate_max + padding)  # Don't go above 1
    
    # If range is very small, ensure minimum visible range of 0.1
    if y_max - y_min < 0.1:
        center = (y_min + y_max) / 2
        y_min = max(0, center - 0.05)
        y_max = min(1, center + 0.05)

    # Reference line at midpoint of visible range (instead of always 0.5)
    mid_point = (y_min + y_max) / 2
    ax.axhline(mid_point, color=Theme.TEXT_MUTED, linewidth=1, linestyle='--', alpha=0.4, zorder=1)

    # Key gates - with fill
    line_k, = ax.plot(layers, k_gates,
                      marker='o', markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                      linestyle='-', linewidth=2.5,
                      color=Theme.ACCENT_PRIMARY, label='Key Modifier', zorder=3)
    ax.fill_between(layers, k_gates, y_min,
                    color=Theme.ACCENT_PRIMARY, alpha=0.15, zorder=2)

    # Value gates - with fill
    line_v, = ax.plot(layers, v_gates,
                      marker='s', markersize=7, markeredgecolor='white', markeredgewidth=1.5,
                      linestyle='--', linewidth=2.5,
                      color=Theme.ACCENT_WARNING, label='Value Modifier', zorder=3)
    ax.fill_between(layers, v_gates, y_min,
                    color=Theme.ACCENT_WARNING, alpha=0.1, zorder=2)

    # Styling with dynamic limits
    ax.set_xlabel("Layer Index", labelpad=10)
    ax.set_ylabel("Gate Strength", labelpad=10)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(layers)

    # Add region labels if range permits (only if we're showing a wide enough range)
    if y_max - y_min > 0.3:
        ax.text(0.98, 0.95, "HIGH CONTRIBUTION", transform=ax.transAxes,
                fontsize=8, color=Theme.ACCENT_SUCCESS, alpha=0.5,
                ha='right', va='top', fontweight='bold')
        ax.text(0.98, 0.05, "LOW CONTRIBUTION", transform=ax.transAxes,
                fontsize=8, color=Theme.ACCENT_WARNING, alpha=0.5,
                ha='right', va='bottom', fontweight='bold')

    # Statistics box (enhancement)
    k_arr, v_arr = np.array(k_gates), np.array(v_gates)
    stats_text = (
        f"Key:   mean={k_arr.mean():.3f}  std={k_arr.std():.3f}  range=[{k_arr.min():.2f}, {k_arr.max():.2f}]\n"
        f"Value: mean={v_arr.mean():.3f}  std={v_arr.std():.3f}  range=[{v_arr.min():.2f}, {v_arr.max():.2f}]\n"
        f"Y-axis: [{y_min:.2f}, {y_max:.2f}] (auto-scaled for clarity)"
    )
    props = dict(boxstyle='round,pad=0.4', facecolor=Theme.BG_ELEVATED,
                 edgecolor=Theme.BORDER, alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, fontfamily='monospace', color=Theme.TEXT_SECONDARY,
            va='top', ha='left', bbox=props)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)

    plt.tight_layout()
    wandb.log({"Interpretability/Gate_Dynamics": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Gate dynamics logged.")



@safe_viz
def log_bridge_heatmap(engine, prompt_text="Explain quantum entanglement."):
    """Visualizes injection heatmap."""
    print(f"[Viz] Generating injection heatmap for: '{prompt_text[:40]}...'")
    apply_theme()

    model = engine.bridge
    model.eval()

    k_norms, v_norms = [], []

    def make_hook(storage):
        def hook(module, inp, out):
            injection = out - inp[0]
            norm = injection.norm(p=2, dim=-1).mean().item()
            storage.append(norm)
        return hook

    handles = []
    for layer in model.key_modifiers:
        handles.append(layer.register_forward_hook(make_hook(k_norms)))
    for layer in model.value_modifiers:
        handles.append(layer.register_forward_hook(make_hook(v_norms)))

    with torch.no_grad():
        engine.evaluator.generate_demo(prompt_text, max_new_tokens=1)

    for h in handles:
        h.remove()

    # Build heatmap data
    data = np.array([k_norms, v_norms])

    # Larger figure for better readability
    num_layers = len(k_norms)
    fig_width = max(14, num_layers * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    # Custom colormap: dark to cyan
    cmap = create_custom_cmap(Theme.GRADIENT_VIRIDIS_DARK, "injection")

    # Heatmap - remove square constraint for larger cells
    im = ax.imshow(data, aspect='auto', cmap=cmap)

    # Find max injection layers
    max_k_layer = np.argmax(k_norms)
    max_v_layer = np.argmax(v_norms)

    # Annotations with max layer highlighting
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Determine text color based on cell brightness
            text_color = Theme.BG_DARK if val > data.max() * 0.6 else Theme.TEXT_PRIMARY

            # Check if this is a max layer
            is_max = (i == 0 and j == max_k_layer) or (i == 1 and j == max_v_layer)
            fontweight = 'bold' if is_max else 'medium'
            fontsize = 10 if is_max else 9

            txt = ax.text(j, i, f"{val:.2f}",
                          ha='center', va='center',
                          fontsize=fontsize, fontweight=fontweight,
                          color=text_color)

            # Add star marker for max values
            if is_max:
                ax.plot(j, i - 0.35, marker='v', markersize=6,
                        color=Theme.ACCENT_CYAN, markeredgecolor='white', markeredgewidth=0.5)

    # Axes
    ax.set_xticks(np.arange(len(k_norms)))
    ax.set_xticklabels([str(i) for i in range(len(k_norms))])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Key", "Value"])
    ax.set_xlabel("Layer Index", labelpad=10)

    # Title with prompt preview
    short_prompt = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
    ax.set_title(f"Injection Magnitude (L2 Norm)\n\"{short_prompt}\"",
                 fontsize=12, fontweight='bold', pad=15, loc='left',
                 color=Theme.TEXT_PRIMARY)

    # Max layer annotation
    ax.text(0.99, 0.98, f"Peak: K@L{max_k_layer}, V@L{max_v_layer}",
            transform=ax.transAxes, fontsize=8, color=Theme.ACCENT_CYAN,
            ha='right', va='top', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=15)
    cbar.set_label("L2 Norm", fontsize=10, color=Theme.TEXT_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=Theme.TEXT_SECONDARY)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=Theme.TEXT_SECONDARY)

    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    wandb.log({"Interpretability/Injection_Heatmap": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Injection heatmap logged.")


@safe_viz
def log_probability_shift(engine, prompt_text, true_label="A"):
    """Visualizes probability shift."""
    print(f"[Viz] Generating probability shift chart...")
    apply_theme()

    # --- Compute probabilities ---
    full_prompt = f"Carefully read the question and all options.\nRespond with only the letter of the correct answer (A, B, C, or D).\n{prompt_text}"
    tok = engine.factory.tok_receiver
    sharer_tok = engine.factory.tok_sharer
    candidates = ["A", "B", "C", "D"]
    candidate_ids = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in candidates]

    s_ids = sharer_tok.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        return_tensors="pt", add_generation_prompt=True
    ).to(engine.factory.device)[:, :-1]

    r_ids = tok.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        return_tensors="pt", add_generation_prompt=True
    ).to(engine.factory.device)
    rec_context = r_ids[:, :-1]

    def get_probs(logits):
        target_logits = logits[0, -1, candidate_ids]
        return F.softmax(target_logits, dim=0).float().cpu().numpy()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_base = engine.receiver(r_ids)
        probs_base = get_probs(out_base.logits)

        mod_cache = engine.evaluator.get_bridged_cache(s_ids, torch.ones_like(s_ids), rec_context)
        out_bridge = engine.receiver(r_ids, past_key_values=mod_cache)
        probs_bridge = get_probs(out_bridge.logits)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(11, 6))
    format_card(ax, title="Answer Probability Distribution",
                 subtitle="How the bridge shifts model confidence")

    x = np.arange(len(candidates))
    width = 0.30  # Slightly narrower bars to give more space

    # Baseline bars
    bars1 = ax.bar(x - width/2, probs_base, width,
                   label='Receiver Only',
                   color=Theme.TEXT_MUTED, alpha=0.7,
                   edgecolor=Theme.BG_ELEVATED, linewidth=1)

    # Bridge bars
    bars2 = ax.bar(x + width/2, probs_bridge, width,
                   label='With H2C Bridge',
                   color=Theme.ACCENT_PRIMARY, alpha=0.9,
                   edgecolor=Theme.BG_ELEVATED, linewidth=1)

    # Delta indicators between bars - positioned to not overlap with bar labels
    for i, (base, bridge) in enumerate(zip(probs_base, probs_bridge)):
        delta = bridge - base
        if abs(delta) > 0.03:  # Only show significant changes
            arrow_color = Theme.ACCENT_SUCCESS if delta > 0 else Theme.ACCENT_DANGER

            # Position delta label to the right of the bar pair, avoiding overlap
            label_x = i + width/2 + 0.12
            label_y = (base + bridge) / 2  # Midpoint between bars

            # Draw a small connecting line from bridge bar to delta label
            ax.plot([i + width/2 + 0.02, label_x - 0.02],
                    [bridge, label_y],
                    color=arrow_color, linewidth=1.5, alpha=0.6)

            # Delta label with arrow indicator
            delta_text = f"{delta:+.0%}"
            ax.text(label_x, label_y, delta_text,
                    ha='left', va='center', fontsize=9, fontweight='bold',
                    color=arrow_color,
                    path_effects=[path_effects.withStroke(linewidth=2, foreground=Theme.BG_CARD)])

    # Highlight correct answer
    if true_label in candidates:
        idx = candidates.index(true_label)
        # Background highlight
        ax.axvspan(idx - 0.48, idx + 0.48,
                   color=Theme.ACCENT_SUCCESS, alpha=0.08, zorder=0)
        # Border on top and bottom
        ax.axhline(y=0, xmin=(idx + 0.02)/4, xmax=(idx + 0.98)/4,
                   color=Theme.ACCENT_SUCCESS, linewidth=3, alpha=0.5)
        # Label
        ax.text(idx, 1.02, "[CORRECT]",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color=Theme.ACCENT_SUCCESS)

    # Value labels on bars
    def label_bars(bars, is_bridge=False):
        for bar in bars:
            height = bar.get_height()
            if height > 0.03:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.0%}',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color=Theme.ACCENT_PRIMARY if is_bridge else Theme.TEXT_SECONDARY)

    label_bars(bars1, False)
    label_bars(bars2, True)

    ax.set_ylabel("Probability", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(candidates, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, len(candidates) - 0.2)  # Room for delta labels on right
    ax.yaxis.grid(True, alpha=0.2)
    ax.xaxis.grid(False)

    ax.legend(loc='upper left')

    plt.tight_layout()
    wandb.log({"Interpretability/Probability_Shift": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Probability shift logged.")


@safe_viz
def log_qualitative_table(engine):
    """Generates qualitative comparison table."""
    print("[Viz] Generating qualitative comparison table...")

    # Diverse prompts to showcase different capabilities
    prompts = [
        # --- Factual Knowledge ---
        "What is the capital of Australia?",
        "Who wrote 'Pride and Prejudice'?",
        "What is the speed of light in meters per second?",
        "Name the largest planet in our solar system.",

        # --- Reasoning & Explanation ---
        "Explain 'Schrodinger's Cat' in one sentence.",
        "Why do we see lightning before we hear thunder?",
        "What causes the seasons on Earth?",
        "Explain the difference between a virus and a bacterium.",

        # --- Logic & Problem Solving ---
        "Riddle: I have no body, but I come alive with wind. What am I?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",

        # --- Math ---
        "What is 17 * 24?",
        "What is the derivative of x^3 + 2x?",
        "If a train travels 120 miles in 2 hours, what is its average speed?",

        # --- Coding ---
        "Write a Python one-liner to reverse a string.",
        "What does the 'finally' block do in Python exception handling?",

        # --- Creative ---
        "Write a haiku about machine learning.",
        "Complete this sentence creatively: 'The robot looked at the sunset and felt...'",

        # --- Multiple Choice (MMLU-style) ---
        "What is the atomic number of carbon?\nA) 6\nB) 12\nC) 14\nD) 8\nAnswer:",

        "Which of the following is NOT a programming paradigm?\nA) Object-oriented\nB) Functional\nC) Recursive\nD) Procedural\nAnswer:",

        "The Treaty of Westphalia (1648) is most associated with:\nA) The end of World War I\nB) The establishment of the United Nations\nC) The emergence of the modern nation-state system\nD) The partition of Africa\nAnswer:",

        "In economics, 'opportunity cost' refers to:\nA) The monetary cost of a decision\nB) The value of the next best alternative foregone\nC) The cost of missed opportunities in the stock market\nD) The price of goods in a competitive market\nAnswer:",

        "Which logical fallacy involves attacking the person making an argument rather than the argument itself?\nA) Straw man\nB) Ad hominem\nC) False dichotomy\nD) Slippery slope\nAnswer:",

        "The Pythagorean theorem states that for a right triangle:\nA) a + b = c\nB) a^2 + b^2 = c^2\nC) a * b = c\nD) a^2 - b^2 = c^2\nAnswer:",
    ]

    table = wandb.Table(columns=[
        "Prompt", "Receiver Only", "Sharer Only", "Text-to-Text", "H2C Bridge"
    ])

    engine.bridge.eval()
    engine.receiver.eval()
    engine.sharer.eval()
    tok_rec = engine.factory.tok_receiver
    tok_sha = engine.factory.tok_sharer
    device = engine.factory.device

    max_tokens = 60  # Enough for most answers

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")

        # --- Receiver Only ---
        r_input = tok_rec.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt", add_generation_prompt=True
        ).to(device)

        with torch.no_grad():
            out_rec = engine.receiver.generate(
                r_input, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tok_rec.pad_token_id
            )
        text_rec = tok_rec.decode(
            out_rec[0, r_input.shape[1]:], skip_special_tokens=False
        ).strip()

        # --- Sharer Only ---
        s_input = tok_sha.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt", add_generation_prompt=True
        ).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_sha = engine.sharer.generate(
                s_input, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tok_sha.pad_token_id
            )
        text_sha = tok_sha.decode(
            out_sha[0, s_input.shape[1]:], skip_special_tokens=False
        ).strip()

        # --- Text-to-Text (Sharer generates, Receiver continues) ---
        # Get sharer's output as text, then feed to receiver
        sharer_response = text_sha[:200]  # Limit length
        t2t_prompt = f"{prompt}\n\nContext from another model: {sharer_response}\n\nYour answer:"
        t2t_input = tok_rec.apply_chat_template(
            [{"role": "user", "content": t2t_prompt}],
            return_tensors="pt", add_generation_prompt=True
        ).to(device)

        with torch.no_grad():
            out_t2t = engine.receiver.generate(
                t2t_input, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tok_rec.pad_token_id
            )
        text_t2t = tok_rec.decode(
            out_t2t[0, t2t_input.shape[1]:], skip_special_tokens=False
        ).strip()

        # --- H2C Bridge ---
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cache = engine.evaluator.get_bridged_cache(
                s_input[:, :-1],
                torch.ones_like(s_input[:, :-1]),
                r_input[:, :-1]
            )

            out_bridge = engine.receiver.generate(
                r_input, past_key_values=cache,
                max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tok_rec.pad_token_id
            )
        text_bridge = tok_rec.decode(
            out_bridge[0, r_input.shape[1]:], skip_special_tokens=False
        ).strip()

        # Truncate long responses for table readability
        def truncate(s, max_len=300):
            return s[:max_len] + "..." if len(s) > max_len else s

        table.add_data(
            prompt,
            truncate(text_rec),
            truncate(text_sha),
            truncate(text_t2t),
            truncate(text_bridge)
        )

    wandb.log({"Evaluation/Qualitative_Comparisons": table})
    print("[Viz] Qualitative table logged with all baselines.")


@safe_viz
def log_training_summary(engine, config, eval_cache=None, baseline_results=None):
    """Generates training summary panel.

    Args:
        eval_cache: Optional cached metrics
        baseline_results: Optional baseline metrics
    """
    print("[Viz] Generating training summary...")
    apply_theme()

    # Collect data - use cache if available
    if eval_cache:
        h2c_acc, h2c_err, h2c_lat = eval_cache['acc'], eval_cache['err'], eval_cache['lat']
    else:
        h2c_acc, h2c_err, h2c_lat = engine.mmlu_evaluator.evaluate_accuracy(engine.mmlu_loader)

    # Prefer freshly calculated baseline_results - compute if not provided
    if baseline_results:
        baselines = {name: {"acc": v["acc"], "latency_s": v["latency_s"]}
                     for name, v in baseline_results.items()}
    else:
        # Compute baselines fresh (don't use precomputed config values)
        print("[Viz] Computing baselines fresh for summary...")
        fresh_baselines = engine.mmlu_evaluator.evaluate_baselines(engine.mmlu_loader)
        baselines = {name: {"acc": v["acc"], "latency_s": v["latency_s"]}
                     for name, v in fresh_baselines.items()}

    # Create figure with subplots - increased spacing to prevent overlap
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    fig.patch.set_facecolor(Theme.BG_DARK)

    # --- Panel 1: Accuracy Gauge (top-left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(Theme.BG_CARD)

    # Simple radial gauge effect using a partial circle
    theta = np.linspace(0.75 * np.pi, 0.25 * np.pi, 100)
    r = 1
    x_arc = r * np.cos(theta)
    y_arc = r * np.sin(theta)

    # Background arc
    ax1.plot(x_arc, y_arc, color=Theme.TEXT_MUTED, linewidth=15, alpha=0.3)

    # Filled arc based on accuracy
    fill_idx = int(len(theta) * h2c_acc)
    ax1.plot(x_arc[:fill_idx], y_arc[:fill_idx],
             color=Theme.ACCENT_PRIMARY, linewidth=15, solid_capstyle='round')

    # Center text
    ax1.text(0, 0.15, f"{h2c_acc:.1%}", ha='center', va='center',
             fontsize=26, fontweight='bold', color=Theme.TEXT_PRIMARY)
    ax1.text(0, -0.15, "Accuracy", ha='center', va='center',
             fontsize=10, color=Theme.TEXT_SECONDARY)

    # Trend indicator vs best baseline - positioned below the gauge
    if baselines:
        best_baseline_acc = max(v["acc"] for v in baselines.values())
        delta_vs_best = h2c_acc - best_baseline_acc
        trend_color = Theme.ACCENT_SUCCESS if delta_vs_best >= 0 else Theme.ACCENT_DANGER
        trend_arrow = "+" if delta_vs_best >= 0 else ""
        ax1.text(0, -0.45, f"{trend_arrow}{delta_vs_best:.1%} vs best baseline",
                 ha='center', va='center',
                 fontsize=9, color=trend_color, fontweight='bold')

    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-0.7, 1.4)
    ax1.axis('off')
    ax1.set_title("H2C Bridge Performance", fontsize=12, fontweight='bold',
                  color=Theme.TEXT_PRIMARY, pad=10)

    # --- Panel 2: Latency Comparison (top-middle) ---
    ax2 = fig.add_subplot(gs[0, 1])
    format_card(ax2, title="Latency (seconds)")

    methods = ["H2C Bridge"] + [k.replace("_", " ").title() for k in baselines.keys()]
    latencies = [h2c_lat]
    for v in baselines.values():
        lat = v.get("latency_s", 0)
        latencies.append(lat)

    colors = [Theme.ACCENT_PRIMARY] + [Theme.TEXT_MUTED] * len(baselines)

    bars = ax2.barh(methods, latencies, color=colors, height=0.5, alpha=0.8)
    ax2.set_xlabel("Seconds", labelpad=5)
    ax2.xaxis.grid(True, alpha=0.2)
    ax2.yaxis.grid(False)
    ax2.spines['left'].set_visible(False)

    # --- Panel 3: Delta vs Baselines (top-right) ---
    ax3 = fig.add_subplot(gs[0, 2])
    format_card(ax3, title="Accuracy Delta vs Baselines")

    baseline_names = [k.replace("_", " ").title() for k in baselines.keys()]
    deltas = [h2c_acc - v["acc"] for v in baselines.values()]

    bar_colors = [Theme.ACCENT_SUCCESS if d >= 0 else Theme.ACCENT_DANGER for d in deltas]

    bars = ax3.barh(baseline_names, deltas, color=bar_colors, height=0.5, alpha=0.8)
    ax3.axvline(0, color=Theme.TEXT_MUTED, linewidth=1, linestyle='--')
    ax3.set_xlabel("Delta (percentage points)", labelpad=5)
    ax3.xaxis.grid(True, alpha=0.2)
    ax3.yaxis.grid(False)
    ax3.spines['left'].set_visible(False)

    # Value labels
    for bar, delta in zip(bars, deltas):
        xpos = bar.get_width() + 0.005 if delta >= 0 else bar.get_width() - 0.005
        ha = 'left' if delta >= 0 else 'right'
        ax3.text(xpos, bar.get_y() + bar.get_height()/2,
                 f"{delta:+.1%}", ha=ha, va='center',
                 fontsize=9, fontweight='bold',
                 color=Theme.ACCENT_SUCCESS if delta >= 0 else Theme.ACCENT_DANGER)

    # --- Panel 4: Gate Distribution (bottom, spanning 2 columns) ---
    ax4 = fig.add_subplot(gs[1, :2])
    format_card(ax4, title="Gate Value Distribution")

    k_gates = [layer.gate.clamp(0.0, 1.0).detach().cpu().item()
               for layer in engine.bridge.key_modifiers]
    v_gates = [layer.gate.clamp(0.0, 1.0).detach().cpu().item()
               for layer in engine.bridge.value_modifiers]

    all_gates = k_gates + v_gates
    labels = ["K"] * len(k_gates) + ["V"] * len(v_gates)

    # Violin plot
    parts = ax4.violinplot([k_gates, v_gates], positions=[0, 1],
                           showmeans=True, showmedians=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(Theme.ACCENT_PRIMARY if i == 0 else Theme.ACCENT_WARNING)
        pc.set_alpha(0.6)

    # Style the statistical lines (check keys exist for compatibility)
    for key in ['cmeans', 'cbars', 'cmins', 'cmaxs']:
        if key in parts:
            color = Theme.TEXT_PRIMARY if key == 'cmeans' else Theme.TEXT_MUTED
            parts[key].set_color(color)

    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["Key Gates", "Value Gates"])
    ax4.set_ylabel("Gate Strength (0-1)")
    ax4.axhline(0.5, color=Theme.TEXT_MUTED, linewidth=1, linestyle='--', alpha=0.5)
    ax4.set_ylim(-0.05, 1.1)
    ax4.yaxis.grid(True, alpha=0.2)

    # --- Panel 5: Method Legend / Info (bottom-right) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(Theme.BG_CARD)
    ax5.axis('off')

    info_text = f"""
    Model Configuration
    -------------------
    Sharer:   {config.get('SHARER_ID', 'N/A').split('/')[-1]}
    Receiver: {config.get('RECEIVER_ID', 'N/A').split('/')[-1]}

    Training
    --------
    Batch Size: {config.get('BATCH_SIZE', 'N/A')}
    Learning Rate: {config.get('lr', 'N/A')}

    Evaluation
    ----------
    MMLU Samples: {config.get('mmlu_sample_size', 'N/A')} per category
    """

    ax5.text(0.1, 0.9, info_text, transform=ax5.transAxes,
             fontsize=9, fontfamily='monospace',
             color=Theme.TEXT_SECONDARY, va='top',
             linespacing=1.5)

    ax5.set_title("Configuration", fontsize=12, fontweight='bold',
                  color=Theme.TEXT_PRIMARY, pad=10, loc='left')

    plt.tight_layout()
    wandb.log({"Evaluation/Training_Summary": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Training summary logged.")


def _compute_category_accuracies(detailed_results):
    """Helper to compute per-category accuracies from detailed results."""
    category_stats = {}
    for result in detailed_results:
        category = categorize_subject(result['subject'])
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        if result['correct']:
            category_stats[category]['correct'] += 1

    accuracies = {}
    for cat, stats in category_stats.items():
        accuracies[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    return accuracies, category_stats


@safe_viz
def log_category_breakdown(engine, config, detailed_results=None, baseline_results=None):
    """
    Generate per-category MMLU accuracy breakdown comparing methods.
    Shows grouped bars for H2C Bridge vs baselines across categories.

    Args:
        detailed_results: List of dicts from evaluate_accuracy_detailed() for H2C Bridge
        baseline_results: Dict from evaluate_baselines_detailed()
    """
    print("[Viz] Generating comparative category breakdown...")
    apply_theme()

    # Get detailed results if not provided
    if detailed_results is None:
        _, _, _, detailed_results = engine.mmlu_evaluator.evaluate_accuracy_detailed(engine.mmlu_loader)

    if not detailed_results:
        print("[Viz] Warning: No detailed results available for category breakdown")
        return

    # Build data for all methods - order matters for legend
    methods_data = {}

    # Add baselines first
    if baseline_results:
        if "receiver_only" in baseline_results and baseline_results["receiver_only"].get("details"):
            methods_data["Receiver Only"] = _compute_category_accuracies(
                baseline_results["receiver_only"]["details"]
            )[0]
        if "sharer_only" in baseline_results and baseline_results["sharer_only"].get("details"):
            methods_data["Sharer Only"] = _compute_category_accuracies(
                baseline_results["sharer_only"]["details"]
            )[0]
        if "text_to_text" in baseline_results and baseline_results["text_to_text"].get("details"):
            methods_data["Text-to-Text"] = _compute_category_accuracies(
                baseline_results["text_to_text"]["details"]
            )[0]

    # Add our method last (will be rightmost in grouped bars)
    methods_data["H2C Bridge (Ours)"] = _compute_category_accuracies(detailed_results)[0]

    # Get all categories and sort by H2C Bridge accuracy
    all_categories = list(methods_data["H2C Bridge (Ours)"].keys())
    all_categories.sort(key=lambda c: methods_data["H2C Bridge (Ours)"].get(c, 0), reverse=True)

    # Method colors
    method_colors = {
        "Receiver Only": Theme.TEXT_MUTED,
        "Sharer Only": Theme.ACCENT_PURPLE,
        "Text-to-Text": Theme.ACCENT_WARNING,
        "H2C Bridge (Ours)": Theme.ACCENT_PRIMARY,
    }

    # Plot - wider figure for more methods
    fig, ax = plt.subplots(figsize=(14, 7))
    format_card(ax, title="Accuracy by Subject Category",
                 subtitle="Comparing H2C Bridge vs All Baselines")

    num_methods = len(methods_data)
    num_categories = len(all_categories)
    bar_height = 0.8 / num_methods

    # Create grouped bars
    for idx, (method_name, cat_accs) in enumerate(methods_data.items()):
        positions = np.arange(num_categories) + idx * bar_height - (num_methods - 1) * bar_height / 2
        accuracies = [cat_accs.get(cat, 0) for cat in all_categories]

        bars = ax.barh(positions, accuracies,
                       height=bar_height * 0.9,
                       color=method_colors.get(method_name, Theme.TEXT_MUTED),
                       alpha=0.85 if "Ours" in method_name else 0.6,
                       label=method_name,
                       edgecolor=Theme.BG_ELEVATED)

    # Y-axis labels
    ax.set_yticks(np.arange(num_categories))
    ax.set_yticklabels(all_categories, fontsize=10)

    # Styling
    ax.set_xlabel("Accuracy", labelpad=10)
    ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # Legend
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    wandb.log({"Evaluation/Category_Breakdown_Comparison": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Comparative category breakdown logged.")


def _build_confusion_matrix(detailed_results):
    """Helper to build a confusion matrix from detailed results."""
    labels = ['A', 'B', 'C', 'D']
    label_to_idx = {l: i for i, l in enumerate(labels)}

    matrix = np.zeros((4, 4), dtype=int)
    invalid_count = 0

    for result in detailed_results:
        true_label = result['label']
        pred_label = result['pred']

        if true_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            if pred_label in label_to_idx:
                pred_idx = label_to_idx[pred_label]
                matrix[true_idx, pred_idx] += 1
            else:
                invalid_count += 1

    # Normalize for percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_pct = matrix / row_sums

    return matrix, matrix_pct, invalid_count


@safe_viz
def log_confusion_matrix(engine, detailed_results=None, baseline_results=None):
    """
    Generate confusion matrices comparing H2C Bridge vs baselines.
    Shows side-by-side comparison of prediction patterns.

    Args:
        detailed_results: List of dicts from evaluate_accuracy_detailed() for H2C Bridge
        baseline_results: Dict from evaluate_baselines_detailed() with all baselines
    """
    print("[Viz] Generating comparative confusion matrices...")
    apply_theme()

    # Get detailed results if not provided
    if detailed_results is None:
        _, _, _, detailed_results = engine.mmlu_evaluator.evaluate_accuracy_detailed(engine.mmlu_loader)

    if not detailed_results:
        print("[Viz] Warning: No detailed results available for confusion matrix")
        return

    # Determine how many matrices to show - order matters for display
    methods = {}

    # Add baselines first (in logical order)
    if baseline_results:
        if "receiver_only" in baseline_results and baseline_results["receiver_only"].get("details"):
            methods["Receiver Only"] = baseline_results["receiver_only"]["details"]
        if "sharer_only" in baseline_results and baseline_results["sharer_only"].get("details"):
            methods["Sharer Only"] = baseline_results["sharer_only"]["details"]
        if "text_to_text" in baseline_results and baseline_results["text_to_text"].get("details"):
            methods["Text-to-Text"] = baseline_results["text_to_text"]["details"]

    # Add our method last (rightmost, highlighted)
    methods["H2C Bridge (Ours)"] = detailed_results

    num_methods = len(methods)
    labels = ['A', 'B', 'C', 'D']

    # Create figure with GridSpec for precise control over colorbar placement
    fig_width = 4.5 * num_methods + 1.5  # Space for matrices + colorbar
    fig = plt.figure(figsize=(fig_width, 6))

    # GridSpec: matrices take up most space, colorbar gets a narrow column on right
    gs = fig.add_gridspec(1, num_methods + 1, width_ratios=[1]*num_methods + [0.08], wspace=0.25)

    axes = [fig.add_subplot(gs[0, i]) for i in range(num_methods)]
    cbar_ax = fig.add_subplot(gs[0, num_methods])

    fig.patch.set_facecolor(Theme.BG_DARK)

    # Custom colormap
    cmap = create_custom_cmap(Theme.GRADIENT_COOL, "confusion")

    for idx, (ax, (method_name, results)) in enumerate(zip(axes, methods.items())):
        matrix, matrix_pct, invalid_count = _build_confusion_matrix(results)

        # Calculate accuracy for title
        accuracy = np.trace(matrix) / matrix.sum() if matrix.sum() > 0 else 0

        im = ax.imshow(matrix_pct, cmap=cmap, vmin=0, vmax=1)

        # Annotations
        for i in range(4):
            for j in range(4):
                val = matrix_pct[i, j]
                count = matrix[i, j]
                text_color = Theme.BG_DARK if val > 0.5 else Theme.TEXT_PRIMARY

                # Highlight diagonal (correct predictions)
                if i == j:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=False, edgecolor=Theme.ACCENT_SUCCESS,
                                               linewidth=2))

                ax.text(j, i, f"{val:.0%}\n({count})",
                        ha='center', va='center',
                        fontsize=8, fontweight='bold' if i == j else 'normal',
                        color=text_color)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicted", labelpad=6, fontsize=9)
        if idx == 0:
            ax.set_ylabel("Actual", labelpad=6, fontsize=9)

        # Title with accuracy - highlight our method
        is_ours = "Ours" in method_name
        title_color = Theme.ACCENT_PRIMARY if is_ours else Theme.TEXT_PRIMARY
        ax.set_title(f"{method_name}\nAcc: {accuracy:.1%}",
                     fontsize=10, fontweight='bold', pad=8,
                     color=title_color)

        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add colorbar in its dedicated axes
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Proportion", fontsize=9, color=Theme.TEXT_SECONDARY)

    # Main title
    fig.suptitle("Confusion Matrix Comparison (Row-Normalized)",
                 fontsize=13, fontweight='bold', y=0.98, color=Theme.TEXT_PRIMARY)

    # Adjust margins
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.05, right=0.95)

    wandb.log({"Evaluation/Confusion_Matrix_Comparison": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Comparative confusion matrices logged.")


@safe_viz
def log_layer_radar(engine):
    """
    Radar/polar chart showing layer-wise injection importance.
    Provides a visually striking view of which layers contribute most.
    """
    print("[Viz] Generating layer radar chart...")
    apply_theme()

    bridge = engine.bridge
    k_gates = [layer.gate.clamp(0.0, 1.0).detach().cpu().item()
               for layer in bridge.key_modifiers]
    v_gates = [layer.gate.clamp(0.0, 1.0).detach().cpu().item()
               for layer in bridge.value_modifiers]

    num_layers = len(k_gates)
    if num_layers < 3:
        print("[Viz] Warning: Not enough layers for radar chart (need >= 3)")
        return

    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_layers, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    k_gates_closed = k_gates + k_gates[:1]
    v_gates_closed = v_gates + v_gates[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.set_facecolor(Theme.BG_CARD)
    fig.patch.set_facecolor(Theme.BG_DARK)

    # Key gates
    ax.plot(angles, k_gates_closed, 'o-', linewidth=2,
            color=Theme.ACCENT_PRIMARY, label='Key Modifier', markersize=6)
    ax.fill(angles, k_gates_closed, alpha=0.2, color=Theme.ACCENT_PRIMARY)

    # Value gates
    ax.plot(angles, v_gates_closed, 's--', linewidth=2,
            color=Theme.ACCENT_WARNING, label='Value Modifier', markersize=5)
    ax.fill(angles, v_gates_closed, alpha=0.15, color=Theme.ACCENT_WARNING)

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)],
                       color=Theme.TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'],
                       color=Theme.TEXT_MUTED, fontsize=8)
    ax.spines['polar'].set_color(Theme.BORDER)
    ax.grid(color=Theme.GRID, alpha=0.5)

    ax.set_title("Layer-wise Gate Strength",
                 fontsize=12, fontweight='bold', pad=30,
                 color=Theme.TEXT_PRIMARY)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))

    plt.tight_layout()
    wandb.log({"Interpretability/Layer_Radar": wandb.Image(fig)})
    plt.close(fig)

    print("[Viz] Layer radar chart logged.")


@safe_viz
def log_attention_flow(engine):
    """
    Sankey diagram showing information flow through the bridge.
    Requires plotly to be installed.
    """
    if not PLOTLY_AVAILABLE:
        print("[Viz] Skipping Sankey diagram - plotly not installed")
        return

    print("[Viz] Generating attention flow Sankey diagram...")

    bridge = engine.bridge
    k_gates = [torch.sigmoid(layer.gate).detach().cpu().item()
               for layer in bridge.key_modifiers]
    v_gates = [torch.sigmoid(layer.gate).detach().cpu().item()
               for layer in bridge.value_modifiers]
    num_layers = len(k_gates)

    # Build Sankey data
    # Nodes: Sharer -> Bridge Key/Value modifiers -> Receiver
    labels = (
        ["Sharer Hidden States"] +
        [f"Key Modifier (Layer {i})" for i in range(num_layers)] +
        [f"Value Modifier (Layer {i})" for i in range(num_layers)] +
        ["Receiver KV Cache"]
    )

    # Colors
    node_colors = (
        [Theme.ACCENT_PURPLE] +  # Sharer
        [Theme.ACCENT_PRIMARY] * num_layers +  # K modifiers
        [Theme.ACCENT_WARNING] * num_layers +  # V modifiers
        [Theme.ACCENT_SUCCESS]  # Receiver
    )

    # Links: Sharer -> each modifier, each modifier -> Receiver
    sources = []
    targets = []
    values = []
    link_colors = []

    sharer_idx = 0
    receiver_idx = 1 + 2 * num_layers

    for i in range(num_layers):
        k_mod_idx = 1 + i
        v_mod_idx = 1 + num_layers + i

        # Sharer -> K modifier
        k_strength = abs(k_gates[i]) + 0.1  # Add small offset for visibility
        sources.append(sharer_idx)
        targets.append(k_mod_idx)
        values.append(k_strength)
        link_colors.append(f"rgba(88, 166, 255, {0.3 + k_strength * 0.4})")

        # Sharer -> V modifier
        v_strength = abs(v_gates[i]) + 0.1
        sources.append(sharer_idx)
        targets.append(v_mod_idx)
        values.append(v_strength)
        link_colors.append(f"rgba(210, 153, 34, {0.3 + v_strength * 0.4})")

        # K modifier -> Receiver
        sources.append(k_mod_idx)
        targets.append(receiver_idx)
        values.append(k_strength)
        link_colors.append(f"rgba(88, 166, 255, {0.3 + k_strength * 0.4})")

        # V modifier -> Receiver
        sources.append(v_mod_idx)
        targets.append(receiver_idx)
        values.append(v_strength)
        link_colors.append(f"rgba(210, 153, 34, {0.3 + v_strength * 0.4})")

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color=Theme.BG_DARK, width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text="H2C Bridge Information Flow",
        title_font=dict(size=16, color=Theme.TEXT_PRIMARY),
        font=dict(size=10, color=Theme.TEXT_SECONDARY),
        paper_bgcolor=Theme.BG_DARK,
        plot_bgcolor=Theme.BG_CARD,
        height=500,
        width=900
    )

    # Log to WandB as HTML
    wandb.log({"Interpretability/Attention_Flow": wandb.Html(fig.to_html())})

    print("[Viz] Attention flow Sankey logged.")
