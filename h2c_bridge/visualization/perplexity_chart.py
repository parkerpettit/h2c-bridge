
@safe_viz
def log_perplexity_comparison(engine, baseline_ppl_results=None):
    """Visualizes perplexity comparison across all methods.
    
    Shows how well each approach models conversational data.
    
    Args:
        baseline_ppl_results: Dict with perplexity results for baselines
    """
    print("[Viz] Generating perplexity comparison chart...")
    apply_theme()
    
    # Get bridge perplexity from validation loss
    val_loss = engine.evaluator.evaluate_loss(engine.val_loader)
    bridge_ppl = torch.exp(torch.tensor(val_loss)).item()
    
    # Prepare data
    methods = ["H2C Bridge"]
    perplexities = [bridge_ppl]
    colors = [Theme.ACCENT_PRIMARY]
    
    if baseline_ppl_results:
        for name, results in baseline_ppl_results.items():
            clean_name = name.replace("_", " ").title()
            methods.append(clean_name)
            perplexities.append(results['perplexity'])
            colors.append(Theme.TEXT_MUTED)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    format_card(ax, title="Perplexity Comparison (OpenHermes Val Set)",
                 subtitle="Lower is better - measures conversational quality")
    
    # Horizontal bars
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, perplexities, color=colors, height=0.6, alpha=0.9,
                   edgecolor=Theme.BG_ELEVATED, linewidth=1)
    
    # Add value labels
    for i, (bar, ppl) in enumerate(zip(bars, perplexities)):
        is_ours = (i =

 0)
        ax.text(bar.get_width() + max(perplexities) * 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{ppl:.2f}",
                ha='left', va='center',
                fontsize=11, fontweight='bold' if is_ours else 'normal',
                color=Theme.ACCENT_PRIMARY if is_ours else Theme.TEXT_SECONDARY)
    
    # Highlight our method with glow
    if len(bars) > 0:
        ax.barh(bars[0].get_y() + bars[0].get_height()/2,
                bars[0].get_width(),
                height=bars[0].get_height() * 1.3,
                color=Theme.ACCENT_PRIMARY, alpha=0.15,
                zorder=0)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Perplexity (lower = better at conversation)", labelpad=10)
    ax.set_xlim(0, max(perplexities) * 1.2)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    
    # Add "better" indicator
    ax.annotate("BETTER", xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=9, color=Theme.ACCENT_SUCCESS, alpha=0.6,
                ha='left', va='top', fontweight='bold')
    ax.annotate("←", xy=(0.12, 0.95), xycoords='axes fraction',
                fontsize=12, color=Theme.ACCENT_SUCCESS, alpha=0.6,
                ha='left', va='top')
    
    plt.tight_layout()
    wandb.log({"Evaluation/Perplexity_Comparison": wandb.Image(fig)})
    plt.close(fig)
    
    print("[Viz] Perplexity comparison logged.")
