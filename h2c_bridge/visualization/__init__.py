"""Visualization module."""

from h2c_bridge.visualization.theme import Theme, apply_theme
from h2c_bridge.visualization.runner import (
    run_all_visualizations,
    run_visualizations_light_only,
    run_visualizations_dark_only
)
from h2c_bridge.visualization.charts import (
    log_performance_charts,
    log_gate_dynamics,
    log_bridge_heatmap,
    log_probability_shift,
    log_category_breakdown,
    log_confusion_matrix,
    log_layer_radar,
    log_attention_flow,
    log_qualitative_table,
    log_training_summary
)

__all__ = [
    "Theme",
    "apply_theme",
    "run_all_visualizations",
    "run_visualizations_light_only",
    "run_visualizations_dark_only",
    "log_performance_charts",
    "log_gate_dynamics",
    "log_bridge_heatmap",
    "log_probability_shift",
    "log_category_breakdown",
    "log_confusion_matrix",
    "log_layer_radar",
    "log_attention_flow",
    "log_qualitative_table",
    "log_training_summary",
]
