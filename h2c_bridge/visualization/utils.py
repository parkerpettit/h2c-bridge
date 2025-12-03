"""Visualization utilities."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

from h2c_bridge.visualization.theme import Theme


def create_custom_cmap(colors, name="custom"):
    """Creates custom colormap."""
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def add_glow(artist, color, alpha=0.3, linewidth=3):
    """Adds glow effect."""
    artist.set_path_effects([
        path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=color, alpha=alpha, linewidth=linewidth),
        path_effects.Normal()
    ])


def add_text_stroke(artist, stroke_color=None, linewidth=3):
    """Adds text outline."""
    if stroke_color is None:
        stroke_color = Theme.BG_DARK
    artist.set_path_effects([
        path_effects.withStroke(linewidth=linewidth, foreground=stroke_color),
    ])


def format_card(ax, title=None, subtitle=None):
    """Applies card styling."""
    ax.set_facecolor(Theme.BG_CARD)

    # Add subtle border effect by setting spine properties
    for spine in ax.spines.values():
        spine.set_color(Theme.BORDER)
        spine.set_linewidth(1)

    if title:
        # Increased pad from 15 to 22 to make room for subtitle
        ax.set_title(title, fontsize=14, fontweight='bold',
                     color=Theme.TEXT_PRIMARY, pad=22, loc='left')
    if subtitle:
        # Moved up from 1.02 to 1.01 (sits just below the title)
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=9, color=Theme.TEXT_SECONDARY, style='italic')


def add_caption(fig, text, fontsize=9):
    """Adds figure caption."""
    fig.text(0.5, 0.01, text, ha='center', fontsize=fontsize,
             color=Theme.TEXT_SECONDARY, style='italic', wrap=True)


# Subject category mappings for MMLU analysis
SUBJECT_CATEGORIES = {
    "STEM": [
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
        'high_school_statistics', 'machine_learning', 'anatomy', 'college_medicine',
        'clinical_knowledge', 'medical_genetics', 'professional_medicine', 'virology', 'nutrition'
    ],
    "Humanities": [
        'formal_logic', 'high_school_european_history', 'high_school_us_history',
        'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies',
        'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law',
        'world_religions'
    ],
    "Social Sciences": [
        'econometrics', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology',
        'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'human_aging'
    ],
    "Other": [
        'business_ethics', 'global_facts', 'management', 'marketing', 'miscellaneous',
        'professional_accounting'
    ]
}


def categorize_subject(subject):
    """Maps subject to category."""
    for category, subjects in SUBJECT_CATEGORIES.items():
        if subject in subjects:
            return category
    return "Other"


def safe_viz(func):
    """Decorator for error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Viz] Warning: {func.__name__} failed with error: {e}")
            return None
    return wrapper
