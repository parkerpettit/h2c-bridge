"""Visualization themes."""

import matplotlib.pyplot as plt
import seaborn as sns


# Helper descriptor for class properties
class classproperty:
    """Class property descriptor."""
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, owner):
        return self.func(owner)


class ThemeDark:
    """Dark theme for presentations."""
    NAME = "dark"

    # Core palette - deep navy background with vibrant accents
    BG_DARK = "#0D1117"
    BG_CARD = "#161B22"
    BG_ELEVATED = "#21262D"

    # Accent colors - carefully chosen for contrast and harmony
    ACCENT_PRIMARY = "#58A6FF"    # Electric blue - our method
    ACCENT_SUCCESS = "#3FB950"    # Emerald green - correct/positive
    ACCENT_WARNING = "#D29922"    # Amber - caution
    ACCENT_DANGER = "#F85149"     # Coral red - error/baseline
    ACCENT_PURPLE = "#A371F7"     # Lavender - secondary accent
    ACCENT_CYAN = "#39D5FF"       # Bright cyan - highlights

    # Text colors
    TEXT_PRIMARY = "#E6EDF3"
    TEXT_SECONDARY = "#8B949E"
    TEXT_MUTED = "#484F58"

    # Grid and borders
    GRID = "#30363D"
    BORDER = "#30363D"

    # Method-specific colors
    COLORS = {
        "H2C Bridge (Ours)": "#58A6FF",
        "Receiver Only": "#8B949E",
        "Sharer Only": "#A371F7",
        "Text To Text": "#D29922",
    }

    # Gradient definitions for heatmaps
    GRADIENT_COOL = ["#0D1117", "#1A3A5C", "#2E6B8A", "#58A6FF"]
    GRADIENT_WARM = ["#0D1117", "#4A2545", "#8B3A62", "#F85149"]
    GRADIENT_VIRIDIS_DARK = ["#0D1117", "#1E4D5C", "#2A9D8F", "#39D5FF"]

    DPI = 300
    FONT_FAMILY = "sans-serif"


class ThemeLight:
    """Light theme for publications."""
    NAME = "light"

    # Core palette - clean white/light gray backgrounds
    BG_DARK = "#FFFFFF"
    BG_CARD = "#FAFBFC"
    BG_ELEVATED = "#F6F8FA"

    # Accent colors - slightly deeper for print readability
    ACCENT_PRIMARY = "#0969DA"    # Strong blue - our method
    ACCENT_SUCCESS = "#1A7F37"    # Forest green - correct/positive
    ACCENT_WARNING = "#9A6700"    # Dark amber - caution
    ACCENT_DANGER = "#CF222E"     # Deep red - error/baseline
    ACCENT_PURPLE = "#8250DF"     # Rich purple - secondary accent
    ACCENT_CYAN = "#0550AE"       # Deep cyan - highlights

    # Text colors
    TEXT_PRIMARY = "#1F2328"
    TEXT_SECONDARY = "#656D76"
    TEXT_MUTED = "#8C959F"

    # Grid and borders
    GRID = "#D0D7DE"
    BORDER = "#D0D7DE"

    # Method-specific colors (print-friendly)
    COLORS = {
        "H2C Bridge (Ours)": "#0969DA",
        "Receiver Only": "#656D76",
        "Sharer Only": "#8250DF",
        "Text To Text": "#9A6700",
    }

    # Gradient definitions for heatmaps (light backgrounds)
    GRADIENT_COOL = ["#FFFFFF", "#C8E1FF", "#54AEFF", "#0969DA"]
    GRADIENT_WARM = ["#FFFFFF", "#FFCDD2", "#EF9A9A", "#CF222E"]
    GRADIENT_VIRIDIS_DARK = ["#FFFFFF", "#B2DFDB", "#4DB6AC", "#00796B"]

    DPI = 300
    FONT_FAMILY = "sans-serif"


class Theme:
    """Dynamic theme proxy."""
    _current = ThemeDark  # Default to dark theme

    @classmethod
    def set_theme(cls, theme_name):
        """Switch between 'dark' and 'light' themes."""
        if theme_name == "dark":
            cls._current = ThemeDark
        elif theme_name == "light":
            cls._current = ThemeLight
        else:
            raise ValueError(f"Unknown theme: {theme_name}. Use 'dark' or 'light'.")
        apply_theme()
        print(f"[Viz] Theme switched to: {theme_name}")

    @classmethod
    def get_theme_name(cls):
        return cls._current.NAME

    # Delegate all attributes to current theme
    def __class_getitem__(cls, key):
        return getattr(cls._current, key)

    # Properties that delegate to current theme
    @classproperty
    def NAME(cls): return cls._current.NAME
    @classproperty
    def BG_DARK(cls): return cls._current.BG_DARK
    @classproperty
    def BG_CARD(cls): return cls._current.BG_CARD
    @classproperty
    def BG_ELEVATED(cls): return cls._current.BG_ELEVATED
    @classproperty
    def ACCENT_PRIMARY(cls): return cls._current.ACCENT_PRIMARY
    @classproperty
    def ACCENT_SUCCESS(cls): return cls._current.ACCENT_SUCCESS
    @classproperty
    def ACCENT_WARNING(cls): return cls._current.ACCENT_WARNING
    @classproperty
    def ACCENT_DANGER(cls): return cls._current.ACCENT_DANGER
    @classproperty
    def ACCENT_PURPLE(cls): return cls._current.ACCENT_PURPLE
    @classproperty
    def ACCENT_CYAN(cls): return cls._current.ACCENT_CYAN
    @classproperty
    def TEXT_PRIMARY(cls): return cls._current.TEXT_PRIMARY
    @classproperty
    def TEXT_SECONDARY(cls): return cls._current.TEXT_SECONDARY
    @classproperty
    def TEXT_MUTED(cls): return cls._current.TEXT_MUTED
    @classproperty
    def GRID(cls): return cls._current.GRID
    @classproperty
    def BORDER(cls): return cls._current.BORDER
    @classproperty
    def COLORS(cls): return cls._current.COLORS
    @classproperty
    def GRADIENT_COOL(cls): return cls._current.GRADIENT_COOL
    @classproperty
    def GRADIENT_WARM(cls): return cls._current.GRADIENT_WARM
    @classproperty
    def GRADIENT_VIRIDIS_DARK(cls): return cls._current.GRADIENT_VIRIDIS_DARK
    @classproperty
    def DPI(cls): return cls._current.DPI
    @classproperty
    def FONT_FAMILY(cls): return cls._current.FONT_FAMILY


def apply_theme():
    """Apply the current theme to matplotlib globally."""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': Theme.BG_DARK,
        'figure.edgecolor': Theme.BG_DARK,
        'figure.dpi': Theme.DPI,

        # Axes
        'axes.facecolor': Theme.BG_CARD,
        'axes.edgecolor': Theme.BORDER,
        'axes.labelcolor': Theme.TEXT_PRIMARY,
        'axes.titlecolor': Theme.TEXT_PRIMARY,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'axes.labelweight': 'medium',

        # Grid
        'axes.grid': True,
        'grid.color': Theme.GRID,
        'grid.alpha': 0.5,
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',

        # Ticks
        'xtick.color': Theme.TEXT_SECONDARY,
        'ytick.color': Theme.TEXT_SECONDARY,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,

        # Text
        'text.color': Theme.TEXT_PRIMARY,
        'font.family': Theme.FONT_FAMILY,

        # Legend
        'legend.facecolor': Theme.BG_ELEVATED,
        'legend.edgecolor': Theme.BORDER,
        'legend.labelcolor': Theme.TEXT_PRIMARY,
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,

        # Save
        'savefig.facecolor': Theme.BG_DARK,
        'savefig.edgecolor': Theme.BG_DARK,
    })

    # Seaborn context
    sns.set_context("notebook", font_scale=1.1)
