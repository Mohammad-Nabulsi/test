from pathlib import Path
import seaborn as sns


def set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")


def save_figure(fig, figures_dir, filename):
    d = Path(figures_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / filename
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out
