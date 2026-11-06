"""README charts, drawn in a light and a dark version from the saved circuit results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33332f",
        "series": ["#3987e5", "#d95926"],
    },
}

CONDITION_LABELS = {
    "base": "Base model",
    "amygdala": "Amygdala circuit",
    "lesion_amygdala": "Lesion: amygdala",
    "lesion_cingulate": "Lesion: cingulate",
    "action_gated": "Learned 'about to destroy' gate",
    "control_shuffled": "Control: shuffled-label sensors",
    "control_random": "Control: random prefrontal",
}


def styled_axes(theme: dict, ncols: int = 1, width: float = 7.0, height: float = 3.4):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False, sharey=True)
    fig.patch.set_facecolor(theme["surface"])
    for ax in axes[0]:
        ax.set_facecolor(theme["surface"])
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
        ax.tick_params(colors=theme["muted"], length=0, labelsize=8)
        ax.xaxis.label.set_color(theme["muted"])
        ax.yaxis.label.set_color(theme["muted"])
    return fig, axes[0]


def title(ax, theme: dict, text: str) -> None:
    ax.set_title(text, loc="left", color=theme["text"], fontsize=10, fontweight="bold", pad=10)


def save(fig, out_dir: Path, name: str, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-{mode}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path}")


def plot_fear_trace(trace: dict, out_dir: Path, mode: str) -> None:
    """Fear per token over the reply, from the last prompt tokens to the end."""
    theme = THEMES[mode]
    tokens, fear = trace["tokens"], trace["fear"]
    # The system prompt also contains <tool_call>, so start from the reply's, the last one.
    start = len(tokens) - 1 - tokens[::-1].index("<tool_call>") - 3
    tokens, fear = tokens[start:], fear[start:]
    xs = range(len(tokens))
    fig, (ax,) = styled_axes(theme, width=9.5, height=3.2)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    title(ax, theme, "Fear inside the model while it writes its tool call")
    ax.fill_between(xs, fear, color=theme["series"][1], alpha=0.15, linewidth=0)
    ax.plot(
        xs,
        fear,
        color=theme["series"][1],
        linewidth=2,
        marker="o",
        markersize=4,
        markeredgecolor=theme["surface"],
        markeredgewidth=1,
    )
    spike = next(i for i, f in enumerate(fear) if f > 0)
    ax.annotate(
        f"cingulate fires at the command start: fear = {fear[spike]:.2f}",
        (spike, fear[spike]),
        xytext=(-12, 8),
        textcoords="offset points",
        ha="right",
        color=theme["text"],
        fontsize=8,
    )
    ax.annotate(
        "fades 10% per token",
        (spike + 4, fear[spike + 4]),
        xytext=(10, 6),
        textcoords="offset points",
        ha="left",
        color=theme["muted"],
        fontsize=8,
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [repr(t)[1:-1] for t in tokens],
        rotation=60,
        ha="right",
        fontsize=7,
        family="monospace",
        color=theme["muted"],
    )
    ax.set_ylim(-0.02, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Fear")
    fig.tight_layout()
    save(fig, out_dir, "fear-trace", mode)


def plot_conditions(report: dict, out_dir: Path, mode: str) -> None:
    """Two panels of rates with 95% intervals, one row per condition."""
    theme = THEMES[mode]
    table = report["table"]
    conditions = [c for c in CONDITION_LABELS if c in table]
    panels = [
        ("ambiguous_destructive", "Destructive calls, 40 vague tasks (lower is better)"),
        ("explicit_delete", "Clear deletes done, 20 tasks (higher is better)"),
    ]
    fig, axes = styled_axes(theme, ncols=2, width=11, height=3.6)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        title(ax, theme, label)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
        for y, condition in enumerate(reversed(conditions)):
            cell = table[condition][metric]
            highlight = condition == "amygdala"
            color = theme["series"][1] if highlight else theme["series"][0]
            ax.plot(
                cell["ci95"],
                [y, y],
                color=color,
                linewidth=2,
                solid_capstyle="round",
                alpha=0.9 if highlight else 0.6,
            )
            ax.scatter(
                [cell["rate"]],
                [y],
                s=56 if highlight else 44,
                color=color,
                edgecolors=theme["surface"],
                linewidths=1.5,
                zorder=3,
            )
            ax.annotate(
                f"{cell['rate']:.0%}",
                (cell["rate"], y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=theme["text"],
            )
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(
            [CONDITION_LABELS[c] for c in reversed(conditions)], color=theme["text"], fontsize=9
        )
    fig.tight_layout(w_pad=3)
    save(fig, out_dir, "conditions", mode)


def plot_all(results: dict, report: dict, out_dir: Path) -> None:
    for mode in THEMES:
        plot_fear_trace(results["trace"], out_dir, mode)
        plot_conditions(report, out_dir, mode)
