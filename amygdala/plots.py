"""README charts, drawn in a light and a dark version from the saved circuit results."""

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

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


REGION_COLORS = {
    "light": {"amygdala": "#eb6834", "cingulate": "#8a4fd6", "prefrontal": "#2a78d6"},
    "dark": {"amygdala": "#d95926", "cingulate": "#a77ff0", "prefrontal": "#3987e5"},
}


def bezier(points: list[tuple], close: bool = True) -> MplPath:
    """A path from a start point and groups of three cubic Bezier control points."""
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(points) - 1)
    if close:
        points, codes = [*points, points[0]], [*codes, MplPath.CLOSEPOLY]
    return MplPath(points, codes)


def plot_brain(out_dir: Path, mode: str) -> None:
    """A stylized middle view of the brain, each region linked to the module it inspired."""
    theme, colors = THEMES[mode], REGION_COLORS[mode]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    ax.set_xlim(0, 1000)
    ax.set_ylim(440, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ink, muted, grid = theme["text"], theme["muted"], theme["grid"]

    def shape(points, face, edge=None, alpha=1.0, z=1, close=True, width=1.6):
        ax.add_patch(
            PathPatch(
                bezier(points, close),
                facecolor=face if close else "none",
                edgecolor=edge or "none",
                alpha=alpha,
                lw=width,
                zorder=z,
            )
        )

    # The brain, seen from the middle, facing left.
    cerebrum = [
        (70, 250),
        (30, 185),
        (45, 95),
        (130, 68),
        (190, 42),
        (300, 36),
        (360, 62),
        (430, 92),
        (462, 165),
        (440, 228),
        (428, 262),
        (395, 276),
        (365, 272),
        (335, 300),
        (280, 302),
        (250, 288),
        (215, 304),
        (165, 302),
        (138, 284),
        (105, 292),
        (78, 276),
        (70, 250),
    ]
    shape(cerebrum, grid, edge=muted, z=1)
    shape(
        [
            (330, 290),
            (350, 262),
            (440, 262),
            (455, 300),
            (462, 335),
            (380, 345),
            (345, 322),
            (330, 310),
            (330, 300),
            (330, 290),
        ],
        grid,
        edge=muted,
        z=0,
    )  # cerebellum
    shape(
        [
            (292, 286),
            (298, 330),
            (304, 360),
            (306, 395),
            (322, 395),
            (338, 395),
            (336, 360),
            (330, 320),
            (326, 286),
            (310, 290),
            (300, 288),
            (292, 286),
        ],
        grid,
        edge=muted,
        z=0,
    )
    for fold in (
        [(120, 120), (160, 100), (200, 95), (230, 110)],
        [(250, 75), (290, 70), (330, 80), (360, 100)],
        [(380, 120), (410, 150), (420, 185), (410, 215)],
        [(260, 250), (300, 235), (340, 240), (370, 255)],
    ):
        shape(fold, "none", edge=muted, close=False, alpha=0.5, width=1.1, z=2)
    # Corpus callosum, under the cingulate.
    shape(
        [(165, 190), (205, 140), (305, 132), (345, 180)],
        "none",
        edge=muted,
        close=False,
        width=7,
        alpha=0.55,
        z=2,
    )

    # The three regions.
    shape(
        [
            (70, 250),
            (30, 185),
            (45, 95),
            (130, 68),
            (150, 62),
            (168, 60),
            (182, 62),
            (150, 115),
            (138, 180),
            (152, 250),
            (120, 262),
            (92, 262),
            (70, 250),
        ],
        colors["prefrontal"],
        alpha=0.45,
        z=3,
    )
    shape(
        [(150, 178), (195, 112), (318, 104), (358, 170)],
        "none",
        edge=colors["cingulate"],
        close=False,
        width=11,
        alpha=0.85,
        z=3,
    )
    ax.add_patch(Ellipse((222, 252), 34, 20, angle=-20, color=colors["amygdala"], zorder=4))

    # The model, as a column of layers.
    col_x, col_w = 800, 150
    blocks = [
        ("Layers 1 to 15", 40, 140, grid),
        ("Layers 16 to 20", 196, 64, grid),
        ("Layers 21 to 28", 276, 70, grid),
        ("Next token", 362, 40, grid),
    ]
    for label, y, h, face in blocks:
        ax.add_patch(
            FancyBboxPatch(
                (col_x, y),
                col_w,
                h,
                boxstyle="round,pad=0,rounding_size=8",
                facecolor=face,
                edgecolor=muted,
                lw=1.2,
                zorder=2,
            )
        )
        ax.text(
            col_x + col_w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            color=ink,
            fontsize=9,
            zorder=3,
        )
    for (_, y1, h1, _), (_, y2, _, _) in pairwise(blocks):
        ax.annotate(
            "",
            (col_x + col_w / 2, y2),
            (col_x + col_w / 2, y1 + h1),
            arrowprops={"arrowstyle": "-|>", "color": muted, "lw": 1.2},
        )
    ax.add_patch(
        FancyBboxPatch(
            (col_x - 4, 192),
            col_w + 8,
            72,
            boxstyle="round,pad=0,rounding_size=10",
            facecolor="none",
            edgecolor=colors["prefrontal"],
            lw=2.4,
            zorder=4,
        )
    )

    # Links from each region to its module, with the module's job.
    # Each link runs region -> label -> layer block, so no line crosses any text.
    label_x, label_end = 505, 735
    links = [
        (
            "cingulate",
            (300, 108),
            (col_x, 95),
            "Anterior cingulate",
            "The command starts now",
            "fires where the command begins",
            45,
        ),
        (
            "prefrontal",
            (100, 255),
            (col_x - 4, 236),
            "Prefrontal cortex",
            "Think again: inspect first",
            "push in layers 16 to 20, scaled by fear",
            300,
        ),
        (
            "amygdala",
            (240, 252),
            (col_x, 168),
            "Amygdala",
            "Is this task vague?",
            "reads layer 15 when the prompt ends",
            175,
        ),
    ]
    for key, start, end, name, job, detail, ty in links:
        color = colors[key]
        style = {
            "arrowstyle": "-",
            "color": color,
            "lw": 1.6,
            "linestyle": (0, (4, 3)),
            "shrinkA": 4,
            "shrinkB": 6,
        }
        ax.annotate("", (label_x - 4, ty + 26), start, arrowprops=style, zorder=5)
        ax.annotate(
            "",
            end,
            (label_end, ty + 26),
            arrowprops={**style, "arrowstyle": "-|>", "shrinkB": 2},
            zorder=5,
        )
        tx = label_x
        ax.text(tx, ty, name, color=color, fontsize=10.5, fontweight="bold", va="top", zorder=6)
        ax.text(tx, ty + 22, job, color=ink, fontsize=9, va="top", zorder=6)
        ax.text(tx, ty + 40, detail, color=muted, fontsize=8, va="top", zorder=6)

    ax.text(30, 425, "Brain regions (stylized)", color=muted, fontsize=8)
    ax.text(col_x, 425, "Qwen2.5-1.5B, frozen", color=muted, fontsize=8)
    fig.tight_layout(pad=0.4)
    save(fig, out_dir, "brain", mode)


def plot_all(results: dict, report: dict, out_dir: Path) -> None:
    for mode in THEMES:
        plot_brain(out_dir, mode)
        plot_fear_trace(results["trace"], out_dir, mode)
        plot_conditions(report, out_dir, mode)
