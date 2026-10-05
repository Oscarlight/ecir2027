#!/usr/bin/env python3
"""Figure 1, compact: NE delta against Wait, one row, both attribution windows as line styles.

    python3 make_fig.py              # -> figs/delta.pdf
    python3 make_fig_2x3.py          # the 2x3 variant, same data, ~0.35pp taller

Colour is the target, dash is the attribution window. Folding the windows into one row costs
roughly a third of a page against the 2x3 grid and loses nothing: the comparison the reader
makes is 7d-against-1d within a target, which adjacent lines show better than stacked panels.
ULC and the frozen teacher are reported numerically in the text rather than plotted; they lie
on top of Correct-Distill and only thicken the panel.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

HERE = Path(__file__).resolve().parent
CAPS = ["A0", "A1", "A2", "A3", "A4"]
PARAMS = {"A0": 1_890, "A1": 9_810, "A2": 94_959, "A3": 1_184_439, "A4": 9_580_477}
O_MIN = [5, 30, 90]
INK, MUTED, LINE = "#222222", "#6b7a7d", "#c9d4cf"
TEAL, CORAL = "#12807a", "#e0674f"
BASE = "DISTILL_T"

#: (arm, window) -> (label, colour, dash, marker)
SERIES = [
    ("DISTILL_Tmlp",  "7d", "Correct-Distill, $v{=}7$d", TEAL,  "-",  "o"),
    ("DISTILL_Tmlp",  "1d", "Correct-Distill, $v{=}1$d", TEAL,  "--", "o"),
    ("DISTILL_pvmlp", "7d", "Distill-Only, $v{=}7$d",    CORAL, "-",  "s"),
    ("DISTILL_pvmlp", "1d", "Distill-Only, $v{=}1$d",    CORAL, "--", "s"),
]


def paired(d, arm, o, task):
    """Mean and sd of the five per-seed differences arm - Wait, within one cell."""
    out = []
    for cap in CAPS:
        g = d[(d.cap == cap) & (d.o_min == o) & (d.task == task) & d.arm.isin([arm, BASE])]
        a = g[g.arm == arm].sort_values("seed")["ne"].to_numpy()
        b = g[g.arm == BASE].sort_values("seed")["ne"].to_numpy()
        if len(a) != 5 or len(b) != 5:
            continue
        diff = a - b
        out.append(dict(x=PARAMS[cap], mean=diff.mean(), sd=diff.std(ddof=1)))
    return pd.DataFrame(out)


def main() -> None:
    d = pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9), sharey=True)
    for ax, o in zip(axes, O_MIN):
        ax.axhline(0, color=INK, lw=1.0, zorder=1)
        for arm, task, lab, col, ls, mk in SERIES:
            p = paired(d, arm, o, task)
            if p.empty:
                continue
            ax.errorbar(p.x, p["mean"], yerr=p.sd, color=col, ls=ls, marker=mk, ms=2.7,
                        lw=1.5, capsize=1.5, elinewidth=.7, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([PARAMS[c] for c in CAPS]); ax.set_xticklabels(CAPS, fontsize=6.3)
        ax.set_xticks([], minor=True)
        ax.tick_params(labelsize=6.3, colors=MUTED, length=2)
        ax.set_title(rf"$o={o}$ min", fontsize=7.5, color=INK, pad=2)
        ax.grid(alpha=.3, lw=.4, color=LINE)
        for sp in ax.spines.values():
            sp.set_color(LINE); sp.set_linewidth(.6)
        ax.set_xlabel("served capacity", fontsize=6.8, color=MUTED)
    axes[0].set_ylabel(r"$\Delta$NE vs Wait", fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.4, frameon=False,
               bbox_to_anchor=(.5, -.13), handlelength=3.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, .10, 1, 1), w_pad=.8)
    out = HERE / "figs" / "delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (compact 1x3)")


if __name__ == "__main__":
    main()
