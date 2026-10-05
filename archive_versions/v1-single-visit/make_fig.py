#!/usr/bin/env python3
"""Figure 1: NE delta against \\textsc{Wait}, per attribution window and write delay.

A 2x3 grid: rows are the attribution window (1 d, 7 d), columns the pipeline write delay. The
7-day row is a robustness check on a paper whose claims are stated for 1 day -- it is included
precisely because the two rows do NOT say the same thing, and a reader is entitled to see that.

    python3 make_fig.py        # -> figs/delta.pdf

Plotting the paired difference, not two absolute curves, is deliberate. The absolute $NE$ of
every arm falls steeply with capacity, which visually swamps the between-arm differences the
paper is about; the difference removes the shared capacity trend and puts zero -- the status
quo -- on a line you can read against.

Error bars are the standard deviation of the FIVE PER-SEED DIFFERENCES, which is the dispersion
of the estimand. Differencing two independently computed standard deviations would ignore the
pairing and materially overstate the uncertainty.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

HERE = Path(__file__).resolve().parent
CAPS = ["A0", "A1", "A2", "A3", "A4"]
PARAMS = {"A0": 1_890, "A1": 9_810, "A2": 94_959, "A3": 1_184_439, "A4": 9_580_477}
O_MIN = [5, 30, 90]
TASKS = [("1d", r"$v = 1$ day"), ("7d", r"$v = 7$ days")]
#: F(o) = P(D<=o | D<=v), measured per window. It differs between rows, so one dict keyed only
#: by o would put the 1-day numbers on the 7-day panels.
F_O = {"1d": {5: 0.355, 30: 0.688, 90: 0.803},
       "7d": {5: 0.265, 30: 0.513, 90: 0.599}}

INK, MUTED, LINE = "#222222", "#6b7a7d", "#c9d4cf"
TEAL, CORAL, SKY = "#12807a", "#e0674f", "#3D8BB8"

#: arm -> (label, colour, linestyle, marker, linewidth, z)
SERIES = [
    ("DISTILL_Tmlp",   "Correct-Distill",    TEAL,  "-",  "o", 1.9, 5),
    ("DISTILL_pvmlp",  "Distill-Only",       CORAL, "-",  "s", 1.7, 4),
    ("ULC_aux5e4",     "ULC (online)",       SKY,   "--", "^", 1.2, 3),
    ("DISTILL_Tmlpfrz", "Correct-Distill, frozen teacher", TEAL, ":", "d", 1.2, 2),
]
BASE = "DISTILL_T"      # Wait: the zero line


def paired(d: pd.DataFrame, arm: str, o: int, task: str) -> pd.DataFrame:
    """Per-capacity mean and sd of the per-seed difference arm - Wait, within one cell."""
    out = []
    for cap in CAPS:
        g = d[(d.cap == cap) & (d.o_min == o) & (d.task == task) & d.arm.isin([arm, BASE])]
        a = g[g.arm == arm].sort_values("seed")["ne"].to_numpy()
        b = g[g.arm == BASE].sort_values("seed")["ne"].to_numpy()
        if len(a) != 5 or len(b) != 5:
            continue
        diff = a - b
        out.append(dict(cap=cap, x=PARAMS[cap], mean=diff.mean(), sd=diff.std(ddof=1)))
    return pd.DataFrame(out)


def main() -> None:
    d = pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")
    have = set(zip(d.arm, d.task))
    fig, axes = plt.subplots(len(TASKS), len(O_MIN), figsize=(6.6, 3.9),
                             sharey="row", sharex=True)
    for r, (task, tlab) in enumerate(TASKS):
        for c, o in enumerate(O_MIN):
            ax = axes[r][c]
            ax.axhline(0, color=INK, lw=1.0, zorder=1)
            for arm, lab, col, ls, mk, lw, z in SERIES:
                if (arm, task) not in have:
                    continue
                p = paired(d, arm, o, task)
                if p.empty:
                    continue
                ax.errorbar(p.x, p["mean"], yerr=p.sd, color=col, ls=ls, marker=mk, ms=2.9,
                            lw=lw, capsize=1.6, elinewidth=.7, zorder=z, label=lab)
            ax.set_xscale("log")
            ax.set_xticks([PARAMS[k] for k in CAPS])
            ax.set_xticklabels(CAPS, fontsize=6.5)
            ax.set_xticks([], minor=True)
            ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
            ax.grid(alpha=.3, lw=.4, color=LINE)
            for sp in ax.spines.values():
                sp.set_color(LINE); sp.set_linewidth(.6)
            if r == 0:
                ax.set_title(rf"$o={o}$ min   ($F(o)={F_O[task][o]:.2f}$)",
                             fontsize=7.6, color=INK, pad=3)
            else:
                # F(o) differs by window, so it belongs on both rows, not only the top
                ax.text(.5, 1.02, rf"$F(o)={F_O[task][o]:.2f}$", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6.5, color=MUTED)
            if r == len(TASKS) - 1:
                ax.set_xlabel("served capacity", fontsize=7, color=MUTED)
        axes[r][0].set_ylabel(rf"{tlab}" "\n" r"$\Delta$NE vs Wait", fontsize=7.6)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.8, frameon=False,
               bbox_to_anchor=(.5, -.055), handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, .055, 1, 1), h_pad=1.6, w_pad=.9)
    out = HERE / "figs" / "delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    missing = [(a, t) for a, *_ in [(x[0],) for x in SERIES] for t, _ in TASKS
               if (a, t) not in have]
    print(f"wrote {out}")
    if missing:
        print(f"  MISSING series/window pairs (omitted from those panels): {missing}")


if __name__ == "__main__":
    main()
