#!/usr/bin/env python3
"""Every number the paper quotes, emitted from data/ and nothing else.

    python3 paper_numbers.py          # print all blocks
    python3 paper_numbers.py --tex    # macros + tables into generated/

NOT named numbers.py: that shadows the stdlib `numbers` module which numpy imports internally,
and the resulting circular-import error is deeply unobvious.

Rule of the file: if a number appears in main.tex it is produced here, as a \\newcommand macro
or inside a generated table. Nothing is transcribed by hand.

SCOPE. The paper's headline is the join-window collapse, which is measured at BOTH attribution
windows because it is stronger at the longer one -- that contrast is the point. The secondary
result (which target to use) is stated at v=1 d, where the crossover is monotone; at v=7 d it
is not, and the paper says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
CAPS = ["A0", "A1", "A2", "A3", "A4"]
PARAMS = {"A0": 1_890, "A1": 9_810, "A2": 94_959, "A3": 1_184_439, "A4": 9_580_477}
O_MIN = [5, 30, 90]
LR_GRID = (1e-6, 2.5e-6, 1e-5, 5e-5, 2e-4, 5e-4, 1e-3)

#: attribution window -> minutes, for the "how many times shorter" ratios
V_MIN = {"1d": 1440, "7d": 10080}
#: F(o) = P(D<=o | D<=v), measured on the corpus; differs by window
F_O = {"1d": {5: 0.3549, 30: 0.6879, 90: 0.8029},
       "7d": {5: 0.2649, 30: 0.5134, 90: 0.5993}}

WAIT = "DISTILL_T"
CD, DO, FRZ, ULC, ORACLE, FRESH = ("DISTILL_Tmlp", "DISTILL_pvmlp", "DISTILL_Tmlpfrz",
                                   "ULC_aux5e4", "Oracle", "Vanilla_fresh")

#: the published methods the screen covers, in citation order
PUBLISHED = ["DFM", "FSIW", "ES-DFM", "ULC", "DEFUSE", "DDFM", "IF-DFM",
             "FTP", "MISS", "PI", "TWICE"]
CITE = {"DFM": "chapelle2014modeling", "FSIW": "yasui2020feedback", "ES-DFM": "yang2021capturing",
        "ULC": "wang2023unbiased", "DEFUSE": "chen2022asymptotically", "DDFM": "dai2023dually",
        "IF-DFM": "ding2026delayed", "FTP": "li2021follow", "MISS": "liu2024online",
        "PI": "zhang2025personalized", "TWICE": "li2026twice"}


def load() -> pd.DataFrame:
    return pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum.accumulate((p[order] * (m - np.arange(m))).clip(max=1.0)[::-1])[::-1]
    out = np.empty(m)
    out[order] = adj
    return out


def contrast(d: pd.DataFrame, a: str, b: str, task: str) -> pd.DataFrame:
    """Paired over the 5 matched seeds in each (capacity, o) cell; Holm across the 15 cells.

    Paired is correct: the data is byte-identical across seeds, so seed i of two arms differs
    only in initialisation and minibatch order. Comparisons never cross a cell, and never cross
    an attribution window -- a different v is a different target.
    """
    s = d[(d.task == task) & d.arm.isin([a, b])]
    rows = []
    for (cap, o), g in s.groupby(["cap", "o_min"]):
        x = g[g.arm == a].sort_values("seed")["ne"].to_numpy()
        y = g[g.arm == b].sort_values("seed")["ne"].to_numpy()
        if len(x) != 5 or len(y) != 5:
            continue
        rows.append(dict(cap=cap, o_min=int(o), delta=x.mean() - y.mean(),
                         p=stats.ttest_rel(x, y).pvalue))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p.to_numpy())
    r["sig"] = r.p_holm < 0.05
    return r.sort_values(["o_min", "cap"]).reset_index(drop=True)


def screen(which=None) -> pd.DataFrame:
    """Records per click in the served model's training stream, v = 1 d.

    `max_visits <= 1` is the criterion, not the mean: a mean of 1.03 still means some click was
    trained on twice, and it is the existence of a second write that breaks the contract.
    """
    a = pd.read_csv(HERE / "data" / "compliance_audit_1d.csv")
    a = a[a.arm != "Pretrain"]
    a["visits"] = a.visits_per_click.round(2)
    a["pass"] = a.max_visits <= 1
    if which is not None:
        a = a[a.arm.isin(which)].set_index("arm").reindex(which).reset_index()
    return a[["arm", "train_rows", "visits", "max_visits", "pass"]]


# ----------------------------------------------------------------- generated artifacts
def write_screen_table() -> None:
    sc = screen(PUBLISHED)
    L = [r"\begin{tabular}{@{}llrrc@{}}", r"\toprule",
         r"Method & & Train rows & Records/click & Single-visit\\", r"\midrule"]
    for _, r in sc.iterrows():
        mark = r"\checkmark" if r["pass"] else r"$\times$"
        note = "" if r.max_visits <= 1 else f" (max {int(r.max_visits)})"
        L.append(f"\\textsc{{{r.arm.replace('-', '{-}')}}} & \\cite{{{CITE[r.arm]}}} & "
                 f"{r.train_rows:,} & {r.visits:.2f}{note} & {mark}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_screen.tex").write_text("\n".join(L) + "\n")


def write_collapse_table(d: pd.DataFrame) -> None:
    """Table 3, the headline: NE change from replacing a v-long join with an o-short one."""
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Window & New join & " + " & ".join(CAPS) + r"\\", r"\midrule"]
    for task in ("1d", "7d"):
        r = contrast(d, CD, WAIT, task)
        for i, o in enumerate(O_MIN):
            cells = []
            for c in CAPS:
                q = r[(r.cap == c) & (r.o_min == o)]
                v, sig = float(q.delta.iloc[0]), bool(q.sig.iloc[0])
                cells.append((r"\textbf{%+.4f}" % v) if sig else ("%+.4f" % v))
            head = (r"$v=1$\,d" if task == "1d" else r"$v=7$\,d") if i == 0 else ""
            ratio = V_MIN[task] // o
            L.append(f"{head} & {o}\\,min ($\\div{ratio}$) & " + " & ".join(cells) + r"\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_collapse.tex").write_text("\n".join(L) + "\n")


def macros(d: pd.DataFrame) -> dict[str, str]:
    cd1, cd7 = contrast(d, CD, WAIT, "1d"), contrast(d, CD, WAIT, "7d")
    do1, do7 = contrast(d, DO, WAIT, "1d"), contrast(d, DO, WAIT, "7d")
    fz1, fz7 = contrast(d, FRZ, CD, "1d"), contrast(d, FRZ, CD, "7d")
    fw7 = contrast(d, FRZ, WAIT, "7d")
    u1 = contrast(d, CD, ULC, "1d")
    wf1 = contrast(d, WAIT, FRESH, "1d")
    sc = screen(PUBLISHED)
    studied = [CD, DO, FRZ, ULC, WAIT, ORACLE]
    sd = d[d.arm.isin(studied)].groupby(["arm", "task", "cap", "o_min"])["ne"].std()
    m = {
        # scale of the run
        "nRuns": f"{len(d):,}", "nArms": str(int(d.arm.nunique())),
        "nEval": f"{int(d[d.task=='1d'].n_eval.iloc[0]):,}",
        "nCohorts": f"{int(d.n_cohorts.iloc[0])}",
        "baseRate": f"{float(d[d.task=='1d'].base.iloc[0]):.4f}",
        "nLr": str(len(LR_GRID)), "nContrast": str(len(CAPS) * len(O_MIN)),
        "capMin": f"{PARAMS['A0']:,}", "capMax": f"{PARAMS['A4']:,}",
        "teacherParams": "9,456,897",
        "seedSdMed": f"{sd.median():.5f}", "seedSdMax": f"{sd.max():.5f}",
        # the screen
        "nScreened": str(len(sc)), "nFail": str(int((~sc["pass"]).sum())),
        "passList": ", ".join(sc[sc["pass"]].arm),
        "maxDup": f"{sc.visits.max():.2f}", "maxDupArm": sc.loc[sc.visits.idxmax(), "arm"],
        "maxVisits": str(int(sc.max_visits.max())),
        "maxVisitsArm": sc.loc[sc.max_visits.idxmax(), "arm"],
        "missDup": f"{float(sc.loc[sc.arm=='MISS','visits'].iloc[0]):.2f}",
        "vanillaDup": f"{float(screen(['Vanilla']).visits.iloc[0]):.2f}",
        # THE HEADLINE: collapsing the join window
        "collapseSevenBest": f"{cd7.delta.min():.4f}", "collapseSevenWorst": f"{cd7.delta.max():.4f}",
        "collapseOneBest": f"{cd1.delta.min():.4f}", "collapseOneWorst": f"{cd1.delta.max():.4f}",
        "collapseSevenSig": str(int(cd7.sig.sum())), "collapseOneSig": str(int(cd1.sig.sum())),
        "collapseSevenMean": f"{cd7.delta.mean():.4f}", "collapseOneMean": f"{cd1.delta.mean():.4f}",
        "ratioSeven": str(V_MIN["7d"] // 5), "ratioOne": str(V_MIN["1d"] // 5),
        "foSevenFive": f"{F_O['7d'][5]:.3f}", "foOneFive": f"{F_O['1d'][5]:.3f}",
        "foOneNinety": f"{F_O['1d'][90]:.3f}",
        # the condition: which target
        "doSmall": f"{do1[do1.cap.isin(['A0','A1'])].delta.mean():.4f}",
        "doLarge": f"{do1[do1.cap.isin(['A2','A3','A4'])].delta.mean():+.4f}",
        "doSig": str(int(do1.sig.sum())),
        "doSevenLargest": f"{do7[do7.cap=='A4'].delta.mean():+.4f}",
        # frozen teacher
        "frzOne": f"{fz1.delta.mean():+.5f}", "frzSeven": f"{fz7.delta.mean():+.5f}",
        "frzSevenBeat": str(int((fw7.delta < 0).sum())), "frzSevenN": str(len(fw7)),
        # references
        "cdUlcMean": f"{u1.delta.mean():+.5f}", "cdUlcNs": str(int((~u1.sig).sum())),
        "waitFreshFive": f"{wf1[wf1.o_min==5].delta.mean():.4f}",
    }
    return m


def main() -> None:
    d = load()
    pd.set_option("display.width", 200)
    print(f"=== {len(d)} runs, {d.arm.nunique()} arms, windows {sorted(d.task.unique())} ===\n")
    print("--- Rule-2 screen (v=1d) ---")
    print(screen(PUBLISHED).to_string(index=False))
    for task in ("1d", "7d"):
        for lab, a, b in (("Correct-Distill - Wait  [THE COLLAPSE]", CD, WAIT),
                          ("Distill-Only  - Wait  [the condition]", DO, WAIT),
                          ("frozen        - live teacher", FRZ, CD)):
            r = contrast(d, a, b, task)
            print(f"\n--- v={task}  {lab} ---")
            print(r.pivot_table(index="o_min", columns="cap", values="delta").round(5).to_string())
            print(f"    mean {r.delta.mean():+.5f}   significant {int(r.sig.sum())}/{len(r)}")
    print("\n--- macros ---")
    M = macros(d)
    for k, v in M.items():
        print(f"  {k:<20} {v}")
    if "--tex" in sys.argv:
        GEN.mkdir(exist_ok=True)
        bad = [k for k in M if not k.isalpha()]
        if bad:
            raise SystemExit(f"macro names must be letters only: {bad}")
        (GEN / "macros.tex").write_text(
            "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
        write_screen_table()
        write_collapse_table(d)
        print(f"\nwrote {GEN}/macros.tex, tab_screen.tex, tab_collapse.tex")


if __name__ == "__main__":
    main()
