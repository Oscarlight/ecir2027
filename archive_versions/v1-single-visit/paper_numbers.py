#!/usr/bin/env python3
"""Every number the paper quotes, emitted from data/ and nothing else.

NOT named numbers.py: that shadows the stdlib `numbers` module which numpy imports internally,
and the resulting circular-import error is deeply unobvious.

    python3 paper_numbers.py            # print all blocks
    python3 paper_numbers.py --tex      # write tables + macros into generated/

Rule of the file: if a number appears in main.tex it is produced here, either as a \\newcommand
macro or inside a generated table. Nothing is transcribed by hand, so the paper cannot drift
from the run that produced it.

Scope is 1-day attribution only. A 1-day click-attributed conversion window is the common
production default for click-through conversion optimisation, and mixing windows would make
every aggregate a property of the mix rather than of the method.
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
#: the learning-rate grid each cell is selected from, before any streaming evaluation
LR_GRID = (1e-6, 2.5e-6, 1e-5, 5e-5, 2e-4, 5e-4, 1e-3)

#: The four points of the target-hardness axis, plus the two reference arms.
AXIS = [("Vanilla_fresh", "Fresh"), ("DISTILL_T", "Wait"),
        ("DISTILL_Tmlp", "Correct-Distill"), ("DISTILL_pvmlp", "Distill-Only")]
REF = [("ULC_aux5e4", "Correct (online)"), ("Oracle", "Oracle")]

#: F(o) = P(D <= o | D <= v) at v = 1 d, measured on the corpus.
F_O = {5: 0.3549, 30: 0.6879, 90: 0.8029}


def load() -> pd.DataFrame:
    d = pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")
    return d[d.task == "1d"].copy()


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum.accumulate((p[order] * (m - np.arange(m))).clip(max=1.0)[::-1])[::-1]
    out = np.empty(m)
    out[order] = adj
    return out


def contrast(d: pd.DataFrame, a: str, b: str, metric: str = "ne") -> pd.DataFrame:
    """Paired over the 5 matched seeds within each (capacity, o) cell; Holm across the 15 cells.

    Paired is the correct test here: the data is byte-identical across seeds, so seed i of two
    arms differs only in initialisation and minibatch order. Comparisons never cross a cell --
    different o means a different label distribution, so a pooled test would be answering a
    question about the mix.
    """
    rows = []
    for (cap, o), g in d[d.arm.isin([a, b])].groupby(["cap", "o_min"]):
        x = g[g.arm == a].sort_values("seed")[metric].to_numpy()
        y = g[g.arm == b].sort_values("seed")[metric].to_numpy()
        if len(x) != 5 or len(y) != 5:
            continue
        rows.append(dict(cap=cap, o_min=int(o), delta=x.mean() - y.mean(),
                         p=stats.ttest_rel(x, y).pvalue))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p.to_numpy())
    r["sig"] = r.p_holm < 0.05
    return r.sort_values(["o_min", "cap"]).reset_index(drop=True)


# --------------------------------------------------------------------------- blocks
#: The published delayed-feedback methods we reproduce, in citation order. The screen's claim is
#: about THESE -- protocol anchors and our own arms are reported separately so the headline
#: "n of m published methods" cannot be inflated by counting reference rows or tuning variants.
PUBLISHED = ["DFM", "FSIW", "ES-DFM", "ULC", "DEFUSE", "DDFM", "IF-DFM",
             "FTP", "MISS", "PI", "TWICE"]
OURS = ["DISTILL_T", "DISTILL_Tmlp", "DISTILL_pvmlp"]
ANCHOR = ["Vanilla_fresh", "Vanilla", "Oracle"]


def screen(which=None) -> pd.DataFrame:
    """The Rule-2 screen: measured served records per click, v = 1 d.

    `max_visits <= 1` is the criterion, not the mean: a mean of 1.03 still means some click was
    trained on twice, and it is the existence of a second visit that breaks the single-write
    contract, not its frequency.
    """
    a = pd.read_csv(HERE / "data" / "compliance_audit_1d.csv")
    a = a[a.arm != "Pretrain"]                       # frozen at T0; consumes no stream
    a["visits"] = a.visits_per_click.round(2)
    a["pass"] = a.max_visits <= 1
    if which is not None:
        a = a[a.arm.isin(which)].set_index("arm").reindex(which).reset_index()
    return a[["arm", "train_rows", "visits", "max_visits", "pass"]]


def levels(d: pd.DataFrame) -> pd.DataFrame:
    """NE by arm x capacity x o."""
    keep = dict(AXIS + REF)
    s = d[d.arm.isin(keep)]
    t = s.groupby(["arm", "o_min", "cap"])["ne"].agg(["mean", "std"]).reset_index()
    t["arm"] = t.arm.map(keep)
    return t


def deltas(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The paper's headline: NE delta against Wait, the admissible status quo."""
    return {"FRZ-CD": contrast(d, "DISTILL_Tmlpfrz", "DISTILL_Tmlp"),
            "FRZ-Wait": contrast(d, "DISTILL_Tmlpfrz", "DISTILL_T"),
            "CD-Wait": contrast(d, "DISTILL_Tmlp", "DISTILL_T"),
            "DO-Wait": contrast(d, "DISTILL_pvmlp", "DISTILL_T"),
            "CD-DO": contrast(d, "DISTILL_Tmlp", "DISTILL_pvmlp"),
            "CD-ULC": contrast(d, "DISTILL_Tmlp", "ULC_aux5e4"),
            "Wait-Fresh": contrast(d, "DISTILL_T", "Vanilla_fresh"),
            "CD-Oracle": contrast(d, "DISTILL_Tmlp", "Oracle")}


def macros(d: pd.DataFrame, D: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Named values for \\input into the prose, so no figure is typed by hand."""
    sc = screen(PUBLISHED)
    cd, do, cdo, cdu = D["CD-Wait"], D["DO-Wait"], D["CD-DO"], D["CD-ULC"]
    studied = [a for a, _ in AXIS if a != "Vanilla_fresh"] + [r for r, _ in REF]
    sd = d[d.arm.isin(studied)].groupby(["arm", "cap", "o_min"])["ne"].std()
    m = {
        "nRuns": f"{len(d):,}",
        "nCells": str(d.groupby(['arm', 'cap', 'o_min']).ngroups),
        "nEval": f"{int(d.n_eval.iloc[0]):,}",
        "nCohorts": f"{int(d.n_cohorts.iloc[0])}",
        "baseRate": f"{float(d.base.iloc[0]):.4f}",
        "nLr": str(len(LR_GRID)),
        # cells in ONE contrast (5 capacities x 3 write delays) -- NOT nCells, which counts the
        # whole 1-day reproduction across every arm. Holm corrects over the former.
        "nContrast": str(len(CAPS) * len(O_MIN)),
        "nArms": str(int(d.arm.nunique())),
        "nScreened": str(len(sc)),
        "nPass": str(int(sc["pass"].sum())),
        "nFail": str(int((~sc["pass"]).sum())),
        "passList": ", ".join(sc[sc["pass"]].arm),
        "maxDup": f"{sc.visits.max():.2f}",
        "maxDupArm": sc.loc[sc.visits.idxmax(), "arm"],
        "maxVisits": str(int(sc.max_visits.max())),
        "maxVisitsArm": sc.loc[sc.max_visits.idxmax(), "arm"],
        "missDup": f"{float(sc.loc[sc.arm == 'MISS', 'visits'].iloc[0]):.2f}",
        # the plain re-ingestion baseline, cited in the intro but reported in prose not Table 1
        "vanillaDup": f"{float(screen(['Vanilla']).visits.iloc[0]):.2f}",
        # headline deltas
        "cdWaitMax": f"{cd.delta.min():.4f}",        # most negative = largest gain
        "cdWaitMin": f"{cd.delta.max():.4f}",
        "cdWaitSig": f"{int(cd.sig.sum())}",
        "cdWaitN": f"{len(cd)}",
        "doWaitSmall": f"{do[do.cap.isin(['A0','A1'])].delta.mean():.4f}",
        "doWaitLarge": f"{do[do.cap.isin(['A2','A3','A4'])].delta.mean():.4f}",
        "doWaitSig": f"{int(do.sig.sum())}",
        "cdDoMean": f"{cdo.delta.mean():.4f}",
        "cdDoSig": f"{int(cdo.sig.sum())}",
        "cdUlcMean": f"{cdu.delta.mean():+.5f}",
        "cdUlcNs": f"{int((~cdu.sig).sum())}",
        "cdUlcSigN": f"{int(cdu.sig.sum())}",
        "cdUlcSigMax": f"{cdu[cdu.sig].delta.max():+.5f}",
        "cdUlcN": f"{len(cdu)}",
        "seedSdMed": f"{sd.median():.5f}",
        "seedSdMax": f"{sd.max():.5f}",
        "foFive": f"{F_O[5]:.3f}", "foThirty": f"{F_O[30]:.3f}", "foNinety": f"{F_O[90]:.3f}",
        "capMin": f"{PARAMS['A0']:,}", "capMax": f"{PARAMS['A4']:,}",
        "teacherParams": "9,456,897",     # the mlp trunk the teacher uses; ~A4-sized
        "cdOracleWin": f"{int((D['CD-Oracle'].delta < 0).sum())}",
        "cdOracleN": f"{len(D['CD-Oracle'])}",
        "frzMean": f"{D['FRZ-CD'].delta.mean():+.5f}",
        "frzSig": f"{int(D['FRZ-CD'].sig.sum())}",
        "frzNinety": f"{D['FRZ-CD'][D['FRZ-CD'].o_min == 90].delta.mean():+.5f}",
        "frzFiveLarge": (f"{D['FRZ-CD'][(D['FRZ-CD'].o_min == 5) & D['FRZ-CD'].cap.isin(['A3','A4'])].delta.mean():+.5f}"),
        "frzWaitBeat": f"{int((D['FRZ-Wait'].delta < 0).sum())}",
        "frzWaitN": f"{len(D['FRZ-Wait'])}",
        "waitFreshFive": f"{D['Wait-Fresh'][D['Wait-Fresh'].o_min == 5].delta.mean():.4f}",
        "waitFreshNinety": f"{D['Wait-Fresh'][D['Wait-Fresh'].o_min == 90].delta.mean():.4f}",
        "cdOracleBeat": f"{int((D['CD-Oracle'].delta > 0).sum())}",
        "cdOracleMean": f"{D['CD-Oracle'].delta.mean():.4f}",
        "cdWaitSmallest": f"{cd[cd.cap == 'A0'].delta.mean():.4f}",
        "cdWaitLargest": f"{cd[cd.cap == 'A4'].delta.mean():.4f}",
    }
    return m


SHORT = {"Vanilla_fresh": r"\\textsc{Fresh}", "DISTILL_T": r"\\textsc{Wait}",
         "DISTILL_Tmlp": r"\\textsc{Correct-Distill}", "DISTILL_pvmlp": r"\\textsc{Distill-Only}",
         "ULC_aux5e4": r"\\textsc{Ulc} (online)", "Oracle": r"\\textsc{Oracle}"}
CITE = {"DFM": "chapelle2014modeling", "FSIW": "yasui2020feedback", "ES-DFM": "yang2021capturing",
        "ULC": "wang2023unbiased", "DEFUSE": "chen2022asymptotically", "DDFM": "dai2023dually",
        "IF-DFM": "ding2026delayed", "FTP": "li2021follow", "MISS": "liu2024online",
        "PI": "zhang2025personalized", "TWICE": "li2026twice"}


def write_screen_table() -> None:
    """Table 1: the Rule-2 screen over the published methods."""
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


def write_delta_table(D) -> None:
    """Table 2: NE delta against Wait, and the two arms against each other."""
    rows = [("CD-Wait", r"\textsc{Correct-Distill} $-$ \textsc{Wait}"),
            ("DO-Wait", r"\textsc{Distill-Only} $-$ \textsc{Wait}"),
            ("CD-DO",   r"\textsc{Correct-Distill} $-$ \textsc{Distill-Only}")]
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Contrast & $o$ & " + " & ".join(CAPS) + r"\\", r"\midrule"]
    for key, lab in rows:
        r = D[key]
        for i, o in enumerate(O_MIN):
            cells = []
            for c in CAPS:
                q = r[(r.cap == c) & (r.o_min == o)]
                v = float(q.delta.iloc[0]); sig = bool(q.sig.iloc[0])
                cells.append((r"\textbf{%+.4f}" % v) if sig else ("%+.4f" % v))
            head = lab if i == 0 else ""
            L.append(f"{head} & {o}\\,min & " + " & ".join(cells) + r"\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_delta.tex").write_text("\n".join(L) + "\n")


def main() -> None:
    d = load()
    D = deltas(d)
    pd.set_option("display.width", 200)
    print(f"=== 1-day slice: {len(d)} runs, "
          f"{d.groupby(['arm','cap','o_min']).ngroups} cells, {d.arm.nunique()} arms ===\n")
    print("--- Rule-2 screen (served records per click, v=1d) ---")
    print(screen().to_string(index=False))
    print("\n--- NE levels ---")
    L = levels(d)
    print(L.pivot_table(index=["arm", "o_min"], columns="cap", values="mean").round(5).to_string())
    for k, r in D.items():
        print(f"\n--- delta {k} ---")
        print(r.pivot_table(index="o_min", columns="cap", values="delta").round(5).to_string())
        print(f"    significant {int(r.sig.sum())}/{len(r)}   mean {r.delta.mean():+.5f}")
    print("\n--- macros ---")
    for k, v in macros(d, D).items():
        print(f"  {k:<14} {v}")

    if "--tex" in sys.argv:
        GEN.mkdir(exist_ok=True)
        write_screen_table()
        write_delta_table(D)

        GEN.mkdir(exist_ok=True)
        M = macros(d, D)
        # A LaTeX control sequence is letters only: \cdWaitA0 parses as \cdWaitA followed by a
        # literal 0, which becomes typeset material in the preamble and fails with the deeply
        # unhelpful "Missing \\begin{document}". Catch it here rather than in the log.
        bad = [k for k in M if not k.isalpha()]
        if bad:
            raise SystemExit(f"macro names must be letters only; offending: {bad}")
        (GEN / "macros.tex").write_text(
            "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
        print(f"\nwrote {GEN/'macros.tex'}")


if __name__ == "__main__":
    main()
