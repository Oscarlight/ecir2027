#!/usr/bin/env python3
"""Independently re-derive every headline claim in the paper, straight from the parquet.

    python3 check_claims.py        # exits non-zero if any claim fails

Deliberately does NOT import paper_numbers. A second implementation is the whole point: it
caught a claim that the significant Correct-Distill/ULC cells "split in both directions", which
was true of an earlier 1d+7d analysis and false at 1d, where both favour ULC.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

d = pd.read_parquet("data/v3_all_runs.parquet")
d = d[d.task == "1d"]


def cells(a, b):
    out = []
    for (c, o), g in d[d.arm.isin([a, b])].groupby(["cap", "o_min"]):
        x = g[g.arm == a].sort_values("seed")["ne"].to_numpy()
        y = g[g.arm == b].sort_values("seed")["ne"].to_numpy()
        if len(x) == 5 == len(y):
            out.append((c, o, x.mean() - y.mean(), stats.ttest_rel(x, y).pvalue))
    r = pd.DataFrame(out, columns=["cap", "o", "delta", "p"])
    k = np.argsort(r.p.to_numpy())
    m = len(r)
    adj = np.minimum.accumulate((r.p.to_numpy()[k] * (m - np.arange(m))).clip(max=1)[::-1])[::-1]
    ph = np.empty(m)
    ph[k] = adj
    r["sig"] = ph < 0.05
    return r


def main() -> int:
    cd, do = cells("DISTILL_Tmlp", "DISTILL_T"), cells("DISTILL_pvmlp", "DISTILL_T")
    cdo, u = cells("DISTILL_Tmlp", "DISTILL_pvmlp"), cells("DISTILL_Tmlp", "ULC_aux5e4")
    wf, fz = cells("DISTILL_T", "Vanilla_fresh"), cells("DISTILL_Tmlpfrz", "DISTILL_Tmlp")
    fw = cells("DISTILL_Tmlpfrz", "DISTILL_T")
    sm = do[do.cap.isin(["A0", "A1"])].delta.mean()
    lg = do[do.cap.isin(["A2", "A3", "A4"])].delta.mean()
    C = [
        ("CD below Wait in all 15", bool((cd.delta < 0).all()), f"{int((cd.delta<0).sum())}/15"),
        ("CD significant in 10", int(cd.sig.sum()) == 10, int(cd.sig.sum())),
        ("CD gain decays with capacity",
         cd[cd.cap == "A0"].delta.mean() < cd[cd.cap == "A4"].delta.mean(),
         f"{cd[cd.cap=='A0'].delta.mean():.4f} -> {cd[cd.cap=='A4'].delta.mean():.4f}"),
        ("DO helps small, hurts large", sm < 0 < lg, f"{sm:+.4f} / {lg:+.4f}"),
        ("DO crosses between A1 and A2 at every o",
         all(do[(do.o == o) & (do.cap == "A1")].delta.iloc[0] < 0
             < do[(do.o == o) & (do.cap == "A2")].delta.iloc[0] for o in (5, 30, 90)), "yes"),
        ("DO significant 15/15", int(do.sig.sum()) == 15, int(do.sig.sum())),
        ("CD beats DO in all 15, all significant",
         bool((cdo.delta < 0).all()) and int(cdo.sig.sum()) == 15, f"sig {int(cdo.sig.sum())}"),
        ("CD~ULC: 13 of 15 not significant", int((~u.sig).sum()) == 13, int((~u.sig).sum())),
        ("CD~ULC: ULC better in BOTH significant cells",
         bool((u[u.sig].delta > 0).all()), list(u[u.sig].delta.round(5))),
        ("Wait beats Fresh hugely at o=5", wf[wf.o == 5].delta.mean() < -0.2,
         f"{wf[wf.o==5].delta.mean():.4f}"),
        ("freezing the teacher costs little on average", 0 < fz.delta.mean() < 0.002,
         f"{fz.delta.mean():+.5f}"),
        ("freezing is free at o=90", fz[fz.o == 90].delta.mean() < 0,
         f"{fz[fz.o==90].delta.mean():+.5f}"),
        ("frozen CD still beats Wait in 12/15", int((fw.delta < 0).sum()) == 12,
         int((fw.delta < 0).sum())),
    ]
    bad = 0
    for n, ok, v in C:
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {n:<44} {v}")
    print(f"\n{len(C)-bad}/{len(C)} claims reproduce")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
