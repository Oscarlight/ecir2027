# v1 — "One Record per Click" (archived 2026-10-03)

Superseded by the join-window framing. Nothing here is wrong; the framing was weaker.

v1's protagonist was the **single-visit constraint** — a screening device, which tells a reader
what cannot be done. v2 keeps the constraint (it is still what rules out duplication as an
escape) but makes the **join window** the protagonist: the claim becomes that a 5-minute join
can replace a 7-day one and *gain* accuracy, with the gain growing as the funnel slows.

Two things v1 under-used, both already in the data:
  * it scoped itself to v=1d, so it never reported that collapsing a 7-day join is worth
    2--3x more than collapsing a 1-day one (-0.0055..-0.0171 vs -0.0005..-0.0074 NE);
  * it framed the Distill-Only crossover as a standalone negative result rather than as the
    *condition* under which the collapse is safe.

Build of this version: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(12 pages; the page budget is the other reason for the rewrite).
