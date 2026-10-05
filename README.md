# One Record per Click — short paper

Self-contained. Everything in `main.pdf` is derived from `data/`; nothing is transcribed by hand.

## Build

```
python3 paper_numbers.py --tex     # macros + Tables 1 and 2 -> generated/
python3 make_fig.py                # Figure 1 -> figs/
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Verification

```
python3 check_claims.py            # re-derives every headline claim from the parquet
```

`check_claims.py` deliberately does **not** import `paper_numbers.py` — it is a second,
independent implementation. That is not ceremony: it caught a claim that the significant
Correct-Distill/ULC cells "split in both directions", which was true of an earlier 1d+7d
analysis and false at 1d, where both favour ULC.

## Layout

| file | role |
|---|---|
| `main.tex` | preamble, notation macros, section includes |
| `sections/` | one file per section |
| `paper_numbers.py` | **single source of every number**; emits `generated/macros.tex` + tables |
| `check_claims.py` | independent re-derivation of the headline claims |
| `make_fig.py` | Figure 1 |
| `data/v3_all_runs.parquet` | the run output (1-day slice is what the paper uses) |
| `data/compliance_audit_1d.csv` | the single-visit screen |

Macro names must be letters only — `\cdWaitA0` parses as `\cdWaitA` followed by `0` and fails
with "Missing \begin{document}". `paper_numbers.py` asserts this.

## Before submission

- `refs.bib` entries `xu2020privileged` and `menon2021statistical` carry `author = {X and others}`.
  Complete them from the DOI/arXiv id. **Do not invent author lists.**
- Fill in author/affiliation in `main.tex`.
- The paper is currently ~12 pages. See the handover notes for what cutting to 6 would cost.
