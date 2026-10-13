# LaTeX source — "Aggregate Accuracy Does Not Reveal Reliability"

Overleaf-ready ISPRS-J (Elsevier `elsarticle`) project; this source is release v1.1.0.

**Files:** `main.tex`, `refs.bib`, `fig2_flagship_reliability.pdf`, `fig3_domain_gaps.pdf`, plus a compiled
`main.pdf` and the `main.bbl` it was built with.

**Compile on Overleaf:** upload the four source files, set the compiler to pdfLaTeX, Recompile.
**Compile locally:** `latexmk -pdf main.tex` (or `pdflatex; bibtex main; pdflatex; pdflatex`).

`main.pdf` (22 pages) was compiled at commit f9d3eb0. Four release commits follow it, and two of them change
the paper (a81f29e: text corrections; 2b83e7f: float spacing and the two figures' legends), so it lags
v1.1.0 in those places -- compile `main.tex` for the release text. `../reliability_paper.pdf` is an earlier
9-page draft (2026-07-22, titled "Robustness is not Reliability"), kept for the record.
