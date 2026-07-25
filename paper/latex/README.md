# LaTeX source — "Robustness is not Reliability"

Overleaf-ready ISPRS-J (Elsevier `elsarticle`) project.

**Files:** `main.tex`, `refs.bib`, `fig2_flagship_reliability.pdf`, `fig3_domain_gaps.pdf`.

**Compile on Overleaf:** upload all four files, set the compiler to pdfLaTeX, Recompile.
**Compile locally:** `latexmk -pdf main.tex` (or `pdflatex; bibtex main; pdflatex; pdflatex`).

Bibliography entries beginning with a `% VERIFY` comment line (liu2024sacp, francis2021sensei,
green2020emit) have one field to confirm against the source before submission. Author affiliations
are placeholders ("Affiliation TBD"). The compiled reference PDF is committed as
`../reliability_paper.pdf` (9 pages).
