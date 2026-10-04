# Arithmetic editing manuscript

- `main.pdf`: compiled 11-page NeurIPS-style manuscript, including references and appendices.
- `main.tex`: entry point; sections, tables, figures, and commands are modular.
- `references.bib`: citations based on the supplied review and archived metadata.
- `REVIEW.md`: completed content, formatting, and build review.
- `../paper/OUTLINE.md`: outline and evidence map prepared before prose.

Build from this directory:

```sh
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The supplied style requires a track notice without defining a default for `[final]`. `commands/macros.tex` supplies a manuscript notice; it does not claim conference acceptance. The requested style options and author line are retained. The abstract uses the standard unnumbered NeurIPS abstract environment; all body section files contain their section commands.

The example directory contained only a README. Formatting follows the supplied lab style guide and command templates. No example-paper content was available or copied. Repository references dated 2026 denote the accessed model/dataset snapshots, not original publication dates.
