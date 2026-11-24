# Report (CVPR template)

`main.tex` is a complete draft: prose, figures, tables and bibliography are
already in place. The numbers are **generated**, not typed.

## How to compile (Overleaf, ~5 minutes)

1. Overleaf → **New Project → Templates** → search **“CVPR”** → use the official
   CVPR template project (it ships `cvpr.sty`, `ieeenat_fullname.bst`, …).
2. In that project, **replace** `main.tex` and add:
   - `references.bib`
   - `figs/` (pipeline.png, comparison.png, auroc_vs_shot.png)
   - `tables/` (main_image.tex, main_pixel.tex, per_category.tex, efficiency.tex)
3. Upload the folder structure as-is — the `\input{}` / `\includegraphics{}`
   paths are relative to `main.tex`.
4. Compile with **pdfLaTeX** (default). If the bibliography does not appear, set
   the compiler to `pdfLaTeX` and hit *Recompile* twice.

`\usepackage[review]{cvpr}` keeps the anonymous review mode. For arXiv /
camera-ready, drop the `[review]` option, fill in the author block properly and
switch the page numbering as the template instructs.

## Regenerating the tables

```bash
cd p2 && PYTHONPATH=.pylibs python3 report/make_tables.py
```

Tables are written to `report/tables/`. Never edit them by hand — if a number
changes, re-run the experiments and this script.

## Before submitting — checklist from the brief

- [ ] **6–8 pages** excluding references (current draft ≈ 6.5 pages of text;
      check after compiling and trim/add as needed)
- [ ] **GitHub link at the end of the abstract** (already present)
- [ ] Related Work cites **all 12** papers of the recommended reading list
      (MVTec AD, PaDiM, PatchCore, DRAEM, CutPaste, DINOv2, AnomalyCLIP,
      AnomalyDINO, EfficientAD, MVTec AD 2, VisA, SAM) — check with a `\cite`
      count: `grep -o "\\\\cite{[^}]*}" main.tex | wc -l`
- [ ] Pipeline figure (`figs/pipeline.png`) and at least one qualitative figure
      (`figs/comparison.png`) are present
- [ ] Quantitative tables: image AUROC, pixel AUROC, per-category, efficiency
- [ ] Ablation study (the 2×2 layer×scale design) and a **categorised failure
      analysis** (the chewing-gum section)
- [ ] Author names / emails / group number filled in the `\author{}` block
- [ ] If uploading to arXiv: camera-ready mode + arXiv ID on the Canvas submission

## Things that are still placeholders

- `\author{}`: emails are guessed (`yqiao`, `wjingkai`) — replace with the real ones.
- The abstract says “six VisA categories”; keep it that way, it matches the brief.
- If you add the Part 3 qualitative figure (fusion vs single-block heatmaps),
  drop it next to `figs/comparison.png` and reference it in Sec. 4.5.
