# Rebuilding the report

The editable report is an IEEE-style course report, rather than a certified conference-template submission. The exported document has four pages of text and references and one page of supplementary results. The author block and personal document metadata are intentionally empty.

`manuscript.md` contains the editable prose and the two main tables. `build_report.py` supplies the typography, verified references, figure generation, and complete sixteen-policy appendix. The appendix and chart counts are read from `../../experiments/v3/results/metrics.json`; rebuilding the report does not run training, call the answering model, or modify experiment artifacts. `references.json` mirrors the bibliography embedded in the builder.

## Dependencies

The build was verified with Python 3.12.14, python-docx 1.2.0, ReportLab 4.4.9, Pillow 13.0.0, Node.js with `sharp`, and Microsoft Word on Windows. Install the Python packages from `requirements_report.txt` in an isolated environment, and make `node` plus the `sharp` package available. This is separate from the experiment requirements. Times New Roman and Arial must be installed for the recorded typography. Node can be selected with the `REPORT_NODE` environment variable; set `NODE_PATH` if `sharp` is outside Node's default module search path.

From this repository's root:

```powershell
python -m pip install -r reports/source/requirements_report.txt
python reports/source/build_report.py
powershell -NoProfile -File reports/source/export_word_pdf.ps1
```

The first script rebuilds `reports/CacheScope_MASC520_Final_Report.docx` and the editable SVG/high-resolution PNG figures. The second exports the PDF with native Microsoft Word, using an independent invisible Word instance and opening only this report read-only. It requires Word's COM automation on Windows. The script closes the report and its own application instance after export. Optional `-DocumentPath` and `-PdfPath` arguments select explicit paths.

## Visual verification

The delivered PDF was exported by Microsoft Word and checked against the DOCX. Its five pages were rasterized at 160 DPI through the document skill's `render_docx.py` PNG pipeline, with the conversion callback transparently replaced by the native Word PDF. Every page was visually inspected for clipping, overflow, split tables, figure readability, and heading placement. The native Word backend was recorded explicitly; LibreOffice was not used. Temporary page PNGs and local QA paths are excluded from this source directory.

After editing, regenerate the PDF, render every page with Poppler (for example `pdftoppm -png -r 160 reports/CacheScope_MASC520_Final_Report.pdf page` in a temporary directory), and inspect all pages again. Page counts may differ with other fonts, Word versions, or rendering backends. Rebuilding the formatting is separate from replaying the frozen experiment; use the repository README for the latter.
