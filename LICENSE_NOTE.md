# License and provenance inventory

No new repository-wide license has been selected. This note does not grant additional rights to the original project code, data or report.

The bundled tokenizer files originate from `Qwen/Qwen2.5-Coder-0.5B-Instruct`, revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`. Their source URL, exact file hashes and retained Apache License 2.0 are recorded in each experiment's `assets/tokenizer_source.json` and `assets/LICENSE`. Those third-party notices remain intact.

The large answering-model weights are not bundled in this repository. Answer labeling used a separately installed local Ollama model with the exact digest documented in each experiment. The small files under `models/` are the locally trained risk predictors, accompanied by their training records and frozen manifests.

Python dependencies retain their respective upstream licenses. They are installed from `requirements.txt`, not copied as vendored libraries. Original synthetic source facts, code and documentation were created with AI assistance, as disclosed in the experiment reports.

The original course syllabus and personal proposal are not included as public artifacts. `docs/requirements_summary.md` paraphrases the applicable deliverables without republishing the course document.
