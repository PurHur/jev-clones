# jev-clones

Survey of open replications ("clones") of TypeSafe's **Jev** (System One decision model): typed questions in,
calibrated probabilities out, one forward pass, no decoding.

Goal of phase 1: collect every known clone and compare them **architecture-wise** (backbone, attention mask,
readout head, question types, context handling, calibration, training data, evals, licence).

Context: our own clone is **Zeiger** (github.com/PurHur/zeiger, weights php-ai/zeiger-0.6b), trained in
PurHur/laya-browser.

Layout:
- `clones/<name>.md` – one profile per clone, same template, every claim with a source link
- `COMPARISON.md` – **start here**: the cross-clone architecture comparison (32 profiles, 197 families)
- `sources/` – pinned upstream checkouts (gitignored, data only, never executed)
- `notes/` – raw research notes
- `notes/discovery.md` – all candidates found (HF + GitHub), `notes/hf-sweep.md` – big-lab HF sweep,
  `notes/arch/arch_table.tsv` – config fields of 479 HF repos (`scripts/hf_arch.py`)
