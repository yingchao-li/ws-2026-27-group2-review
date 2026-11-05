# Group 2 results

`fallback-v1/` contains the formal first-metric full-pool outputs and, in this working
tree, the accompanying descriptive artifacts. Do not replace these files with fixture or
small-sample results.

| File | Content |
| --- | --- |
| `scores.jsonl` | 78,840 formal normalized score rows |
| `perceptual_quality_budget_0.25x.jsonl` | deterministic 548-row training subset |
| `top_bottom_examples.jpg` | global Top/Bottom-10 visual inspection sheet |
| `source_group_distribution.csv` | selected-source counts and fractions |
| `source_group_eligibility_and_selection.csv` | analysis/training/selection counts by source |

The JSONL files are tracked in the supplied main project. The CSV/JPEG files may be
ignored by the repository's broad generated-results rule even when they exist locally, so
their presence must be checked rather than assumed in another clone.

Run the read-only integrity and compatibility check from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:group-2/src \
  group-2/.venv/bin/python group-2/scripts/validate_results.py
```

Formal provenance, counts and hashes are documented in `group-2/README.md` and in the
HPC logs under `group-2/logs/`.

The NIQE/BRISQUE implementation targets separate `niqe-v1/` and `brisque-v1/`
directories. Those directories are not formal results until an HPC run has scanned the
complete analysis pool and written `completion.json`. A smoke-test checkpoint or a
`--max-images` run must never be copied or renamed into a formal result directory.
