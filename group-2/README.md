# Group 2 — Perceptual Image Quality

## Status and research question

Group 2 investigates whether selecting synthetic images by perceptual quality improves
real-road semantic segmentation when the synthetic budget is fixed at **548 images**.
The repository contains a complete first metric, deterministic selection, shared-interface
tests, a formal full-pool scoring run, and descriptive analysis artifacts. It does **not**
yet contain downstream DeepLabV3+ evidence, so a higher image-quality score must not be
interpreted as better segmentation performance.

The two supplied Group 2 snapshots were audited on 2026-11-05. Their code, tests,
scripts, logs, READMEs and two JSONL result files were byte-identical. The main snapshot
additionally contained the two CSV summaries and the Top/Bottom figure listed below.
The integration therefore keeps one implementation and preserves the main snapshot's
additional artifacts; it does not maintain duplicate scorers or selectors.

## Completed and verified

- Contrast-gradient metric version `1`, including documented RGB decoding, grayscale
  conversion, float precision, native-size processing and score direction.
- Min-max normalization over all successfully scored `eligibility.analysis=true` rows.
- Deterministic Top-548 selection from rows that are also
  `eligibility.segmentation_training=true` (the canonical schema has no
  `eligibility.train` field).
- Shared score JSONL output, fixed-budget subset JSONL output, exact-manifest SHA-256
  provenance, deterministic tie-breaking and seed metadata.
- Relocated image lookup, duplicate-ID rejection, invalid-score rejection, missing or
  corrupt image diagnostics, insufficient-pool failure, output/input collision checks,
  and default protection against overwriting existing formal outputs.
- Automated unit, boundary, CLI, schema, overlay-join and shared dataset-constructor
  tests using temporary fixtures only.
- Full-pool HPC scoring and selection plus a read-only local validation against the
  exact canonical manifest and the shared training reader.
- Top/Bottom examples and source-group eligibility/selection statistics.

### Formal artifacts found in this working tree

| Artifact | Verified content |
| --- | --- |
| `results/fallback-v1/scores.jsonl` | 78,840 analysis-eligible rows; normalized range `[0, 1]`; SHA-256 `3d1410c57b91cc18f2d75d2ab81c9ed3a958a3993155099f66ebdec1449a9693` |
| `results/fallback-v1/perceptual_quality_budget_0.25x.jsonl` | exactly 548 unique, ranked, training-eligible rows; SHA-256 `f63674deb150ab619d05b09bc33429404c24ae29db2d2ed73a61bc2a2ad106c8` |
| `results/fallback-v1/top_bottom_examples.jpg` | valid 1600×1060 JPEG showing the global score pool's 10 highest and 10 lowest examples |
| `results/fallback-v1/source_group_distribution.csv` | selected counts: group3 309, group5 141, group7 69, group9 29 |
| `results/fallback-v1/source_group_eligibility_and_selection.csv` | per-source analysis, training-candidate, selected counts and selection rates |
| `logs/build-manifest-12841196.out` | successful 78,883-row manifest build: 78,840 analysis and 43,800 training eligible |
| `logs/score-quality-12841210.out` | successful full scoring run with 0 skipped images and the same 548-row subset |

The exact manifest used by the formal results is present as
`synthetic-data-seminar/manifest.jsonl`. Its SHA-256 is
`f50ef57736713ff9f9c83950ad453acf4501719358056cd4ad8d4ce2275fdf8e`, matching both
the HPC log and every subset row. The read-only validator confirmed all 78,883 manifest
rows against schema `1.1`, exact score coverage of the 78,840 analysis rows,
raw-to-normalized min-max consistency, the deterministic eligible Top-548 reconstruction,
both CSVs, the figure's file integrity and acceptance of all selected rows by
`SyntheticSegmentationDataset`.

The source images are not available in this local checkout (`datasets/` is absent), so
the 78,840 raw scores could not be recomputed pixel by pixel here. The existing results
remain compatible with the unchanged metric and selection implementation and do not need
regeneration. Pixel-level reproduction requires the original dataset copy.

## Metric and selection definition

Pillow decodes each analysis-eligible image with `convert("RGB")`. The resulting uint8
0–255 array is converted to float64 without resizing, cropping, EXIF transposition or
intensity rescaling. Grayscale is the unweighted arithmetic mean of the three channels:

```python
gray = np.mean(rgb, axis=2)

raw_score = (
    np.std(gray)
    + np.mean(np.abs(np.diff(gray, axis=0)))
    + np.mean(np.abs(np.diff(gray, axis=1)))
)
```

Raw scores are min-max normalized across every successfully scored analysis row:

```text
score = (raw_score - min_raw) / (max_raw - min_raw)
```

Higher is better. If the whole pool is constant, the shared normalizer assigns `1.0` to
every row. Missing, corrupt, non-finite or smaller-than-2×2 images receive no score and
are reported as skipped. Images keep their native dimensions, so resolution and source
style can influence the metric.

Selection retains only records with both canonical eligibility flags true, sorts by
`(-score, synthetic_id)`, and takes exactly 548. Fewer than 548 valid training candidates
is an error; the budget is never silently reduced. `seed=42` is shared metadata only—this
selection algorithm has no random operation. Each subset row records the SHA-256 of the
exact input manifest bytes.

## Setup and tests

Run from the `ws-2026-27/` repository root with Python 3.11 or newer:

```bash
python3 -m venv group-2/.venv
group-2/.venv/bin/python -m pip install numpy Pillow pytest jsonschema
PYTHONDONTWRITEBYTECODE=1 group-2/.venv/bin/python -m pytest \
  -q -p no:cacheprovider \
  group-2/tests tests/test_scoring.py tests/test_selection.py tests/test_manifest.py
```

All generated test images, manifests and outputs live in pytest temporary directories.
The end-to-end fixture creates 550 four-by-four images to exercise the real 548-image
budget without touching the formal artifacts.

Validate the existing formal result bundle without reading source images or modifying
any file:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:group-2/src \
  group-2/.venv/bin/python group-2/scripts/validate_results.py
```

This validator requires `jsonschema` and checks the canonical JSON Schema, score schema,
normalization, exact eligibility coverage, deterministic Top-548 reconstruction, manifest
checksum, shared training-constructor compatibility and any locally present analysis
artifacts.

## Scoring and report commands

For a new or deliberately separate run, point outputs at new paths:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:group-2/src \
  group-2/.venv/bin/python -m group_2.cli \
  --data-root /absolute/path/to/datasets \
  --manifest synthetic-data-seminar/manifest.jsonl \
  --scores-output /tmp/group2-run/scores.jsonl \
  --subset-output /tmp/group2-run/perceptual_quality_budget_0.25x.jsonl
```

`--data-root` is the directory containing `synthetic-data/`. Canonical paths are
relocated from their `synthetic-data/...` suffix; IDs and manifest rows are never
rewritten. Output paths must differ from the manifest and remain outside the data root.
Existing outputs are refused by default. `--overwrite` is available only for an
explicitly intended regeneration; do not use it on the preserved formal results without
first archiving them.

After a full run, generate both CSV summaries and the Top/Bottom figure with:

```bash
group-2/.venv/bin/python group-2/scripts/make_quality_report.py \
  --data-root /absolute/path/to/datasets \
  --output-dir /tmp/group2-report
```

The report script also refuses existing outputs by default. It needs the source images for
the figure, unlike `validate_results.py`.

## Output contracts and shared training compatibility

| Output | Contract |
| --- | --- |
| `scores.jsonl` | Shared score schema `1.0`: `synthetic_id`, `method=perceptual_fallback`, string `method_version=1`, `direction=higher_is_better`, normalized `score`, raw component and preprocessing/normalization provenance. Rows are ordered by ID. |
| `perceptual_quality_budget_0.25x.jsonl` | Exactly 548 rows with `selected=true`, ranks 1–548, `method=perceptual_quality_budget_0.25x`, `budget=budget_0.25x`, seed metadata, source group, score method/version and `manifest_sha256`. |

The shared training code reads subset IDs, joins them to the canonical manifest and passes
the resulting rows to `SyntheticSegmentationDataset`; no image is copied into the subset.
For a later training run, the experiment configuration must use the canonical manifest,
this 548-row subset and `synthetic_samples: 548`.

## Limitations and remaining research work

Contrast and gradients measure visible detail, not human opinion or downstream utility.
Noise, excessive sharpening, extreme contrast, native resolution and source-group style
can raise the score. The global Top-10 visualization is dominated by source group4, which
has no training-eligible records; the actual selected subset uses only eligible groups.
Scores also depend on the normalization pool and cannot be compared directly across
different pools.

The following project work is genuinely incomplete:

1. Add and justify a second independent no-reference quality metric.
2. Calibrate metric direction/scale and perform agreement, correlation and sensitivity
   analyses; paired-reference metrics require proven geometric alignment.
3. Run the downstream DeepLabV3+ experiments for
   `perceptual_quality_budget_0.25x` with seeds 17, 42 and 73.
4. Compare those results with the instructor-provided `real_only` and the exact
   `random_budget_0.25x` baseline on real validation and test splits.
5. Report mean ± standard deviation and analyze whether quality selection changes
   segmentation performance.

Optional engineering improvements such as parallel decoding, resumable score caches and
additional plots are lower priority than the research steps above.
