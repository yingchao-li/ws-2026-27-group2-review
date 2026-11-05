# Group 2 — Perceptual Image Quality

## Research question

With a fixed budget of **548 synthetic images**, does perceptual-quality selection improve
semantic segmentation on real road scenes?

The downstream protocol is DeepLabV3+ with seeds 17, 42 and 73, compared with
`real_only` and `random_budget_0.25x`. Downstream training has not been run yet.

## Methods and results

| Method | Description | Current result |
| --- | --- | --- |
| 1. Contrast-gradient fallback | Grayscale standard deviation plus horizontal and vertical absolute gradients; higher is better | **Complete:** 78,840 scores, 0 skipped, deterministic Top-548 |
| 2. NIQE | pyiqa no-reference natural-scene-statistics metric; raw score is lower-is-better | Code and real HPC smoke test complete; **no full result yet** |
| 3. BRISQUE | pyiqa no-reference spatial natural-scene-statistics metric; raw score is lower-is-better | Code and real HPC smoke test complete; **no full result yet** |

The completed fallback score is:

```python
gray = np.mean(rgb, axis=2)
raw_score = (
    np.std(gray)
    + np.mean(np.abs(np.diff(gray, axis=0)))
    + np.mean(np.abs(np.diff(gray, axis=1)))
)
```

Scores are min-max normalized to `[0, 1]`. Selection keeps records with both
`eligibility.analysis=true` and `eligibility.segmentation_training=true`, sorts by
`(-score, synthetic_id)`, and selects exactly 548 images.

NIQE and BRISQUE use `pyiqa==0.1.16`. Their raw values are saved first and converted to
normalized higher-is-better scores only after the complete image pool has been processed.
Both real HPC smoke tests processed 8/8 images with zero skips. Full results are not yet
available because the current safe one-image-at-a-time pipeline is slow: processing all
78,840 images is expected to take **tens of hours** and may require several resumed HPC
jobs. Smoke outputs are not reported as research results.

## Completed fallback result

| Output | Result |
| --- | --- |
| `results/fallback-v1/scores.jsonl` | 78,840 normalized scores |
| `results/fallback-v1/perceptual_quality_budget_0.25x.jsonl` | 548 unique selected IDs |
| `results/fallback-v1/top_bottom_examples.jpg` | Global Top-10/Bottom-10 visualization |
| `results/fallback-v1/source_group_distribution.csv` | group3: 309; group5: 141; group7: 69; group9: 29 |
| `results/fallback-v1/source_group_eligibility_and_selection.csv` | Eligibility and selection rates for groups 1–9 |

![Fallback Top-10 and Bottom-10](results/fallback-v1/top_bottom_examples.jpg)

These are image-quality selection results, not segmentation-performance results.

## Project structure

### Folders

| Folder | Purpose |
| --- | --- |
| `src/group_2/` | Scoring and selection code |
| `scripts/` | Validation and analysis entry points |
| `scripts/hpc/` | NIQE/BRISQUE HPC execution |
| `tests/` | Automated tests |
| `results/fallback-v1/` | Completed formal fallback results |
| `logs/` | Historical Slurm logs |

### Files

| File | Purpose |
| --- | --- |
| `README.md` | English project summary |
| `readme-chinese.md` | Chinese project summary |
| `HPC_NIQE_BRISQUE.md` | HPC run instructions |
| `requirements-iqa.txt` | NIQE/BRISQUE dependencies |
| `.gitignore` | Local IQA environment exclusions |
| `src/group_2/perceptual_quality.py` | Fallback score and Top-548 |
| `src/group_2/cli.py` | Fallback CLI |
| `src/group_2/nr_iqa.py` | NIQE/BRISQUE pipeline |
| `src/group_2/nr_iqa_cli.py` | NIQE/BRISQUE CLI |
| `src/group_2/metric_comparison.py` | Metric comparison statistics and plots |
| `scripts/build_manifest.slurm` | Manifest generation job |
| `scripts/score_quality.slurm` | Fallback scoring job |
| `scripts/make_quality_report.py` | Fallback figure and CSV report |
| `scripts/validate_results.py` | Formal-result validation |
| `scripts/compare_metrics.py` | Three-metric comparison CLI |
| `scripts/hpc/run_nr_iqa.sh` | Shared HPC runtime |
| `scripts/hpc/score_niqe.slurm` | NIQE Slurm template |
| `scripts/hpc/score_brisque.slurm` | BRISQUE Slurm template |
| `tests/test_group_2.py` | Fallback and shared-interface tests |
| `tests/test_nr_iqa.py` | NIQE/BRISQUE and comparison tests |
| `results/README.md` | Result-status rules |
| `results/fallback-v1/scores.jsonl` | Formal fallback scores |
| `results/fallback-v1/perceptual_quality_budget_0.25x.jsonl` | Formal fallback Top-548 |
| `results/fallback-v1/top_bottom_examples.jpg` | Fallback visualization |
| `results/fallback-v1/source_group_distribution.csv` | Selected-source distribution |
| `results/fallback-v1/source_group_eligibility_and_selection.csv` | Per-source selection statistics |
| `logs/build-manifest-12841194.out` | Failed first manifest attempt |
| `logs/build-manifest-12841196.out` | Successful manifest run |
| `logs/score-quality-12841210.out` | Successful fallback run |

## Validation

- Local automated tests: **39 passed, 2 skipped**.
- NIQE HPC smoke: **8 valid, 0 skipped**.
- BRISQUE HPC smoke: **8 valid, 0 skipped**.
- Formal fallback validation: **78,840 scores and exactly 548 selected records**.

## Next steps

1. Resume NIQE and BRISQUE until all 78,840 images are processed.
2. Generate their normalized scores, Top-548 subsets and comparison plots.
3. Compare correlations, Top-548 overlap and source-group selection rates.
4. Train DeepLabV3+ with seeds 17, 42 and 73.
5. Compare with `real_only` and `random_budget_0.25x` on real validation/test data.

Quality-score improvement or agreement between IQA metrics must not be interpreted as
segmentation improvement before the downstream experiments are complete.
