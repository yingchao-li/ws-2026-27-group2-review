# Task 1 feedback response — Group 2

Decision recorded on 2026-11-10, before downstream training/test evaluation.

| Feedback | Action and evidence |
| --- | --- |
| Required: primary metric | Contrast-gradient fallback v1 defines `perceptual_quality_budget_0.25x`; the existing 548-image subset is fixed. |
| Required: score conversion and preprocessing | Both READMEs specify RGB range/dtype, geometry/alpha handling, full-pool normalization, constant/empty cases and failed-image policy; supplementary library operations are linked to pinned pyiqa sources. |
| Clarify three selectors/composite | NIQE and BRISQUE are independent supplementary comparisons, never a composite or test-based replacement for the primary method. |
| Unequal component scales | Keep coefficients `(1,1,1)` and no individual normalization in v1; acknowledge possible dominance. A changed formula would be a new experiment. |
| Noise/halos/oversharpening | Explicit limitation: edge/contrast energy is a proxy, not a validated perceptual judgment. Real-image artifact inspection remains future work on HPC. |
| Recommended: scene/variant/source audit | Executed on the complete existing Manifest and formal fallback subset; results below and in `analysis/task1-feedback-v1/`. |
| Optional: robust scaling / metric agreement | Robust scaling deferred; metric comparison tooling already exists, but real cross-metric analysis awaits full NIQE/BRISQUE results. No proxy outputs substituted. |

## Executed audit

The audit reconstructs the exact Top-548 using the shared existing Group 2 selector,
checks complete analysis score coverage, records SHA-256 for all three inputs and
compares the selected distribution to **training-eligible candidates**, not all images.
Source folder means the immediate parent of `source_relpath`; `original_id` identifies
the underlying real scene across groups; variant is reported within its source group.

| Quantity | Training candidates | Selected |
| --- | ---: | ---: |
| Images | 43,800 | 548 |
| Unique original scenes | 2,190 | 298 |
| Original scenes contributing multiple images | 2,190 | 102 |
| Images belonging to repeated original scenes | 43,800 | 352 |
| Images beyond one per original | 41,610 | 250 |
| Maximum images per original | 20 | 7 |
| Unique group/original pairs | 10,950 | 325 |
| Unique group/original/variant triples | 43,800 | 548 |
| Images with content hashes | 0 | 0 |

There are 548 distinct canonical IDs. Repeated original scenes are not identical-image
duplicates; exact content duplication cannot be assessed without images or hashes.
The 10 most represented selected scenes contribute 53/548 images (9.67%).

| Source group | Candidates | Selected | Selected share | Selection rate | Selected share / candidate share |
| --- | ---: | ---: | ---: | ---: | ---: |
| group3 | 8,760 | 309 | 56.39% | 3.53% | 2.82 |
| group5 | 8,760 | 141 | 25.73% | 1.61% | 1.29 |
| group7 | 8,760 | 69 | 12.59% | 0.79% | 0.63 |
| group8 | 8,760 | 0 | 0.00% | 0.00% | 0.00 |
| group9 | 8,760 | 29 | 5.29% | 0.33% | 0.26 |

Each group contributes 20% of training candidates. Groups outside this table have no
training-eligible candidates. Source-group HHI is 0.4028 versus 0.2 for the balanced
candidate pool. This is descriptive evidence of concentration, not its causal explanation.
For example, group5 variants 00/01 contribute 70/71 selected images, while 02/03 contribute
zero. Group3 variant counts are 70/75/76/88. Full variant/folder counts are in the CSVs.

Global Top-548 confounds score preference with source, scene and variant composition.
No per-scene cap, content deduplication or group quota is introduced into v1. Interpretation
of downstream results must account for this limitation; any balanced/deduplicated method
would require a separately declared experiment before test evaluation.

Existing scores retain the summed raw contrast-gradient value, not the three separate
component contributions. Actual component dominance and artifact sensitivity cannot be
measured from that sum alone; this requires source-image inspection/recomputation on
HPC. We state this as a limitation, not as an empirically measured finding.

## Reproduce and validate

From the repository root, using the README environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/group-2/src"
group-2/.venv/bin/python group-2/scripts/audit_selection.py \
  --output-dir group-2/analysis/task1-feedback-reproduction
group-2/.venv/bin/python group-2/scripts/validate_results.py \
  --manifest group-2/manifest.jsonl
group-2/.venv/bin/python -m pytest -q -p no:cacheprovider \
  group-2/tests tests/test_scoring.py tests/test_selection.py tests/test_manifest.py
```

Outputs: `summary.json`, `source_groups.csv`, `source_folders.csv`, `original_scenes.csv`
and `variants.csv`. Existing output directories are rejected; no image loading or
rescoring occurs. Local tests: 45 passed, 2 opt-in real-pyiqa tests skipped. Real NIQE/
BRISQUE full runs and DeepLabV3+ results remain unavailable. The full fallback validator
checks schema 1.1, normalization, 78,840 score IDs, exact Top-548 and the shared dataset
constructor; it does not load every training image/mask or establish training performance.

## 中文回应

已完成老师的两项必改：固定 Contrast-gradient v1 为主要方法，并按真实实现补全各指标
的归一化、预处理和异常规则。NIQE/BRISQUE为独立补充比较，不按测试成绩更换主方法。
也完成建议的重复度与来源分析：548张来自298个原始场景，102个场景重复入选，最多7张；
group3从候选占比20%上升到入选占比56.39%，group8未入选。没有内容哈希，无法声称
已经完成像素内容去重。不同分量尺度、噪声/锐化偏好和来源混杂已列为主方法局限。
保留现有 v1 公式和入选集。可选稳健归一化暂缓，跨指标实测比较等待全量分数。
下一步是HPC图片检查及固定主方法的三种子DeepLabV3+比较；这些尚未执行。
