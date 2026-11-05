"""Compare normalized higher-is-better IQA overlays on explicit ID sets."""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sensation_seminar.synthetic_data.scoring import validate_score_overlay

from group_2.perceptual_quality import _index_records, select_ranked_subset


OUTPUT_NAMES = (
    "summary.json",
    "correlations.csv",
    "top548_overlap.csv",
    "score_distributions.csv",
    "source_group_selection_rates.csv",
    "score_scatter.png",
    "score_distributions.png",
    "correlation_heatmap.png",
    "top548_overlap.png",
    "source_group_selection_rates.png",
)


def _read_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = validate_score_overlay(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"Score overlay is empty: {path}")
    methods = {(row["method"], row["method_version"]) for row in rows}
    if len(methods) != 1:
        raise ValueError(f"Score overlay mixes methods or versions: {path}")
    return rows


def _statistic(result: Any) -> float | None:
    value = result.statistic if hasattr(result, "statistic") else result[0]
    value = float(value)
    return value if math.isfinite(value) else None


def compute_correlations(
    score_maps: Mapping[str, Mapping[str, float]],
    common_ids: list[str],
    *,
    statistics: Any | None = None,
) -> list[dict[str, Any]]:
    """Compute all correlations on one global common-ID set."""
    if statistics is None:
        try:
            from scipy import stats as statistics
        except ImportError as error:
            raise RuntimeError(
                "Correlation analysis requires scipy; install group-2/requirements-iqa.txt"
            ) from error
    if len(common_ids) < 2:
        raise ValueError("At least two common valid image IDs are required")
    rows = []
    for left, right in combinations(sorted(score_maps), 2):
        x = np.asarray([score_maps[left][key] for key in common_ids], dtype=np.float64)
        y = np.asarray([score_maps[right][key] for key in common_ids], dtype=np.float64)
        rows.append(
            {
                "metric_a": left,
                "metric_b": right,
                "common_count": len(common_ids),
                "pearson": _statistic(statistics.pearsonr(x, y)),
                "spearman": _statistic(statistics.spearmanr(x, y)),
                "kendall_tau": _statistic(statistics.kendalltau(x, y)),
            }
        )
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _distribution_rows(score_maps: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(score_maps):
        values = np.asarray(list(score_maps[name].values()), dtype=np.float64)
        quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        rows.append(
            {
                "metric": name,
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "max": float(quantiles[4]),
            }
        )
    return rows


def _generate_plots(
    output_dir: Path,
    score_maps: Mapping[str, Mapping[str, float]],
    common_ids: list[str],
    correlations: list[Mapping[str, Any]],
    overlap_rows: list[Mapping[str, Any]],
    source_rows: list[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires matplotlib; install group-2/requirements-iqa.txt"
        ) from error

    names = sorted(score_maps)
    pairs = list(combinations(names, 2))
    figure, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
    for axis, (left, right) in zip(axes[0], pairs):
        axis.scatter(
            [score_maps[left][key] for key in common_ids],
            [score_maps[right][key] for key in common_ids],
            s=5,
            alpha=0.3,
        )
        axis.set(xlabel=left, ylabel=right, title=f"{left} vs {right} (n={len(common_ids)})")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "score_scatter.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for name in names:
        axis.hist(list(score_maps[name].values()), bins=50, alpha=0.45, density=True, label=name)
    axis.set(xlabel="normalized higher-is-better score", ylabel="density", title="Score distributions")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "score_distributions.png", dpi=180)
    plt.close(figure)

    matrix = np.eye(len(names), dtype=np.float64)
    for row in correlations:
        i, j = names.index(row["metric_a"]), names.index(row["metric_b"])
        matrix[i, j] = matrix[j, i] = np.nan if row["pearson"] is None else row["pearson"]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title("Pearson correlation on global common-ID set")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_heatmap.png", dpi=180)
    plt.close(figure)

    labels = [f"{row['metric_a']}\nvs\n{row['metric_b']}" for row in overlap_rows]
    values = [row["overlap_count"] for row in overlap_rows]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 2), 4))
    axis.bar(labels, values)
    axis.axhline(548, color="black", linestyle="--", linewidth=1)
    axis.set(ylabel="Top-548 overlap count", title="Top-548 overlap")
    figure.tight_layout()
    figure.savefig(output_dir / "top548_overlap.png", dpi=180)
    plt.close(figure)

    groups = sorted({row["source_group"] for row in source_rows})
    width = 0.8 / len(names)
    x = np.arange(len(groups))
    figure, axis = plt.subplots(figsize=(max(8, len(groups)), 5))
    for index, name in enumerate(names):
        by_group = {
            row["source_group"]: row["selection_rate"]
            for row in source_rows
            if row["metric"] == name
        }
        axis.bar(x + index * width, [by_group.get(group, 0.0) for group in groups], width, label=name)
    axis.set_xticks(x + width * (len(names) - 1) / 2, groups, rotation=30, ha="right")
    axis.set(ylabel="selected / valid training candidates", title="Source-group Top-548 selection rates")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "source_group_selection_rates.png", dpi=180)
    plt.close(figure)


def analyze_metric_scores(
    manifest: list[Mapping[str, Any]],
    score_paths: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    manifest_sha256: str,
    overwrite: bool = False,
    create_plots: bool = True,
    statistics: Any | None = None,
) -> dict[str, Any]:
    """Write comparison tables and optional plots without changing input artifacts."""
    if len(score_paths) < 2:
        raise ValueError("At least two score overlays are required for comparison")
    output = Path(output_dir)
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing comparison output(s): "
            + ", ".join(str(path) for path in existing)
        )
    if create_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Plot generation requires matplotlib; install group-2/requirements-iqa.txt"
            ) from error

    indexed = _index_records(manifest)
    analysis_ids = {
        key
        for key, row in indexed.items()
        if row.get("eligibility", {}).get("analysis") is True
    }
    score_rows: dict[str, list[dict[str, Any]]] = {}
    score_maps: dict[str, dict[str, float]] = {}
    method_metadata: dict[str, tuple[str, str]] = {}
    for name, value in sorted(score_paths.items()):
        rows = _read_scores(Path(value))
        unknown = {row["synthetic_id"] for row in rows} - set(indexed)
        if unknown:
            raise ValueError(f"{name} references {len(unknown)} IDs outside the manifest")
        score_rows[name] = rows
        score_maps[name] = {row["synthetic_id"]: float(row["score"]) for row in rows}
        method_metadata[name] = (str(rows[0]["method"]), str(rows[0]["method_version"]))

    common_ids = sorted(
        set.intersection(*(set(values) for values in score_maps.values())) & analysis_ids
    )
    correlations = compute_correlations(score_maps, common_ids, statistics=statistics)
    distributions = _distribution_rows(score_maps)

    top_sets: dict[str, set[str]] = {}
    source_rows: list[dict[str, Any]] = []
    groups = sorted({str(row["source_group"]) for row in indexed.values()})
    for name in sorted(score_rows):
        method, version = method_metadata[name]
        subset = select_ranked_subset(
            indexed.values(),
            score_rows[name],
            manifest_sha256=manifest_sha256,
            score_method=method,
            score_method_version=version,
            subset_method=f"comparison_only_{name}_budget_0.25x",
        )
        selected = {row["synthetic_id"] for row in subset}
        top_sets[name] = selected
        valid_ids = set(score_maps[name])
        for group in groups:
            candidates = {
                key
                for key, row in indexed.items()
                if key in valid_ids
                and row["source_group"] == group
                and row.get("eligibility", {}).get("analysis") is True
                and row.get("eligibility", {}).get("segmentation_training") is True
            }
            selected_count = len(selected & candidates)
            source_rows.append(
                {
                    "metric": name,
                    "source_group": group,
                    "valid_training_candidates": len(candidates),
                    "selected_count": selected_count,
                    "selection_rate": selected_count / len(candidates) if candidates else 0.0,
                }
            )

    overlap_rows = []
    for left, right in combinations(sorted(top_sets), 2):
        intersection = top_sets[left] & top_sets[right]
        union = top_sets[left] | top_sets[right]
        overlap_rows.append(
            {
                "metric_a": left,
                "metric_b": right,
                "overlap_count": len(intersection),
                "overlap_fraction_of_548": len(intersection) / 548,
                "jaccard": len(intersection) / len(union),
            }
        )

    union_ids = set.union(*(set(values) for values in score_maps.values()))
    coverage = {
        name: {
            "score_count": len(values),
            "missing_from_global_union_count": len(union_ids - set(values)),
            "missing_from_manifest_analysis_count": len(analysis_ids - set(values)),
            "extra_outside_manifest_analysis_count": len(set(values) - analysis_ids),
            "missing_from_manifest_analysis_ids": sorted(analysis_ids - set(values)),
            "extra_outside_manifest_analysis_ids": sorted(set(values) - analysis_ids),
        }
        for name, values in sorted(score_maps.items())
    }
    summary = {
        "schema_version": "1.0",
        "score_basis": "normalized higher-is-better score overlays; raw metric values are not mixed",
        "correlation_id_policy": "global intersection shared by every supplied metric",
        "distribution_id_policy": "each metric's complete valid score pool",
        "top548_id_policy": "each metric's valid analysis-and-training-eligible pool",
        "manifest_sha256": manifest_sha256,
        "metrics": sorted(score_maps),
        "global_common_valid_count": len(common_ids),
        "global_union_count": len(union_ids),
        "coverage": coverage,
        "correlations": correlations,
        "top548_overlap": overlap_rows,
        "warning": "IQA correlation or Top-548 overlap is not evidence of segmentation improvement",
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    _write_csv(
        output / "correlations.csv",
        ["metric_a", "metric_b", "common_count", "pearson", "spearman", "kendall_tau"],
        correlations,
    )
    _write_csv(
        output / "top548_overlap.csv",
        ["metric_a", "metric_b", "overlap_count", "overlap_fraction_of_548", "jaccard"],
        overlap_rows,
    )
    _write_csv(
        output / "score_distributions.csv",
        ["metric", "count", "mean", "std", "min", "q25", "median", "q75", "max"],
        distributions,
    )
    _write_csv(
        output / "source_group_selection_rates.csv",
        ["metric", "source_group", "valid_training_candidates", "selected_count", "selection_rate"],
        source_rows,
    )
    if create_plots:
        _generate_plots(output, score_maps, common_ids, correlations, overlap_rows, source_rows)
    return summary
