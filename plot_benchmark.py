#!/usr/bin/env python3
"""Plot paper-weighted benchmark metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from evaluate_outputs import match_references, normalize_gold_record, normalize_pred_record


DEFAULT_OUTPUT_ROOT = Path("benchmark_runs")
DEFAULT_TITLE_THRESHOLD = 0.88
DEFAULT_RAW_THRESHOLD = 0.82
DEFAULT_PIPELINE_ORDER = ["direct_pdf", "pymupdf_markdown", "pdf_images"]
PIPELINE_LABELS = {
    "direct_pdf": "Direct PDF",
    "pymupdf_markdown": "PyMuPDF Markdown",
    "pdf_images": "PDF Images",
}
PIPELINE_SHORT_LABELS = {
    "direct_pdf": "Direct",
    "pymupdf_markdown": "Markdown",
    "pdf_images": "Images",
}
MODEL_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
]
LEGACY_PLOT_STEMS = [
    "accuracy_overview",
    "field_f1_heatmap",
    "identifier_accuracy",
    "reference_counts",
    "runtime_tokens",
    "paper_heatmaps",
    "run_health",
]
GENERATED_PLOT_STEMS = [
    "average_f1_per_paper",
    "average_precision_per_paper",
    "average_recall_per_paper",
    "average_wall_clock_time_per_page",
    "average_error_rate",
    "average_input_output_tokens_per_page",
    "average_cost_per_page",
    "reference_counts_by_pipeline",
    "reference_counts_by_model",
]

GroupKey = tuple[str, str]


class PlottingError(Exception):
    """Raised when benchmark plots cannot be generated."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PlottingError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def find_latest_run(output_root: Path) -> Path:
    if not output_root.exists():
        raise PlottingError(f"Run root does not exist: {output_root}")
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise PlottingError(f"No benchmark runs found under {output_root}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def load_inputs(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_path = run_dir / "metrics.jsonl"
    scores_path = run_dir / "scores.jsonl"

    if not metrics_path.exists():
        raise PlottingError(f"Missing benchmark metrics file: {metrics_path}")
    if not scores_path.exists():
        raise PlottingError(
            f"Missing accuracy file: {scores_path}\n"
            f"Run: uv run evaluate-benchmark --run-dir {run_dir}"
        )

    return read_jsonl(metrics_path), read_jsonl(scores_path)


def parse_list(values: list[str]) -> list[str]:
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if not item or item in seen:
                continue
            parsed.append(item)
            seen.add(item)
    return parsed


def pipeline_label(pipeline: str) -> str:
    return PIPELINE_LABELS.get(pipeline, pipeline)


def normalize_model_id(provider: str | None, model: str) -> str:
    model_id = model.strip()
    if provider == "gemini":
        model_id = model_id.removeprefix("models/")
    model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.lower()).strip("._-")
    return model_id or "model"


def row_model_id(row: dict[str, Any]) -> str:
    model_id = row.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id
    model = row.get("model")
    provider = row.get("provider") if isinstance(row.get("provider"), str) else None
    if isinstance(model, str) and model.strip():
        return normalize_model_id(provider, model)
    return "default"


def row_model_label(row: dict[str, Any]) -> str:
    model = row.get("model")
    if isinstance(model, str) and model.strip():
        return model
    return row_model_id(row)


def row_pipeline(row: dict[str, Any]) -> str | None:
    pipeline = row.get("pipeline_id")
    return pipeline if isinstance(pipeline, str) and pipeline.strip() else None


def paper_id(row: dict[str, Any]) -> str | None:
    value = row.get("paper_id")
    if value is not None:
        return str(value)
    pdf = row.get("pdf")
    if pdf:
        return Path(str(pdf)).stem
    return None


def group_key(row: dict[str, Any]) -> GroupKey | None:
    pipeline = row_pipeline(row)
    if not pipeline:
        return None
    return (pipeline, row_model_id(row))


def enrich_score_models(scores: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_by_pipeline_paper: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        pipeline = row_pipeline(row)
        paper = paper_id(row)
        if pipeline and paper:
            metrics_by_pipeline_paper[(pipeline, paper)].append(row)

    enriched: list[dict[str, Any]] = []
    for score in scores:
        if score.get("model_id") or score.get("model"):
            enriched.append(score)
            continue

        pipeline = row_pipeline(score)
        paper = paper_id(score)
        matches = metrics_by_pipeline_paper.get((pipeline or "", paper or ""))
        unique_model_ids = {row_model_id(row) for row in matches}
        if len(unique_model_ids) != 1:
            enriched.append(score)
            continue

        metric = matches[0]
        updated = dict(score)
        updated["provider"] = metric.get("provider")
        updated["model"] = metric.get("model")
        updated["model_id"] = row_model_id(metric)
        enriched.append(updated)
    return enriched


def ordered_pipelines(rows: list[dict[str, Any]]) -> list[str]:
    available = {pipeline for row in rows if (pipeline := row_pipeline(row))}
    known = [pipeline for pipeline in DEFAULT_PIPELINE_ORDER if pipeline in available]
    extra = sorted(pipeline for pipeline in available if pipeline not in DEFAULT_PIPELINE_ORDER)
    return known + extra


def ordered_models(rows: list[dict[str, Any]]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for row in rows:
        model_id = row_model_id(row)
        if model_id in seen:
            continue
        models.append(model_id)
        seen.add(model_id)
    return models


def model_matches(row: dict[str, Any], requested_models: list[str]) -> bool:
    if not requested_models:
        return True
    model_id = row_model_id(row)
    model = row.get("model")
    provider = row.get("provider") if isinstance(row.get("provider"), str) else None
    for requested_model in requested_models:
        if requested_model == model_id or requested_model == model:
            return True
        if normalize_model_id(provider, requested_model) == model_id:
            return True
    return False


def filter_rows(
    rows: list[dict[str, Any]],
    pipelines: list[str],
    requested_models: list[str],
) -> list[dict[str, Any]]:
    pipeline_set = set(pipelines)
    return [
        row
        for row in rows
        if row_pipeline(row) in pipeline_set and model_matches(row, requested_models)
    ]


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        result = float(value)
        return None if math.isnan(result) else result
    return None


def positive_number(value: Any) -> float | None:
    result = numeric(value)
    if result is None or result <= 0:
        return None
    return result


def rows_by_group(rows: list[dict[str, Any]]) -> dict[GroupKey, list[dict[str, Any]]]:
    grouped: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = group_key(row)
        if key:
            grouped[key].append(row)
    return grouped


def collect_page_counts(metrics: list[dict[str, Any]], warnings: list[str]) -> dict[str, float]:
    page_counts: dict[str, float] = {}
    pdfs_by_paper: dict[str, Path] = {}

    for row in metrics:
        paper = paper_id(row)
        if paper is None:
            continue
        if row.get("pdf"):
            pdfs_by_paper.setdefault(paper, Path(str(row["pdf"])))
        pages = positive_number(row.get("num_pages"))
        if pages is None:
            continue
        existing = page_counts.get(paper)
        if existing is not None and existing != pages:
            warnings.append(
                f"Conflicting page counts for {paper}: {existing:g} and {pages:g}; using {existing:g}."
            )
            continue
        page_counts[paper] = pages

    missing_papers = sorted(set(pdfs_by_paper) - set(page_counts))
    if not missing_papers:
        return page_counts

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        for paper in missing_papers:
            warnings.append(f"No page count found for {paper}; PyMuPDF fallback unavailable.")
        return page_counts

    for paper in missing_papers:
        pdf_path = pdfs_by_paper[paper]
        if not pdf_path.exists():
            warnings.append(f"No page count found for {paper}; PDF path does not exist: {pdf_path}")
            continue
        try:
            with fitz.open(pdf_path) as doc:
                if doc.page_count > 0:
                    page_counts[paper] = float(doc.page_count)
                else:
                    warnings.append(f"No positive page count found for {paper}: {pdf_path}")
        except Exception as exc:  # pragma: no cover - defensive around local PDFs
            warnings.append(f"Could not read page count for {paper} from {pdf_path}: {exc}")

    return page_counts


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_lookup(scores: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in scores:
        key = group_key(row)
        paper = paper_id(row)
        if key and paper:
            lookup[(key[0], key[1], paper)] = row
    return lookup


def metric_lookup(metrics: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in metrics:
        key = group_key(row)
        paper = paper_id(row)
        if key and paper:
            lookup[(key[0], key[1], paper)] = row
    return lookup


def ordered_papers_for_plot(metrics: list[dict[str, Any]], scores: list[dict[str, Any]]) -> list[str]:
    papers = {
        paper
        for row in metrics + scores
        if (paper := paper_id(row)) is not None
    }
    return sorted(papers)


def papers_for_group(key: GroupKey, metric_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> list[str]:
    papers = {
        paper
        for row in metric_rows + score_rows
        if (paper := paper_id(row)) is not None
    }
    return sorted(papers)


def paper_weighted_accuracy(
    groups: list[GroupKey],
    metrics_by_group: dict[GroupKey, list[dict[str, Any]]],
    scores_by_group: dict[GroupKey, list[dict[str, Any]]],
    scores: list[dict[str, Any]],
    metric_name: str,
    warnings: list[str],
) -> dict[GroupKey, float | None]:
    scores_by_key = score_lookup(scores)
    values_by_group: dict[GroupKey, float | None] = {}

    for key in groups:
        values: list[float] = []
        for paper in papers_for_group(key, metrics_by_group.get(key, []), scores_by_group.get(key, [])):
            score = scores_by_key.get((key[0], key[1], paper))
            if score is None:
                warnings.append(f"No score found for {key[0]}/{key[1]}/{paper}; counted as 0 for {metric_name}.")
                values.append(0.0)
                continue
            value = numeric(score.get(metric_name))
            if value is None:
                warnings.append(f"Missing {metric_name} for {key[0]}/{key[1]}/{paper}; counted as 0.")
                values.append(0.0)
                continue
            values.append(value)
        values_by_group[key] = mean(values)

    return values_by_group


def average_wall_clock_seconds_per_page(
    groups: list[GroupKey],
    metrics_by_group: dict[GroupKey, list[dict[str, Any]]],
    page_counts: dict[str, float],
    warnings: list[str],
) -> dict[GroupKey, float | None]:
    values_by_group: dict[GroupKey, float | None] = {}

    for key in groups:
        values: list[float] = []
        for row in metrics_by_group.get(key, []):
            paper = paper_id(row)
            seconds = numeric(row.get("wall_clock_seconds"))
            pages = page_counts.get(paper or "")
            if paper is None or seconds is None:
                warnings.append(f"Missing wall-clock time for a {key[0]}/{key[1]} row; excluded from time/page.")
                continue
            if pages is None:
                warnings.append(f"No page count for {key[0]}/{key[1]}/{paper}; excluded from time/page.")
                continue
            values.append(seconds / pages)
        values_by_group[key] = mean(values)

    return values_by_group


def error_rates(groups: list[GroupKey], metrics_by_group: dict[GroupKey, list[dict[str, Any]]]) -> dict[GroupKey, float | None]:
    values_by_group: dict[GroupKey, float | None] = {}
    for key in groups:
        rows = metrics_by_group.get(key, [])
        if not rows:
            values_by_group[key] = None
            continue
        error_count = sum(1 for row in rows if row.get("error"))
        values_by_group[key] = 100 * error_count / len(rows)
    return values_by_group


def average_tokens_per_page(
    groups: list[GroupKey],
    metrics_by_group: dict[GroupKey, list[dict[str, Any]]],
    page_counts: dict[str, float],
    warnings: list[str],
) -> dict[GroupKey, dict[str, float | None]]:
    values_by_group: dict[GroupKey, dict[str, float | None]] = {}

    for key in groups:
        input_values: list[float] = []
        output_values: list[float] = []
        for row in metrics_by_group.get(key, []):
            paper = paper_id(row)
            pages = page_counts.get(paper or "")
            input_tokens = numeric(row.get("input_tokens"))
            output_tokens = numeric(row.get("output_tokens"))
            if paper is None or pages is None:
                warnings.append(f"No page count for a {key[0]}/{key[1]} token row; excluded from tokens/page.")
                continue
            if input_tokens is None or output_tokens is None:
                warnings.append(f"Token usage missing for {key[0]}/{key[1]}/{paper}; excluded from tokens/page.")
                continue
            input_values.append(input_tokens / pages)
            output_values.append(output_tokens / pages)
        values_by_group[key] = {
            "input_tokens_per_page": mean(input_values),
            "output_tokens_per_page": mean(output_values),
        }

    return values_by_group


def average_cost_per_page(
    groups: list[GroupKey],
    metrics_by_group: dict[GroupKey, list[dict[str, Any]]],
    page_counts: dict[str, float],
    warnings: list[str],
) -> dict[GroupKey, float | None]:
    values_by_group: dict[GroupKey, float | None] = {}

    for key in groups:
        values: list[float] = []
        for row in metrics_by_group.get(key, []):
            paper = paper_id(row)
            pages = page_counts.get(paper or "")
            cost = numeric(row.get("estimated_cost_usd"))
            if paper is None or pages is None:
                warnings.append(f"No page count for a {key[0]}/{key[1]} cost row; excluded from cost/page.")
                continue
            if cost is None:
                warnings.append(f"Estimated cost missing for {key[0]}/{key[1]}/{paper}; excluded from cost/page.")
                continue
            values.append(cost / pages)
        values_by_group[key] = mean(values)

    return values_by_group


def resolve_run_path(run_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    return path


def read_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_correct_reference_set(
    output_path: Path,
    gold_path: Path,
    fallback_paper_id: str | None,
    title_threshold: float,
    raw_threshold: float,
    warnings: list[str],
) -> set[str]:
    if not gold_path.exists():
        warnings.append(f"Gold file for correct reference count plot does not exist: {gold_path}")
        return set()

    if not output_path.exists():
        warnings.append(f"Output file for correct reference count plot does not exist: {output_path}")
        return set()

    try:
        output_payload = read_json_file(output_path)
    except Exception as exc:
        warnings.append(f"Could not read output file for correct reference count plot {output_path}: {exc}")
        return set()

    try:
        gold_payload = read_json_file(gold_path)
    except Exception as exc:
        warnings.append(f"Could not read gold file for correct reference count plot {gold_path}: {exc}")
        return set()

    if not isinstance(output_payload, dict):
        warnings.append(f"Output file for correct reference count plot is not a JSON object: {output_path}")
        return set()
    if not isinstance(gold_payload, dict):
        warnings.append(f"Gold file for correct reference count plot is not a JSON object: {gold_path}")
        return set()

    paper = str(output_payload.get("paper_id") or fallback_paper_id or output_path.stem)
    predicted = normalize_pred_record(output_payload)
    gold = normalize_gold_record(gold_payload)
    matches = match_references(
        predicted,
        gold,
        title_threshold=title_threshold,
        raw_threshold=raw_threshold,
    )
    return {f"{paper}::gold:{match.gold_index}" for match in matches}


def collect_correct_reference_sets(
    run_dir: Path,
    scores: list[dict[str, Any]],
    groups: list[GroupKey],
    title_threshold: float,
    raw_threshold: float,
    warnings: list[str],
) -> dict[GroupKey, set[str]]:
    reference_sets: dict[GroupKey, set[str]] = {group: set() for group in groups}
    seen_paths_by_group: dict[GroupKey, set[Path]] = defaultdict(set)

    for row in scores:
        key = group_key(row)
        if key not in reference_sets:
            continue
        output_path = resolve_run_path(run_dir, row.get("output_path"))
        gold_path = resolve_run_path(run_dir, row.get("gold_path"))
        if output_path is None:
            continue
        if gold_path is None:
            warnings.append(f"No gold path found for correct reference count plot group {key[0]}/{key[1]}.")
            continue
        if output_path in seen_paths_by_group[key]:
            continue
        seen_paths_by_group[key].add(output_path)
        reference_sets[key].update(
            load_correct_reference_set(
                output_path,
                gold_path,
                paper_id(row),
                title_threshold,
                raw_threshold,
                warnings,
            )
        )

    for key in groups:
        if not seen_paths_by_group.get(key):
            warnings.append(f"No scored output files found for correct reference count plot group {key[0]}/{key[1]}.")

    return reference_sets


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "item"


def annotate_bars(ax: plt.Axes, bars: Any, values: list[float | None], fmt: str) -> None:
    for bar, value in zip(bars, values, strict=False):
        if value is None:
            bar.set_alpha(0.25)
            bar.set_hatch("//")
            continue
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, image_format: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.{image_format}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def group_display_label(key: GroupKey, model_labels: dict[str, str]) -> str:
    model = model_labels.get(key[1], key[1])
    model = model.removeprefix("gemini-")
    model = model.replace("-preview", " prev")
    model = model.replace("-", " ")
    pipeline = PIPELINE_SHORT_LABELS.get(key[0], pipeline_label(key[0]))
    return f"{pipeline}\n{model}"


def paper_metric_matrix(
    papers: list[str],
    groups: list[GroupKey],
    score_by_key: dict[tuple[str, str, str], dict[str, Any]],
    metric_by_key: dict[tuple[str, str, str], dict[str, Any]],
    page_counts: dict[str, float],
    metric_name: str,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for paper in papers:
        row_values: list[float] = []
        for pipeline, model_id in groups:
            score = score_by_key.get((pipeline, model_id, paper))
            metric = metric_by_key.get((pipeline, model_id, paper))
            value: float | None = None

            if metric_name in {"reference_f1", "reference_precision", "reference_recall", "reference_count_error"}:
                value = numeric(score.get(metric_name)) if score else None
            elif metric_name == "wall_clock_seconds_per_page":
                seconds = numeric(metric.get("wall_clock_seconds")) if metric else None
                pages = page_counts.get(paper)
                value = seconds / pages if seconds is not None and pages else None
            elif metric_name == "total_tokens_per_page":
                input_tokens = numeric(metric.get("input_tokens")) if metric else None
                output_tokens = numeric(metric.get("output_tokens")) if metric else None
                pages = page_counts.get(paper)
                if input_tokens is not None and output_tokens is not None and pages:
                    value = (input_tokens + output_tokens) / pages
            elif metric_name == "estimated_cost_usd_per_page":
                cost = numeric(metric.get("estimated_cost_usd")) if metric else None
                pages = page_counts.get(paper)
                value = cost / pages if cost is not None and pages else None
            elif metric_name == "error":
                value = 1.0 if metric and metric.get("error") else 0.0 if metric else None

            row_values.append(value if value is not None else float("nan"))
        matrix.append(row_values)
    return matrix


def finite_values(matrix: list[list[float]]) -> list[float]:
    values: list[float] = []
    for row in matrix:
        for value in row:
            if not math.isnan(value):
                values.append(value)
    return values


def format_heatmap_value(value: float, fmt: str) -> str:
    if math.isnan(value):
        return ""
    if fmt == "currency":
        return f"{value:.4f}"
    if fmt == "signed_int":
        return f"{value:+.0f}"
    if fmt == "int":
        return f"{value:.0f}"
    if fmt == "one_decimal":
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    matrix: list[list[float]],
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    cmap_name: str,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#E6E6E6")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(length=0)

    values = finite_values(matrix)
    annotate = len(row_labels) * max(len(column_labels), 1) <= 36
    if annotate:
        midpoint = (min(values) + max(values)) / 2 if values else 0
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if math.isnan(value):
                    continue
                text_color = "white" if value > midpoint else "black"
                ax.text(
                    column_index,
                    row_index,
                    format_heatmap_value(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def model_color(model_id: str, models: list[str]) -> str:
    return MODEL_COLORS[models.index(model_id) % len(MODEL_COLORS)]


def nested_metric(values: dict[GroupKey, Any], pipelines: list[str], models: list[str]) -> dict[str, dict[str, Any]]:
    return {
        pipeline: {
            model_id: values.get((pipeline, model_id))
            for model_id in models
            if (pipeline, model_id) in values
        }
        for pipeline in pipelines
    }


def plot_grouped_metric(
    output_dir: Path,
    image_format: str,
    dpi: int,
    pipelines: list[str],
    models: list[str],
    model_labels: dict[str, str],
    values_by_group: dict[GroupKey, float | None],
    stem: str,
    title: str,
    ylabel: str,
    value_format: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    width = min(0.76 / max(len(models), 1), 0.24)
    x_positions = list(range(len(pipelines)))
    fig, ax = plt.subplots(figsize=(max(9, 2.1 * len(pipelines) + 2.2), 5.2))

    for model_index, model_id in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        values = [values_by_group.get((pipeline, model_id)) for pipeline in pipelines]
        bars = ax.bar(
            [position + offset for position in x_positions],
            [value or 0.0 for value in values],
            width,
            color=model_color(model_id, models),
            label=model_labels.get(model_id, model_id),
        )
        annotate_bars(ax, bars, values, value_format)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([pipeline_label(pipeline) for pipeline in pipelines], rotation=18, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem, image_format, dpi)


def plot_tokens_per_page(
    output_dir: Path,
    image_format: str,
    dpi: int,
    pipelines: list[str],
    models: list[str],
    model_labels: dict[str, str],
    values_by_group: dict[GroupKey, dict[str, float | None]],
) -> Path:
    width = min(0.76 / max(len(models), 1), 0.24)
    x_positions = list(range(len(pipelines)))
    fig, ax = plt.subplots(figsize=(max(9.5, 2.1 * len(pipelines) + 2.2), 5.2))

    for model_index, model_id in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        input_values = [
            values_by_group.get((pipeline, model_id), {}).get("input_tokens_per_page")
            for pipeline in pipelines
        ]
        output_values = [
            values_by_group.get((pipeline, model_id), {}).get("output_tokens_per_page")
            for pipeline in pipelines
        ]
        input_numeric = [value or 0.0 for value in input_values]
        output_numeric = [value or 0.0 for value in output_values]
        positions = [position + offset for position in x_positions]
        ax.bar(
            positions,
            input_numeric,
            width,
            color=model_color(model_id, models),
            label=model_labels.get(model_id, model_id),
        )
        ax.bar(
            positions,
            output_numeric,
            width,
            bottom=input_numeric,
            color=model_color(model_id, models),
            alpha=0.45,
            hatch="//",
        )
        totals = [
            None if input_value is None or output_value is None else input_value + output_value
            for input_value, output_value in zip(input_values, output_values, strict=False)
        ]
        for position, total in zip(positions, totals, strict=False):
            if total is None:
                continue
            ax.annotate(
                f"{total:.0f}",
                xy=(position, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Average Input/Output Tokens per Page")
    ax.set_ylabel("Known tokens per page")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([pipeline_label(pipeline) for pipeline in pipelines], rotation=18, ha="right")
    model_legend = ax.legend(loc="upper left", title="Model")
    ax.add_artist(model_legend)
    ax.legend(
        handles=[
            Patch(facecolor="#777777", label="Input"),
            Patch(facecolor="#777777", alpha=0.45, hatch="//", label="Output"),
        ],
        loc="upper right",
        title="Token type",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "average_input_output_tokens_per_page", image_format, dpi)


def plot_paper_heatmaps(
    output_dir: Path,
    image_format: str,
    dpi: int,
    papers: list[str],
    groups: list[GroupKey],
    model_labels: dict[str, str],
    score_by_key: dict[tuple[str, str, str], dict[str, Any]],
    metric_by_key: dict[tuple[str, str, str], dict[str, Any]],
    page_counts: dict[str, float],
) -> Path:
    column_labels = [group_display_label(group, model_labels) for group in groups]
    heatmaps = [
        ("reference_f1", "Reference F1", "viridis", "default", 0.0, 1.0),
        ("reference_precision", "Reference Precision", "viridis", "default", 0.0, 1.0),
        ("reference_recall", "Reference Recall", "viridis", "default", 0.0, 1.0),
        ("reference_count_error", "Reference Count Error", "coolwarm", "signed_int", None, None),
        ("wall_clock_seconds_per_page", "Seconds/Page", "magma", "one_decimal", None, None),
        ("total_tokens_per_page", "Tokens/Page", "plasma", "int", None, None),
        ("estimated_cost_usd_per_page", "Cost/Page USD", "cividis", "currency", None, None),
        ("error", "Run Error", "Reds", "int", 0.0, 1.0),
    ]
    fig_width = max(18.0, 1.1 * len(groups) + 7.0)
    fig_height = max(13.5, 0.42 * len(papers) + 11.0)
    fig, axes = plt.subplots(4, 2, figsize=(fig_width, fig_height), squeeze=False)

    for index, (metric_name, title, cmap_name, fmt, vmin, vmax) in enumerate(heatmaps):
        row_index = index // 2
        column_index = index % 2
        matrix = paper_metric_matrix(
            papers,
            groups,
            score_by_key,
            metric_by_key,
            page_counts,
            metric_name,
        )
        draw_heatmap(
            axes[row_index][column_index],
            fig,
            matrix,
            papers,
            column_labels,
            title,
            cmap_name,
            fmt,
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle("Per-Paper Benchmark Metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir, "paper_heatmaps", image_format, dpi)


def plot_single_paper_dashboard(
    output_dir: Path,
    image_format: str,
    dpi: int,
    paper: str,
    groups: list[GroupKey],
    model_labels: dict[str, str],
    score_by_key: dict[tuple[str, str, str], dict[str, Any]],
    metric_by_key: dict[tuple[str, str, str], dict[str, Any]],
    page_counts: dict[str, float],
) -> Path:
    labels = [group_display_label(group, model_labels) for group in groups]
    metrics = [
        ("reference_f1", "Reference F1", "F1", "{:.2f}", (0, 1.05)),
        ("reference_precision", "Reference Precision", "Precision", "{:.2f}", (0, 1.05)),
        ("reference_recall", "Reference Recall", "Recall", "{:.2f}", (0, 1.05)),
        ("wall_clock_seconds_per_page", "Seconds/Page", "Seconds/page", "{:.1f}", None),
        ("total_tokens_per_page", "Tokens/Page", "Tokens/page", "{:.0f}", None),
        ("estimated_cost_usd_per_page", "Cost/Page", "USD/page", "${:.4f}", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), squeeze=False)

    for index, (metric_name, title, ylabel, value_format, ylim) in enumerate(metrics):
        ax = axes[index // 3][index % 3]
        matrix = paper_metric_matrix(
            [paper],
            groups,
            score_by_key,
            metric_by_key,
            page_counts,
            metric_name,
        )
        values = matrix[0] if matrix else []
        numeric_values = [0.0 if math.isnan(value) else value for value in values]
        bars = ax.bar(range(len(groups)), numeric_values, color="#4C78A8")
        for bar, value in zip(bars, values, strict=False):
            if math.isnan(value):
                bar.set_alpha(0.25)
                bar.set_hatch("//")
                continue
            ax.annotate(
                value_format.format(value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Per-Paper Dashboard: {paper}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paper_dir = output_dir / "papers"
    return save_figure(fig, paper_dir, slug(paper), image_format, dpi)


def plot_paper_dashboards(
    output_dir: Path,
    image_format: str,
    dpi: int,
    papers: list[str],
    groups: list[GroupKey],
    model_labels: dict[str, str],
    score_by_key: dict[tuple[str, str, str], dict[str, Any]],
    metric_by_key: dict[tuple[str, str, str], dict[str, Any]],
    page_counts: dict[str, float],
) -> list[Path]:
    return [
        plot_single_paper_dashboard(
            output_dir,
            image_format,
            dpi,
            paper,
            groups,
            model_labels,
            score_by_key,
            metric_by_key,
            page_counts,
        )
        for paper in papers
    ]


def reference_count_breakdown(sets_by_name: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    breakdown: dict[str, dict[str, int]] = {}
    for name, values in sets_by_name.items():
        other_values: set[str] = set()
        for other_name, other_set in sets_by_name.items():
            if other_name != name:
                other_values.update(other_set)
        unique_count = len(values - other_values)
        breakdown[name] = {
            "total_references": len(values),
            "unique_references": unique_count,
            "shared_references": len(values) - unique_count,
        }
    return breakdown


def draw_reference_count_breakdown(
    ax: plt.Axes,
    sets_by_name: dict[str, set[str]],
    labels: dict[str, str],
    title: str,
    show_legend: bool,
) -> dict[str, dict[str, int]]:
    names = list(sets_by_name)
    breakdown = reference_count_breakdown(sets_by_name)
    unique_values = [breakdown[name]["unique_references"] for name in names]
    shared_values = [breakdown[name]["shared_references"] for name in names]
    total_values = [breakdown[name]["total_references"] for name in names]

    x_positions = list(range(len(names)))
    width = 0.62

    ax.bar(
        x_positions,
        unique_values,
        width,
        color="#F58518",
        label="Unique to this set",
    )
    ax.bar(
        x_positions,
        shared_values,
        width,
        bottom=unique_values,
        color="#4C78A8",
        label="Also found elsewhere",
    )

    max_total = max(total_values or [0])
    small_segment_threshold = max(max_total * 0.06, 8)
    for position, total, unique in zip(x_positions, total_values, unique_values, strict=False):
        ax.annotate(
            str(total),
            xy=(position, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
        if unique:
            y = unique / 2
            va = "center"
            if unique < small_segment_threshold:
                y = unique + max(max_total * 0.025, 2)
                va = "bottom"
            ax.annotate(
                str(unique),
                xy=(position, y),
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
                color="#000000",
            )

    ax.set_title(title)
    ax.set_ylabel("Correct references")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([labels.get(name, name) for name in names], rotation=18, ha="right")
    ax.set_ylim(0, max_total * 1.15 if max_total else 1)
    if show_legend:
        ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    return breakdown


def plot_reference_count_breakdown(
    output_dir: Path,
    image_format: str,
    dpi: int,
    sets_by_name: dict[str, set[str]],
    labels: dict[str, str],
    stem: str,
    title: str,
) -> tuple[Path, dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(max(7.2, 1.4 * len(sets_by_name) + 2.6), 5.4))
    breakdown = draw_reference_count_breakdown(ax, sets_by_name, labels, title, show_legend=True)
    fig.tight_layout()
    path = save_figure(fig, output_dir, stem, image_format, dpi)

    return path, breakdown


def plot_reference_count_grid(
    output_dir: Path,
    image_format: str,
    dpi: int,
    charts: list[tuple[dict[str, set[str]], dict[str, str], str]],
    stem: str,
    title: str,
    columns: int,
) -> Path:
    columns = max(1, min(columns, len(charts)))
    rows = math.ceil(len(charts) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.3 * columns, 4.9 * rows), squeeze=False)

    for index, (sets_by_name, labels, chart_title) in enumerate(charts):
        row_index = index // columns
        column_index = index % columns
        draw_reference_count_breakdown(
            axes[row_index][column_index],
            sets_by_name,
            labels,
            chart_title,
            show_legend=False,
        )

    for index in range(len(charts), rows * columns):
        row_index = index // columns
        column_index = index % columns
        axes[row_index][column_index].axis("off")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.legend(
        handles=[
            Patch(facecolor="#F58518", label="Unique to this set"),
            Patch(facecolor="#4C78A8", label="Also found elsewhere"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, output_dir, stem, image_format, dpi)


def plot_reference_count_breakdowns(
    output_dir: Path,
    image_format: str,
    dpi: int,
    pipelines: list[str],
    models: list[str],
    model_labels: dict[str, str],
    reference_sets: dict[GroupKey, set[str]],
) -> tuple[list[Path], dict[str, Any]]:
    plot_paths: list[Path] = []
    summaries: dict[str, Any] = {
        "by_pipeline": {},
        "by_model": {},
        "identity": "paper_id plus matched gold reference index",
        "counts": "total matched/correct references, with unique_references found only by that set within the chart",
    }
    by_pipeline_charts: list[tuple[dict[str, set[str]], dict[str, str], str]] = []
    by_model_charts: list[tuple[dict[str, set[str]], dict[str, str], str]] = []

    for pipeline in pipelines:
        sets_by_name = {
            model_id: reference_sets.get((pipeline, model_id), set())
            for model_id in models
        }
        title = f"By Model: {pipeline_label(pipeline)}"
        by_pipeline_charts.append((sets_by_name, model_labels, title))
        path, summary = plot_reference_count_breakdown(
            output_dir,
            image_format,
            dpi,
            sets_by_name,
            model_labels,
            f"reference_counts_by_pipeline_{slug(pipeline)}",
            f"Correct Reference Counts {title}",
        )
        plot_paths.append(path)
        summaries["by_pipeline"][pipeline] = summary

    for model_id in models:
        sets_by_name = {
            pipeline: reference_sets.get((pipeline, model_id), set())
            for pipeline in pipelines
        }
        labels = {pipeline: pipeline_label(pipeline) for pipeline in pipelines}
        title = f"By Pipeline: {model_labels.get(model_id, model_id)}"
        by_model_charts.append((sets_by_name, labels, title))
        path, summary = plot_reference_count_breakdown(
            output_dir,
            image_format,
            dpi,
            sets_by_name,
            labels,
            f"reference_counts_by_model_{slug(model_id)}",
            f"Correct Reference Counts {title}",
        )
        plot_paths.append(path)
        summaries["by_model"][model_id] = summary

    if by_model_charts:
        plot_paths.append(
            plot_reference_count_grid(
                output_dir,
                image_format,
                dpi,
                by_model_charts,
                "reference_counts_by_model",
                "Correct Reference Counts by Model",
                columns=2 if len(by_model_charts) > 3 else len(by_model_charts),
            )
        )
    if by_pipeline_charts:
        plot_paths.append(
            plot_reference_count_grid(
                output_dir,
                image_format,
                dpi,
                by_pipeline_charts,
                "reference_counts_by_pipeline",
                "Correct Reference Counts by Pipeline",
                columns=min(3, len(by_pipeline_charts)),
            )
        )

    return plot_paths, summaries


def cleanup_legacy_plots(output_dir: Path, image_format: str) -> list[str]:
    removed: list[str] = []
    for stem in LEGACY_PLOT_STEMS:
        path = output_dir / f"{stem}.{image_format}"
        if path.exists():
            path.unlink()
            removed.append(str(path))
    for pattern in (
        f"reference_overlap_by_pipeline_*.{image_format}",
        f"reference_overlap_by_model_*.{image_format}",
        f"reference_counts_by_pipeline_*.{image_format}",
        f"reference_counts_by_model_*.{image_format}",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()
            removed.append(str(path))
    paper_dir = output_dir / "papers"
    if paper_dir.exists():
        for path in paper_dir.glob(f"*.{image_format}"):
            path.unlink()
            removed.append(str(path))
    return removed


def create_manifest(
    run_dir: Path,
    output_dir: Path,
    image_format: str,
    pipelines: list[str],
    models: list[str],
    papers: list[str],
    model_labels: dict[str, str],
    title_threshold: float,
    raw_threshold: float,
    plot_paths: list[Path],
    warnings: list[str],
    removed_legacy_plots: list[str],
    computed_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "format": image_format,
        "pipelines": pipelines,
        "models": models,
        "papers": papers,
        "model_labels": model_labels,
        "matching_thresholds": {
            "title_threshold": title_threshold,
            "raw_threshold": raw_threshold,
        },
        "averaging": {
            "accuracy": "paper-weighted mean over attempted papers per pipeline/model",
            "unscored_attempts": "counted as 0 for precision, recall, and F1",
            "time": "paper-weighted mean of wall_clock_seconds / page_count",
            "tokens": "paper-weighted mean over rows with known input and output token usage",
            "cost": "paper-weighted mean over rows with known estimated_cost_usd",
            "error_rate": "errored metric rows / attempted metric rows",
            "reference_counts": "matched/correct references keyed within each source paper by matched gold reference index",
        },
        "source_files": {
            "metrics": str(run_dir / "metrics.jsonl"),
            "scores": str(run_dir / "scores.jsonl"),
        },
        "plots": [str(path) for path in plot_paths],
        "removed_legacy_plots": removed_legacy_plots,
        "computed_metrics": computed_metrics,
        "warnings": warnings,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-weighted static plots for a completed benchmark run."
    )
    parser.add_argument("--run-dir", type=Path, help="Benchmark run directory. Defaults to latest.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing benchmark runs when --run-dir is omitted.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for generated plots.")
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Image format for generated plots.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="DPI for raster plot output.")
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        help="Pipeline to plot. Repeatable or comma-separated. Defaults to all pipelines.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model/model_id to plot. Repeatable or comma-separated. Defaults to all models.",
    )
    parser.add_argument(
        "--title-threshold",
        type=float,
        default=DEFAULT_TITLE_THRESHOLD,
        help="Title similarity threshold for recomputing correct-reference count plots.",
    )
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=DEFAULT_RAW_THRESHOLD,
        help="Raw-reference similarity threshold for recomputing correct-reference count plots.",
    )
    parser.add_argument(
        "--paper-plots",
        action="store_true",
        help="Also generate one per-paper dashboard PNG under plots/papers/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        run_dir = args.run_dir or find_latest_run(args.output_root)
        output_dir = args.output_dir or (run_dir / "plots")
        warnings: list[str] = []

        metrics, scores = load_inputs(run_dir)
        scores = enrich_score_models(scores, metrics)
        requested_pipelines = parse_list(args.pipeline)
        requested_models = parse_list(args.model)

        all_rows = metrics + scores
        pipelines = requested_pipelines or ordered_pipelines(all_rows)
        metrics = filter_rows(metrics, pipelines, requested_models)
        scores = filter_rows(scores, pipelines, requested_models)
        all_rows = metrics + scores
        models = ordered_models(all_rows)
        if not pipelines:
            raise PlottingError("No pipeline data found to plot.")
        if not models:
            raise PlottingError("No model data found to plot.")

        model_labels = {model_id: model_id for model_id in models}
        for row in all_rows:
            model_id = row_model_id(row)
            if model_id in model_labels and model_labels[model_id] == model_id:
                model_labels[model_id] = row_model_label(row)

        page_counts = collect_page_counts(metrics, warnings)
        metrics_by_group = rows_by_group(metrics)
        scores_by_group = rows_by_group(scores)
        groups = [(pipeline, model_id) for pipeline in pipelines for model_id in models]
        papers = ordered_papers_for_plot(metrics, scores)
        score_by_key = score_lookup(scores)
        metric_by_key = metric_lookup(metrics)
        reference_sets = collect_correct_reference_sets(
            run_dir,
            scores,
            groups,
            args.title_threshold,
            args.raw_threshold,
            warnings,
        )
        removed_legacy_plots = cleanup_legacy_plots(output_dir, args.format)

        average_f1 = paper_weighted_accuracy(
            groups, metrics_by_group, scores_by_group, scores, "reference_f1", warnings
        )
        average_precision = paper_weighted_accuracy(
            groups, metrics_by_group, scores_by_group, scores, "reference_precision", warnings
        )
        average_recall = paper_weighted_accuracy(
            groups, metrics_by_group, scores_by_group, scores, "reference_recall", warnings
        )
        average_time_per_page = average_wall_clock_seconds_per_page(
            groups, metrics_by_group, page_counts, warnings
        )
        average_error_rate = error_rates(groups, metrics_by_group)
        average_tokens = average_tokens_per_page(groups, metrics_by_group, page_counts, warnings)
        average_cost = average_cost_per_page(groups, metrics_by_group, page_counts, warnings)

        plot_paths = [
            plot_grouped_metric(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_f1,
                "average_f1_per_paper",
                "Average F1 per Paper",
                "Paper-weighted F1",
                "{:.3f}",
                ylim=(0, 1.08),
            ),
            plot_grouped_metric(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_precision,
                "average_precision_per_paper",
                "Average Precision per Paper",
                "Paper-weighted precision",
                "{:.3f}",
                ylim=(0, 1.08),
            ),
            plot_grouped_metric(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_recall,
                "average_recall_per_paper",
                "Average Recall per Paper",
                "Paper-weighted recall",
                "{:.3f}",
                ylim=(0, 1.08),
            ),
            plot_grouped_metric(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_time_per_page,
                "average_wall_clock_time_per_page",
                "Average Wall-Clock Time per Page",
                "Seconds per page",
                "{:.1f}",
            ),
            plot_grouped_metric(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_error_rate,
                "average_error_rate",
                "Average Error Rate",
                "Errored papers (%)",
                "{:.1f}%",
                ylim=(0, 100),
            ),
            plot_tokens_per_page(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_tokens,
            ),
            plot_grouped_metric(
                output_dir,
                args.format,
                args.dpi,
                pipelines,
                models,
                model_labels,
                average_cost,
                "average_cost_per_page",
                "Average Cost per Page",
                "Estimated USD per page",
                "${:.4f}",
            ),
        ]
        plot_paths.append(
            plot_paper_heatmaps(
                output_dir,
                args.format,
                args.dpi,
                papers,
                groups,
                model_labels,
                score_by_key,
                metric_by_key,
                page_counts,
            )
        )
        if args.paper_plots:
            plot_paths.extend(
                plot_paper_dashboards(
                    output_dir,
                    args.format,
                    args.dpi,
                    papers,
                    groups,
                    model_labels,
                    score_by_key,
                    metric_by_key,
                    page_counts,
                )
            )
        reference_count_plot_paths, reference_count_summary = plot_reference_count_breakdowns(
            output_dir,
            args.format,
            args.dpi,
            pipelines,
            models,
            model_labels,
            reference_sets,
        )
        plot_paths.extend(reference_count_plot_paths)

        computed_metrics = {
            "average_f1_per_paper": nested_metric(average_f1, pipelines, models),
            "average_precision_per_paper": nested_metric(average_precision, pipelines, models),
            "average_recall_per_paper": nested_metric(average_recall, pipelines, models),
            "average_wall_clock_seconds_per_page": nested_metric(average_time_per_page, pipelines, models),
            "average_error_rate_percent": nested_metric(average_error_rate, pipelines, models),
            "average_input_output_tokens_per_page": nested_metric(average_tokens, pipelines, models),
            "average_cost_usd_per_page": nested_metric(average_cost, pipelines, models),
            "reference_counts": reference_count_summary,
        }
        manifest = create_manifest(
            run_dir,
            output_dir,
            args.format,
            pipelines,
            models,
            papers,
            model_labels,
            args.title_threshold,
            args.raw_threshold,
            plot_paths,
            warnings,
            removed_legacy_plots,
            computed_metrics,
        )
        manifest_path = output_dir / "manifest.json"
        write_json(manifest_path, manifest)

        print(f"Generated {len(plot_paths)} plot files in {output_dir}")
        print(f"Wrote manifest to {manifest_path}")
        if warnings:
            print(f"{len(warnings)} warning(s); see manifest.json for details.")
        return 0
    except PlottingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
