#!/usr/bin/env python3
"""Evaluate benchmark pipeline outputs against gold references.

This scorer compares local model-extracted bibliographies to the local gold set.
It does not call Semantic Scholar, Crossref, OpenAlex, DBLP, search engines, or
any other external citation database.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from run_pipelines import REFERENCE_KEYS, model_id_for, validate_bibliography


DEFAULT_GOLD_DIR = Path("pdfs/gold_set")
FIELD_NAMES = ["title", "authors", "year", "venue", "doi", "arxiv_id", "url"]
IDENTIFIER_METHODS = {"doi", "arxiv_id"}


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot continue."""


@dataclass(frozen=True)
class ReferenceMatch:
    pred_index: int
    gold_index: int
    method: str
    score: float


@dataclass(frozen=True)
class OutputFile:
    path: Path
    pipeline_id: str
    paper_id: str
    model_id: str | None = None


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def compact_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = compact_spaces(str(value))
    return text or None


def normalize_text(value: Any) -> str:
    text = clean_scalar(value)
    if not text:
        return ""
    text = normalize_unicode(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return compact_spaces(text)


def normalize_identifier(value: Any) -> str:
    text = clean_scalar(value)
    if not text:
        return ""
    text = normalize_unicode(text).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.strip(" .;,")
    return text


def normalize_arxiv(value: Any) -> str:
    text = clean_scalar(value)
    if not text:
        return ""
    text = normalize_unicode(text).lower()
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"^arxiv:\s*", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip(" .;,")


def normalize_url(value: Any) -> str:
    text = clean_scalar(value)
    if not text:
        return ""
    text = normalize_unicode(text).lower().strip(" .;,")
    text = re.sub(r"^https?://", "", text)
    text = text.rstrip("/")
    return text


def similarity(left: Any, right: Any) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return True


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_recall_f1(correct: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = correct / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = correct / gold if gold else (1.0 if predicted == 0 else 0.0)
    return {
        "correct": correct,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    records.append(data)
    return records


def row_model_id(row: dict[str, Any]) -> str:
    model_id = row.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    provider = row.get("provider")
    model = row.get("model")
    if isinstance(provider, str) and isinstance(model, str) and model.strip():
        return model_id_for(provider, model)
    if isinstance(model, str) and model.strip():
        return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip().lower()).strip("._-") or "model"
    return "default"


def row_model_label(row: dict[str, Any], model_id: str) -> str:
    model = row.get("model")
    return model if isinstance(model, str) and model.strip() else model_id


def parse_model_list(values: list[str]) -> list[str]:
    models: list[str] = []
    for value in values:
        models.extend(part.strip() for part in value.split(",") if part.strip())
    return models


def model_matches(requested: list[str], row: dict[str, Any] | None, model_id: str | None) -> bool:
    if not requested:
        return True
    model = row.get("model") if row else None
    provider = row.get("provider") if row else None
    for requested_model in requested:
        if requested_model == model_id or requested_model == model:
            return True
        if isinstance(provider, str) and model_id and model_id_for(provider, requested_model) == model_id:
            return True
    return False


def load_run_metrics(
    run_dir: Path,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    exact_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    legacy_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_jsonl(run_dir / "metrics.jsonl"):
        pipeline_id = record.get("pipeline_id")
        paper_id = record.get("paper_id")
        if isinstance(pipeline_id, str) and isinstance(paper_id, str):
            model_id = row_model_id(record)
            exact_metrics[(pipeline_id, model_id, paper_id)] = record
            legacy_metrics[(pipeline_id, paper_id)] = record
    return exact_metrics, legacy_metrics


def metric_for_output(
    output: OutputFile,
    exact_metrics: dict[tuple[str, str, str], dict[str, Any]],
    legacy_metrics: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    if output.model_id:
        metric = exact_metrics.get((output.pipeline_id, output.model_id, output.paper_id))
        if metric:
            return metric
    return legacy_metrics.get((output.pipeline_id, output.paper_id))


def external_id(external_ids: dict[str, Any], *names: str) -> str | None:
    lowered = {str(key).lower(): value for key, value in external_ids.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return str(value)
    return None


def author_names(authors: Any) -> list[str]:
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if isinstance(author, str) and author.strip():
            names.append(compact_spaces(author))
        elif isinstance(author, dict) and clean_scalar(author.get("name")):
            names.append(clean_scalar(author.get("name")) or "")
    return names


def reference_text(reference: dict[str, Any]) -> str:
    parts: list[str] = []
    authors = reference.get("authors")
    if isinstance(authors, list):
        parts.extend(str(author) for author in authors if str(author).strip())
    for key in ("title", "venue", "year", "doi", "arxiv_id", "url", "raw_reference"):
        value = reference.get(key)
        if nonempty(value):
            parts.append(str(value))
    return compact_spaces(" ".join(parts))


def normalize_gold_reference(item: dict[str, Any], index: int) -> dict[str, Any]:
    cited_paper = item.get("citedPaper") if isinstance(item.get("citedPaper"), dict) else {}
    external_ids = cited_paper.get("externalIds") if isinstance(cited_paper.get("externalIds"), dict) else {}
    doi = external_id(external_ids, "DOI")
    arxiv_id = external_id(external_ids, "ArXiv", "arxiv")
    reference = {
        "ref_id": f"R{index + 1:03d}",
        "raw_reference": None,
        "authors": author_names(cited_paper.get("authors")),
        "title": clean_scalar(cited_paper.get("title")),
        "year": cited_paper.get("year") if isinstance(cited_paper.get("year"), int) else None,
        "venue": clean_scalar(cited_paper.get("venue")),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": clean_scalar(cited_paper.get("url")),
    }
    reference["_gold_index"] = index
    reference["_reference_text"] = reference_text(reference)
    return reference


def normalize_pred_reference(item: dict[str, Any], index: int) -> dict[str, Any]:
    reference = {
        "ref_id": clean_scalar(item.get("ref_id")) or f"P{index + 1:03d}",
        "raw_reference": clean_scalar(item.get("raw_reference")),
        "authors": author_names(item.get("authors")),
        "title": clean_scalar(item.get("title")),
        "year": item.get("year") if isinstance(item.get("year"), int) and not isinstance(item.get("year"), bool) else None,
        "venue": clean_scalar(item.get("venue")),
        "doi": clean_scalar(item.get("doi")),
        "arxiv_id": clean_scalar(item.get("arxiv_id")),
        "url": clean_scalar(item.get("url")),
    }
    reference["_pred_index"] = index
    reference["_reference_text"] = reference_text(reference)
    return reference


def normalize_gold_record(data: dict[str, Any]) -> list[dict[str, Any]]:
    references = data.get("references") if isinstance(data, dict) else []
    if not isinstance(references, list):
        return []
    return [
        normalize_gold_reference(item, index)
        for index, item in enumerate(references)
        if isinstance(item, dict)
    ]


def normalize_pred_record(data: dict[str, Any]) -> list[dict[str, Any]]:
    references = data.get("references") if isinstance(data, dict) else []
    if not isinstance(references, list):
        return []
    return [
        normalize_pred_reference(item, index)
        for index, item in enumerate(references)
        if isinstance(item, dict)
    ]


def candidate_match(
    pred: dict[str, Any],
    gold: dict[str, Any],
    title_threshold: float,
    raw_threshold: float,
) -> tuple[str, float] | None:
    pred_doi = normalize_identifier(pred.get("doi"))
    gold_doi = normalize_identifier(gold.get("doi"))
    if pred_doi and gold_doi and pred_doi == gold_doi:
        return "doi", 1.0

    pred_arxiv = normalize_arxiv(pred.get("arxiv_id"))
    gold_arxiv = normalize_arxiv(gold.get("arxiv_id"))
    if pred_arxiv and gold_arxiv and pred_arxiv == gold_arxiv:
        return "arxiv_id", 1.0

    title_score = similarity(pred.get("title"), gold.get("title"))
    if title_score >= title_threshold:
        return "title", title_score

    raw_score = similarity(pred.get("raw_reference") or pred.get("_reference_text"), gold.get("_reference_text"))
    if raw_score >= raw_threshold:
        return "raw_reference", raw_score

    return None


def match_references(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    title_threshold: float,
    raw_threshold: float,
) -> list[ReferenceMatch]:
    method_priority = {"doi": 0, "arxiv_id": 1, "title": 2, "raw_reference": 3}
    candidates: list[tuple[int, float, int, int, str]] = []
    for pred_index, pred in enumerate(predicted):
        for gold_index, gold_ref in enumerate(gold):
            match = candidate_match(pred, gold_ref, title_threshold, raw_threshold)
            if not match:
                continue
            method, score = match
            candidates.append((method_priority[method], -score, pred_index, gold_index, method))

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[ReferenceMatch] = []
    for _, negative_score, pred_index, gold_index, method in sorted(candidates):
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matches.append(
            ReferenceMatch(
                pred_index=pred_index,
                gold_index=gold_index,
                method=method,
                score=-negative_score,
            )
        )
    return sorted(matches, key=lambda match: match.gold_index)


def first_initial(name: str) -> str:
    tokens = normalize_text(name).split()
    return tokens[0][0] if tokens and tokens[0] else ""


def last_name(name: str) -> str:
    tokens = normalize_text(name).split()
    return tokens[-1] if tokens else ""


def author_similarity(pred_author: str, gold_author: str) -> float:
    pred_norm = normalize_text(pred_author)
    gold_norm = normalize_text(gold_author)
    if not pred_norm or not gold_norm:
        return 0.0
    if pred_norm == gold_norm:
        return 1.0
    if last_name(pred_author) and last_name(pred_author) == last_name(gold_author):
        pred_initial = first_initial(pred_author)
        gold_initial = first_initial(gold_author)
        if not pred_initial or not gold_initial or pred_initial == gold_initial:
            return 0.9
    return SequenceMatcher(None, pred_norm, gold_norm).ratio()


def author_match_count(pred_authors: list[str], gold_authors: list[str], threshold: float) -> int:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_author in enumerate(pred_authors):
        for gold_index, gold_author in enumerate(gold_authors):
            score = author_similarity(pred_author, gold_author)
            if score >= threshold:
                candidates.append((-score, pred_index, gold_index))

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    count = 0
    for _, pred_index, gold_index in sorted(candidates):
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        count += 1
    return count


def scalar_field_correct(
    field: str,
    pred_value: Any,
    gold_value: Any,
    title_threshold: float,
    venue_threshold: float,
) -> bool:
    if not nonempty(pred_value) or not nonempty(gold_value):
        return False
    if field == "doi":
        return normalize_identifier(pred_value) == normalize_identifier(gold_value)
    if field == "arxiv_id":
        return normalize_arxiv(pred_value) == normalize_arxiv(gold_value)
    if field == "url":
        return normalize_url(pred_value) == normalize_url(gold_value)
    if field == "year":
        return pred_value == gold_value
    if field == "title":
        return similarity(pred_value, gold_value) >= title_threshold
    if field == "venue":
        return similarity(pred_value, gold_value) >= venue_threshold
    return normalize_text(pred_value) == normalize_text(gold_value)


def score_fields(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    matches: list[ReferenceMatch],
    title_threshold: float,
    venue_threshold: float,
    author_threshold: float,
) -> dict[str, dict[str, float | int]]:
    counts = {
        field: {"correct": 0, "predicted": 0, "gold": 0}
        for field in FIELD_NAMES
    }

    for match in matches:
        pred_ref = predicted[match.pred_index]
        gold_ref = gold[match.gold_index]
        for field in FIELD_NAMES:
            if field == "authors":
                pred_authors = pred_ref.get("authors") if isinstance(pred_ref.get("authors"), list) else []
                gold_authors = gold_ref.get("authors") if isinstance(gold_ref.get("authors"), list) else []
                counts[field]["predicted"] += len(pred_authors)
                counts[field]["gold"] += len(gold_authors)
                counts[field]["correct"] += author_match_count(pred_authors, gold_authors, author_threshold)
                continue

            pred_present = nonempty(pred_ref.get(field))
            gold_present = nonempty(gold_ref.get(field))
            counts[field]["predicted"] += int(pred_present)
            counts[field]["gold"] += int(gold_present)
            if scalar_field_correct(
                field,
                pred_ref.get(field),
                gold_ref.get(field),
                title_threshold,
                venue_threshold,
            ):
                counts[field]["correct"] += 1

    return {
        field: precision_recall_f1(
            int(values["correct"]),
            int(values["predicted"]),
            int(values["gold"]),
        )
        for field, values in counts.items()
    }


def exact_identifier_counts(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    matches: list[ReferenceMatch],
    field: str,
) -> dict[str, float | int | None]:
    correct = 0
    gold_present = 0
    pred_present = 0
    for match in matches:
        pred_value = predicted[match.pred_index].get(field)
        gold_value = gold[match.gold_index].get(field)
        pred_present += int(nonempty(pred_value))
        gold_present += int(nonempty(gold_value))
        if field == "doi":
            correct += int(
                bool(normalize_identifier(pred_value))
                and normalize_identifier(pred_value) == normalize_identifier(gold_value)
            )
        elif field == "arxiv_id":
            correct += int(
                bool(normalize_arxiv(pred_value))
                and normalize_arxiv(pred_value) == normalize_arxiv(gold_value)
            )

    accuracy = correct / gold_present if gold_present else None
    return {
        "correct": correct,
        "predicted_present": pred_present,
        "gold_present": gold_present,
        "accuracy": accuracy,
    }


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def match_details(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    matches: list[ReferenceMatch],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for match in matches:
        pred_ref = predicted[match.pred_index]
        gold_ref = gold[match.gold_index]
        details.append(
            {
                "pred_index": match.pred_index,
                "gold_index": match.gold_index,
                "method": match.method,
                "score": match.score,
                "pred_ref_id": pred_ref.get("ref_id"),
                "gold_ref_id": gold_ref.get("ref_id"),
                "pred_title": pred_ref.get("title"),
                "gold_title": gold_ref.get("title"),
                "pred_doi": pred_ref.get("doi"),
                "gold_doi": gold_ref.get("doi"),
                "pred_arxiv_id": pred_ref.get("arxiv_id"),
                "gold_arxiv_id": gold_ref.get("arxiv_id"),
            }
        )
    return details


def score_output(
    output: OutputFile,
    gold_path: Path,
    args: argparse.Namespace,
    run_metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = output.path
    paper_id = output.paper_id
    pipeline_id = output.pipeline_id
    model_id = output.model_id
    provider = None
    model = None
    if run_metric:
        provider = run_metric.get("provider") if isinstance(run_metric.get("provider"), str) else None
        model = run_metric.get("model") if isinstance(run_metric.get("model"), str) else None
        model_id = row_model_id(run_metric)
    if model_id is None:
        model_id = "default"
    if model is None:
        model = model_id
    schema_errors: list[str] = []
    try:
        pred_record = read_json(output_path)
    except json.JSONDecodeError as exc:
        pred_record = {"paper_id": paper_id, "references": []}
        schema_errors.append(f"prediction JSON parse error: {exc}")

    if not isinstance(pred_record, dict):
        schema_errors.append("prediction top-level JSON is not an object")
        pred_record = {"paper_id": paper_id, "references": []}
    else:
        schema_errors.extend(validate_bibliography(pred_record))

    if not gold_path.exists():
        gold_record: dict[str, Any] = {"references": []}
        schema_errors.append(f"gold file not found: {gold_path}")
    else:
        gold_record = read_json(gold_path)

    predicted = normalize_pred_record(pred_record)
    gold = normalize_gold_record(gold_record)
    matches = match_references(
        predicted,
        gold,
        title_threshold=args.title_threshold,
        raw_threshold=args.raw_threshold,
    )

    matched_count = len(matches)
    pred_count = len(predicted)
    gold_count = len(gold)
    reference_precision = matched_count / pred_count if pred_count else (1.0 if gold_count == 0 else 0.0)
    reference_recall = matched_count / gold_count if gold_count else (1.0 if pred_count == 0 else 0.0)
    reference_f1 = f1_score(reference_precision, reference_recall)

    title_scores = [
        similarity(predicted[match.pred_index].get("title"), gold[match.gold_index].get("title"))
        for match in matches
        if nonempty(gold[match.gold_index].get("title"))
    ]
    raw_scores = [
        similarity(
            predicted[match.pred_index].get("raw_reference")
            or predicted[match.pred_index].get("_reference_text"),
            gold[match.gold_index].get("_reference_text"),
        )
        for match in matches
        if nonempty(gold[match.gold_index].get("_reference_text"))
    ]

    field_metrics = score_fields(
        predicted,
        gold,
        matches,
        title_threshold=args.title_threshold,
        venue_threshold=args.venue_threshold,
        author_threshold=args.author_threshold,
    )
    doi_exact = exact_identifier_counts(predicted, gold, matches, "doi")
    arxiv_exact = exact_identifier_counts(predicted, gold, matches, "arxiv_id")
    method_counts = Counter(match.method for match in matches)

    output_schema_valid = not schema_errors
    run_schema_valid = None
    run_error = None
    if run_metric:
        run_schema_valid = run_metric.get("schema_valid")
        run_error = run_metric.get("error")

    score = {
        "paper_id": paper_id,
        "pipeline_id": pipeline_id,
        "provider": provider,
        "model": model,
        "model_id": model_id,
        "output_path": output_path.as_posix(),
        "gold_path": gold_path.as_posix(),
        "schema_valid": output_schema_valid and run_schema_valid is not False,
        "output_schema_valid": output_schema_valid,
        "run_schema_valid": run_schema_valid,
        "run_error": run_error,
        "schema_errors": schema_errors,
        "predicted_reference_count": pred_count,
        "gold_reference_count": gold_count,
        "reference_count_error": pred_count - gold_count,
        "absolute_reference_count_error": abs(pred_count - gold_count),
        "matched_reference_count": matched_count,
        "reference_precision": reference_precision,
        "reference_recall": reference_recall,
        "reference_f1": reference_f1,
        "match_method_counts": dict(method_counts),
        "average_match_score": mean_or_none([match.score for match in matches]),
        "average_title_similarity": mean_or_none(title_scores),
        "title_similarity_count": len(title_scores),
        "average_raw_reference_similarity": mean_or_none(raw_scores),
        "raw_reference_similarity_count": len(raw_scores),
        "doi_exact": doi_exact,
        "arxiv_exact": arxiv_exact,
        "field_metrics": field_metrics,
    }
    if not args.no_match_details:
        score["matches"] = match_details(predicted, gold, matches)
    return score


def find_latest_run(output_root: Path) -> Path:
    if not output_root.exists():
        raise EvaluationError(f"Output root not found: {output_root}")
    runs = [path for path in output_root.iterdir() if path.is_dir()]
    if not runs:
        raise EvaluationError(f"No benchmark runs found in {output_root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def output_files(
    run_dir: Path,
    pipelines: list[str],
    requested_models: list[str],
    exact_metrics: dict[tuple[str, str, str], dict[str, Any]],
    legacy_metrics: dict[tuple[str, str], dict[str, Any]],
) -> list[OutputFile]:
    outputs_dir = run_dir / "outputs"
    if not outputs_dir.exists():
        raise EvaluationError(f"Outputs directory not found: {outputs_dir}")
    files: list[OutputFile] = []
    for pipeline_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        if pipelines and pipeline_dir.name not in pipelines:
            continue
        for child in sorted(pipeline_dir.iterdir()):
            if child.is_file() and child.suffix == ".json":
                output = OutputFile(
                    path=child,
                    pipeline_id=pipeline_dir.name,
                    paper_id=child.stem,
                )
                metric = metric_for_output(output, exact_metrics, legacy_metrics)
                model_id = row_model_id(metric) if metric else None
                if model_matches(requested_models, metric, model_id):
                    files.append(output)
            elif child.is_dir():
                model_id = child.name
                for output_path in sorted(child.glob("*.json")):
                    output = OutputFile(
                        path=output_path,
                        pipeline_id=pipeline_dir.name,
                        paper_id=output_path.stem,
                        model_id=model_id,
                    )
                    metric = metric_for_output(output, exact_metrics, legacy_metrics)
                    if model_matches(requested_models, metric, model_id):
                        files.append(output)
    if not files:
        raise EvaluationError(f"No output JSON files found in {outputs_dir}")
    return files


def aggregate_field_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    totals = {
        field: {"correct": 0, "predicted": 0, "gold": 0}
        for field in FIELD_NAMES
    }
    for record in records:
        for field in FIELD_NAMES:
            metrics = record.get("field_metrics", {}).get(field, {})
            totals[field]["correct"] += int(metrics.get("correct") or 0)
            totals[field]["predicted"] += int(metrics.get("predicted") or 0)
            totals[field]["gold"] += int(metrics.get("gold") or 0)

    return {
        field: precision_recall_f1(
            int(values["correct"]),
            int(values["predicted"]),
            int(values["gold"]),
        )
        for field, values in totals.items()
    }


def aggregate_identifier(records: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    correct = sum(int(record.get(key, {}).get("correct") or 0) for record in records)
    predicted_present = sum(int(record.get(key, {}).get("predicted_present") or 0) for record in records)
    gold_present = sum(int(record.get(key, {}).get("gold_present") or 0) for record in records)
    return {
        "correct": correct,
        "predicted_present": predicted_present,
        "gold_present": gold_present,
        "accuracy": correct / gold_present if gold_present else None,
    }


def weighted_average(records: list[dict[str, Any]], value_key: str, count_key: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for record in records:
        value = record.get(value_key)
        count = record.get(count_key) or 0
        if value is None or not count:
            continue
        numerator += float(value) * int(count)
        denominator += int(count)
    if not denominator:
        return None
    return numerator / denominator


def aggregate_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    pred_count = sum(int(record["predicted_reference_count"]) for record in records)
    gold_count = sum(int(record["gold_reference_count"]) for record in records)
    matched_count = sum(int(record["matched_reference_count"]) for record in records)
    reference_precision = matched_count / pred_count if pred_count else (1.0 if gold_count == 0 else 0.0)
    reference_recall = matched_count / gold_count if gold_count else (1.0 if pred_count == 0 else 0.0)

    method_counts: Counter[str] = Counter()
    for record in records:
        method_counts.update(record.get("match_method_counts") or {})

    return {
        "num_records": len(records),
        "schema_valid_rate": sum(1 for record in records if record.get("schema_valid")) / len(records),
        "predicted_reference_count": pred_count,
        "gold_reference_count": gold_count,
        "matched_reference_count": matched_count,
        "reference_precision": reference_precision,
        "reference_recall": reference_recall,
        "reference_f1": f1_score(reference_precision, reference_recall),
        "mean_reference_count_error": statistics.fmean(
            float(record["reference_count_error"]) for record in records
        ),
        "mean_absolute_reference_count_error": statistics.fmean(
            float(record["absolute_reference_count_error"]) for record in records
        ),
        "match_method_counts": dict(method_counts),
        "average_title_similarity": weighted_average(
            records,
            "average_title_similarity",
            "title_similarity_count",
        ),
        "average_raw_reference_similarity": weighted_average(
            records,
            "average_raw_reference_similarity",
            "raw_reference_similarity_count",
        ),
        "doi_exact": aggregate_identifier(records, "doi_exact"),
        "arxiv_exact": aggregate_identifier(records, "arxiv_exact"),
        "field_metrics": aggregate_field_metrics(records),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_pipeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pipeline_model: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        pipeline_id = str(record["pipeline_id"])
        model_id = str(record.get("model_id") or "default")
        by_pipeline[pipeline_id].append(record)
        by_model[model_id].append(record)
        by_pipeline_model[pipeline_id][model_id].append(record)

    return {
        "overall": aggregate_scores(records),
        "by_pipeline": {
            pipeline_id: aggregate_scores(pipeline_records)
            for pipeline_id, pipeline_records in sorted(by_pipeline.items())
        },
        "by_model": {
            model_id: aggregate_scores(model_records)
            for model_id, model_records in sorted(by_model.items())
        },
        "by_pipeline_model": {
            pipeline_id: {
                model_id: aggregate_scores(model_records)
                for model_id, model_records in sorted(model_records_by_id.items())
            }
            for pipeline_id, model_records_by_id in sorted(by_pipeline_model.items())
        },
    }


def parse_pipeline_list(values: list[str]) -> list[str]:
    pipelines: list[str] = []
    for value in values:
        pipelines.extend(part.strip() for part in value.split(",") if part.strip())
    return pipelines


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark outputs against local gold reference files."
    )
    parser.add_argument("--run-dir", type=Path, help="Benchmark run directory. Defaults to latest.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_runs"),
        help="Root containing benchmark runs when --run-dir is omitted.",
    )
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR, help="Gold JSON directory.")
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        help="Pipeline to evaluate. Repeatable or comma-separated. Defaults to all output dirs.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model/model_id to evaluate. Repeatable or comma-separated. Defaults to all models.",
    )
    parser.add_argument("--scores-output", type=Path, help="Scores JSONL path.")
    parser.add_argument("--summary-output", type=Path, help="Summary JSON path.")
    parser.add_argument("--title-threshold", type=float, default=0.88)
    parser.add_argument("--raw-threshold", type=float, default=0.82)
    parser.add_argument("--venue-threshold", type=float, default=0.80)
    parser.add_argument("--author-threshold", type=float, default=0.80)
    parser.add_argument(
        "--no-match-details",
        action="store_true",
        help="Omit per-reference match details from scores.jsonl.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        run_dir = args.run_dir or find_latest_run(args.output_root)
        pipelines = parse_pipeline_list(args.pipeline)
        requested_models = parse_model_list(args.model)
        exact_metrics, legacy_metrics = load_run_metrics(run_dir)
        files = output_files(run_dir, pipelines, requested_models, exact_metrics, legacy_metrics)
        records: list[dict[str, Any]] = []
        for output in files:
            gold_path = args.gold_dir / f"{output.paper_id}.json"
            metric = metric_for_output(output, exact_metrics, legacy_metrics)
            records.append(score_output(output, gold_path, args, run_metric=metric))

        scores_output = args.scores_output or (run_dir / "scores.jsonl")
        summary_output = args.summary_output or (run_dir / "scores_summary.json")
        summary = build_summary(records)
        write_jsonl(scores_output, records)
        write_json(summary_output, summary)

        print(f"Evaluated {len(records)} output files from {run_dir}")
        print(f"Wrote scores to {scores_output}")
        print(f"Wrote summary to {summary_output}")
        return 0
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
