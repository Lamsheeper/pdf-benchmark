#!/usr/bin/env python3
"""Run bibliography extraction benchmark pipelines.

The three supported pipelines match ``pipeline_instructions.md``:

* direct_pdf: send the PDF itself to the LLM.
* pymupdf_markdown: extract page text with PyMuPDF, then send text to the LLM.
* pdf_images: render pages with PyMuPDF, then send page images to the LLM.

The extraction step intentionally does not use Semantic Scholar, Crossref,
OpenAlex, DBLP, or any other citation database.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import getpass
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_PDF_DIR = Path("pdfs")
DEFAULT_OUTPUT_ROOT = Path("benchmark_runs")
DEFAULT_PRICE_FILE = Path("price.jsonl")
DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_DPI = 200
DEFAULT_IMAGE_BATCH_PAGES = 5
PIPELINES = ("direct_pdf", "pymupdf_markdown", "pdf_images")
PROVIDERS = ("gemini", "openai")
GEMINI_MODEL_CHOICES = [
    ("gemini-2.5-flash", "Fast default for broad benchmark sweeps"),
    ("gemini-2.5-pro", "Higher-capability option for harder PDFs"),
    ("gemini-2.5-flash-lite", "Lowest-cost Gemini 2.5 option"),
    ("gemini-3.1-flash-lite", "Gemini 3 cost-efficient high-volume option"),
    ("gemini-3.1-pro-preview", "Gemini 3 highest-capability preview option"),
]
OPENAI_MODEL_CHOICES = [
    ("gpt-4o-mini", "Fast default fallback"),
    ("gpt-4o", "Stronger multimodal fallback"),
]

REFERENCE_KEYS = [
    "ref_id",
    "raw_reference",
    "authors",
    "title",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
]


class PipelineError(RuntimeError):
    """Raised when a benchmark pipeline cannot complete."""


@dataclass(frozen=True)
class PriceRecord:
    provider: str
    model: str
    input_price_per_million: float
    output_price_per_million: float
    currency: str = "USD"
    source: str | None = None
    notes: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMCallResult:
    parsed: dict[str, Any]
    response: dict[str, Any]
    raw_text: str
    input_tokens: int | None
    output_tokens: int | None
    schema_valid: bool
    validation_errors: list[str]


@dataclass(frozen=True)
class PipelineRunResult:
    output: dict[str, Any]
    metric: dict[str, Any]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            key, separator, value = stripped.partition("=")
            if not separator:
                continue
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, value)


def normalize_provider(value: str) -> str:
    return value.strip().lower()


def normalize_model(provider: str, value: str) -> str:
    model = value.strip()
    if normalize_provider(provider) == "gemini":
        model = model.removeprefix("models/")
    return model.lower()


def model_id_for(provider: str, model: str) -> str:
    normalized = normalize_model(provider, model)
    model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    return model_id or "model"


def parse_price_record(data: dict[str, Any], path: Path, line_number: int) -> PriceRecord:
    provider = data.get("provider")
    model = data.get("model")
    input_price = data.get("input_price_per_million")
    output_price = data.get("output_price_per_million")
    currency = data.get("currency", "USD")
    source = data.get("source")
    notes = data.get("notes")

    if not isinstance(provider, str) or not provider.strip():
        raise PipelineError(f"{path}:{line_number} price record must include provider.")
    if not isinstance(model, str) or not model.strip():
        raise PipelineError(f"{path}:{line_number} price record must include model.")
    if not isinstance(input_price, int | float) or isinstance(input_price, bool):
        raise PipelineError(f"{path}:{line_number} input_price_per_million must be a number.")
    if not isinstance(output_price, int | float) or isinstance(output_price, bool):
        raise PipelineError(f"{path}:{line_number} output_price_per_million must be a number.")
    if input_price < 0:
        raise PipelineError(f"{path}:{line_number} input_price_per_million must be non-negative.")
    if output_price < 0:
        raise PipelineError(f"{path}:{line_number} output_price_per_million must be non-negative.")
    if not isinstance(currency, str) or not currency.strip():
        raise PipelineError(f"{path}:{line_number} currency must be a string.")
    if source is not None and not isinstance(source, str):
        raise PipelineError(f"{path}:{line_number} source must be a string when present.")
    if notes is not None and not isinstance(notes, str):
        raise PipelineError(f"{path}:{line_number} notes must be a string when present.")

    return PriceRecord(
        provider=normalize_provider(provider),
        model=normalize_model(provider, model),
        input_price_per_million=float(input_price),
        output_price_per_million=float(output_price),
        currency=currency.strip().upper(),
        source=source,
        notes=notes,
        raw=data,
    )


def load_price_records(path: Path) -> list[PriceRecord]:
    if not path.exists():
        return []

    records: list[PriceRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise PipelineError(f"{path}:{line_number} price record must be a JSON object.")
            records.append(parse_price_record(data, path, line_number))
    return records


def find_price_record(
    records: list[PriceRecord],
    provider: str,
    model: str,
) -> PriceRecord | None:
    normalized_provider = normalize_provider(provider)
    normalized_model = normalize_model(provider, model)
    for record in records:
        if record.provider == normalized_provider and record.model == normalized_model:
            return record
    return None


def apply_price_defaults(args: argparse.Namespace) -> None:
    args.price_source = None
    args.price_currency = None
    args.price_record = None

    cli_input_price = args.input_price_per_million is not None
    cli_output_price = args.output_price_per_million is not None
    if cli_input_price and cli_output_price:
        args.price_source = "cli"
        args.price_currency = "USD"
        return

    record = find_price_record(load_price_records(args.price_file), args.provider, args.model)

    if record:
        args.price_record = record.raw
        args.price_currency = record.currency
        if args.input_price_per_million is None:
            args.input_price_per_million = record.input_price_per_million
        if args.output_price_per_million is None:
            args.output_price_per_million = record.output_price_per_million

    if cli_input_price or cli_output_price:
        if record:
            args.price_source = f"cli+{args.price_file.as_posix()}"
        else:
            args.price_source = "cli"
    elif record:
        args.price_source = args.price_file.as_posix()
    if (
        args.price_currency is None
        and args.input_price_per_million is not None
        and args.output_price_per_million is not None
    ):
        args.price_currency = "USD"


def nullable_json_type(type_name: str) -> dict[str, Any]:
    return {"anyOf": [{"type": type_name}, {"type": "null"}]}


BIBLIOGRAPHY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "string"},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ref_id": {"type": "string"},
                    "raw_reference": nullable_json_type("string"),
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title": nullable_json_type("string"),
                    "year": nullable_json_type("integer"),
                    "venue": nullable_json_type("string"),
                    "doi": nullable_json_type("string"),
                    "arxiv_id": nullable_json_type("string"),
                    "url": nullable_json_type("string"),
                },
                "required": REFERENCE_KEYS,
            },
        },
    },
    "required": ["paper_id", "references"],
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "bibliography_extraction",
    "strict": True,
    "schema": BIBLIOGRAPHY_SCHEMA,
}

GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "propertyOrdering": ["paper_id", "references"],
    "properties": {
        "paper_id": {"type": "string"},
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "propertyOrdering": REFERENCE_KEYS,
                "properties": {
                    "ref_id": {"type": "string"},
                    "raw_reference": {"type": ["string", "null"]},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "venue": {"type": ["string", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                },
                "required": REFERENCE_KEYS,
            },
        },
    },
    "required": ["paper_id", "references"],
}

BIBLIOGRAPHY_PROMPT = """\
You extract bibliography/reference-list entries from academic papers.

Rules:
- Return only JSON matching the requested schema.
- Extract entries from the bibliography, references, or works-cited section.
- Do not use Semantic Scholar, Crossref, OpenAlex, DBLP, search engines, or any external citation database.
- Use only the supplied PDF, extracted text, or page images.
- Use null for unknown scalar fields and [] for unknown authors.
- Preserve the raw reference text as faithfully as possible in raw_reference.
- Use stable reference IDs R001, R002, R003, ... in document order.
- Do not include in-text citations unless they are part of the reference list.
"""


def make_user_prompt(paper_id: str, source_description: str) -> str:
    return (
        f"paper_id: {paper_id}\n"
        f"source: {source_description}\n\n"
        "Extract the bibliography/reference list for this paper into the JSON schema."
    )


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "paper"


def now_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def retry_after_seconds(exc: HTTPError, fallback: float) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), fallback)
        except ValueError:
            return fallback
    return fallback


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": "pdf-benchmark-pipelines/1.0",
    }
    request_headers.update(headers)

    for attempt in range(retries + 1):
        request = Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {429, 500, 502, 503, 504}
            if retryable and attempt < retries:
                wait = retry_after_seconds(exc, sleep_seconds * (2**attempt))
                time.sleep(wait)
                continue
            raise PipelineError(f"LLM request failed ({exc.code}): {error_body}") from exc
        except URLError as exc:
            if attempt < retries:
                time.sleep(sleep_seconds * (2**attempt))
                continue
            raise PipelineError(f"LLM request failed: {exc}") from exc

    raise PipelineError("LLM request failed after retries.")


def extract_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    refusals: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif content_type == "refusal" and isinstance(content.get("refusal"), str):
                refusals.append(content["refusal"])

    if refusals:
        raise PipelineError(f"Model refusal: {' '.join(refusals)}")
    if not parts:
        raise PipelineError("OpenAI response did not contain output text.")
    return "\n".join(parts)


def usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    return input_tokens, output_tokens


def gemini_usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = response.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount")
    output_tokens = usage.get("candidatesTokenCount")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    return input_tokens, output_tokens


def validate_bibliography(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top-level output must be an object"]

    extra_top_level = sorted(set(data) - {"paper_id", "references"})
    if extra_top_level:
        errors.append(f"top-level output has extra keys: {', '.join(extra_top_level)}")

    if not isinstance(data.get("paper_id"), str):
        errors.append("paper_id must be a string")

    references = data.get("references")
    if not isinstance(references, list):
        errors.append("references must be a list")
        return errors

    nullable_strings = {"raw_reference", "title", "venue", "doi", "arxiv_id", "url"}
    for index, reference in enumerate(references):
        prefix = f"references[{index}]"
        if not isinstance(reference, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra_keys = sorted(set(reference) - set(REFERENCE_KEYS))
        if extra_keys:
            errors.append(f"{prefix} has extra keys: {', '.join(extra_keys)}")
        missing = [key for key in REFERENCE_KEYS if key not in reference]
        if missing:
            errors.append(f"{prefix} missing keys: {', '.join(missing)}")

        if not isinstance(reference.get("ref_id"), str):
            errors.append(f"{prefix}.ref_id must be a string")

        authors = reference.get("authors")
        if not isinstance(authors, list) or not all(isinstance(author, str) for author in authors):
            errors.append(f"{prefix}.authors must be a list of strings")

        year = reference.get("year")
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            errors.append(f"{prefix}.year must be an integer or null")

        for key in nullable_strings:
            value = reference.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{prefix}.{key} must be a string or null")

    return errors


def empty_bibliography(paper_id: str) -> dict[str, Any]:
    return {"paper_id": paper_id, "references": []}


def parse_llm_json(raw_text: str, paper_id: str) -> tuple[dict[str, Any], list[str]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return empty_bibliography(paper_id), [f"model output was not valid JSON: {exc}"]

    errors = validate_bibliography(parsed)
    if errors:
        return parsed if isinstance(parsed, dict) else empty_bibliography(paper_id), errors
    return parsed, []


def call_openai_bibliography(
    content: list[dict[str, Any]],
    args: argparse.Namespace,
    paper_id: str,
) -> LLMCallResult:
    payload: dict[str, Any] = {
        "model": args.model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": BIBLIOGRAPHY_PROMPT}],
            },
            {"role": "user", "content": content},
        ],
        "text": {"format": RESPONSE_FORMAT},
    }
    if args.max_output_tokens:
        payload["max_output_tokens"] = args.max_output_tokens

    response = post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep,
    )
    raw_text = extract_output_text(response)
    parsed, validation_errors = parse_llm_json(raw_text, paper_id)
    input_tokens, output_tokens = usage_tokens(response)
    return LLMCallResult(
        parsed=parsed,
        response=response,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=not validation_errors,
        validation_errors=validation_errors,
    )


def gemini_part(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "input_text":
        return {"text": item.get("text") or ""}
    if item_type == "input_file":
        return {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": item.get("file_data") or "",
            }
        }
    if item_type == "input_image":
        image_url = item.get("image_url") or ""
        prefix = "data:image/png;base64,"
        if not isinstance(image_url, str) or not image_url.startswith(prefix):
            raise PipelineError("Gemini image inputs must be PNG data URLs.")
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": image_url[len(prefix) :],
            }
        }
    raise PipelineError(f"Unsupported content item for Gemini: {item_type}")


def extract_gemini_output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    candidates = response.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])

    if texts:
        return "\n".join(texts)

    prompt_feedback = response.get("promptFeedback")
    if prompt_feedback:
        raise PipelineError(f"Gemini prompt feedback: {prompt_feedback}")
    raise PipelineError("Gemini response did not contain output text.")


def call_gemini_bibliography(
    content: list[dict[str, Any]],
    args: argparse.Namespace,
    paper_id: str,
) -> LLMCallResult:
    parts = [{"text": BIBLIOGRAPHY_PROMPT}]
    parts.extend(gemini_part(item) for item in content)
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": GEMINI_RESPONSE_SCHEMA,
    }
    if args.max_output_tokens:
        generation_config["maxOutputTokens"] = args.max_output_tokens

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    model_name = str(args.model).removeprefix("models/")
    url = f"{GEMINI_GENERATE_URL.format(model=quote(model_name, safe=''))}?{urlencode({'key': args.api_key})}"
    response = post_json(
        url,
        payload,
        headers={},
        timeout=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep,
    )
    raw_text = extract_gemini_output_text(response)
    parsed, validation_errors = parse_llm_json(raw_text, paper_id)
    input_tokens, output_tokens = gemini_usage_tokens(response)
    return LLMCallResult(
        parsed=parsed,
        response=response,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        schema_valid=not validation_errors,
        validation_errors=validation_errors,
    )


def call_llm_bibliography(
    content: list[dict[str, Any]],
    args: argparse.Namespace,
    paper_id: str,
) -> LLMCallResult:
    if args.provider == "gemini":
        return call_gemini_bibliography(content, args, paper_id)
    if args.provider == "openai":
        return call_openai_bibliography(content, args, paper_id)
    raise PipelineError(f"Unsupported provider: {args.provider}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> float | None:
    if (
        input_tokens is None
        or output_tokens is None
        or input_price_per_million is None
        or output_price_per_million is None
    ):
        return None
    return (
        (input_tokens / 1_000_000) * input_price_per_million
        + (output_tokens / 1_000_000) * output_price_per_million
    )


def sum_optional_int(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def base_metric(
    pdf_path: Path,
    pipeline_id: str,
    args: argparse.Namespace,
    wall_clock_seconds: float,
    input_tokens: int | None,
    output_tokens: int | None,
    num_llm_calls: int,
    schema_valid: bool,
    error: str | None,
) -> dict[str, Any]:
    return {
        "paper_id": safe_stem(pdf_path),
        "pdf": pdf_path.as_posix(),
        "pipeline_id": pipeline_id,
        "provider": args.provider,
        "model": args.model,
        "model_id": args.model_id,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimate_cost(
            input_tokens,
            output_tokens,
            args.input_price_per_million,
            args.output_price_per_million,
        ),
        "input_price_per_million": args.input_price_per_million,
        "output_price_per_million": args.output_price_per_million,
        "price_currency": args.price_currency,
        "price_source": args.price_source,
        "num_llm_calls": num_llm_calls,
        "schema_valid": schema_valid,
        "error": error,
    }


def save_llm_artifacts(
    run_dir: Path,
    pipeline_id: str,
    paper_id: str,
    args: argparse.Namespace,
    call: LLMCallResult,
    suffix: str = "",
) -> dict[str, str]:
    raw_dir = run_dir / "raw" / pipeline_id
    if args.multi_model:
        raw_dir = raw_dir / args.model_id
    raw_dir = raw_dir / paper_id
    name = "response" if not suffix else f"response-{suffix}"
    response_path = raw_dir / f"{name}.json"
    text_path = raw_dir / f"{name}.txt"
    write_json(response_path, call.response)
    text_path.write_text(call.raw_text, encoding="utf-8")
    return {"raw_response_path": response_path.as_posix(), "raw_text_path": text_path.as_posix()}


def save_pipeline_output(
    run_dir: Path,
    pipeline_id: str,
    paper_id: str,
    args: argparse.Namespace,
    output: dict[str, Any],
) -> Path:
    output_dir = run_dir / "outputs" / pipeline_id
    if args.multi_model:
        output_dir = output_dir / args.model_id
    output_path = output_dir / f"{paper_id}.json"
    write_json(output_path, output)
    return output_path


def run_direct_pdf(pdf_path: Path, run_dir: Path, args: argparse.Namespace) -> PipelineRunResult:
    pipeline_id = "direct_pdf"
    paper_id = safe_stem(pdf_path)
    start = time.perf_counter()

    pdf_data = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    content = [
        {"type": "input_text", "text": make_user_prompt(paper_id, "the attached PDF file")},
        {"type": "input_file", "filename": pdf_path.name, "file_data": pdf_data},
    ]
    call = call_llm_bibliography(content, args, paper_id)
    wall_clock_seconds = time.perf_counter() - start
    artifact_paths = save_llm_artifacts(run_dir, pipeline_id, paper_id, args, call)
    output_path = save_pipeline_output(run_dir, pipeline_id, paper_id, args, call.parsed)
    error = "; ".join(call.validation_errors) if call.validation_errors else None

    metric = base_metric(
        pdf_path,
        pipeline_id,
        args,
        wall_clock_seconds,
        call.input_tokens,
        call.output_tokens,
        1,
        call.schema_valid,
        error,
    )
    metric.update(artifact_paths)
    metric["output_path"] = output_path.as_posix()
    return PipelineRunResult(output=call.parsed, metric=metric)


def require_pymupdf():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise PipelineError(
            "PyMuPDF is required for this pipeline. Run `uv sync` to install project dependencies."
        ) from exc
    return fitz


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    fitz = require_pymupdf()
    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            pages.append(f"# Page {page_number}\n\n{text.strip()}\n")
    return "\n".join(pages), page_count


def maybe_truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def run_pymupdf_markdown(
    pdf_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> PipelineRunResult:
    pipeline_id = "pymupdf_markdown"
    paper_id = safe_stem(pdf_path)
    start = time.perf_counter()

    conversion_start = time.perf_counter()
    markdown, page_count = extract_pdf_text(pdf_path)
    pdf_to_markdown_seconds = time.perf_counter() - conversion_start

    markdown_path = run_dir / "intermediate" / pipeline_id / f"{paper_id}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    text_for_model, truncated = maybe_truncate_text(markdown, args.max_text_chars)
    prompt = make_user_prompt(paper_id, "PyMuPDF-extracted page text")
    if truncated:
        prompt += (
            f"\n\nNote: the extracted text was truncated to {args.max_text_chars} characters "
            "by the benchmark runner."
        )
    content = [{"type": "input_text", "text": f"{prompt}\n\n{text_for_model}"}]

    call = call_llm_bibliography(content, args, paper_id)
    wall_clock_seconds = time.perf_counter() - start
    artifact_paths = save_llm_artifacts(run_dir, pipeline_id, paper_id, args, call)
    output_path = save_pipeline_output(run_dir, pipeline_id, paper_id, args, call.parsed)
    error = "; ".join(call.validation_errors) if call.validation_errors else None

    metric = base_metric(
        pdf_path,
        pipeline_id,
        args,
        wall_clock_seconds,
        call.input_tokens,
        call.output_tokens,
        1,
        call.schema_valid,
        error,
    )
    metric.update(artifact_paths)
    metric.update(
        {
            "output_path": output_path.as_posix(),
            "markdown_path": markdown_path.as_posix(),
            "pdf_to_markdown_seconds": round(pdf_to_markdown_seconds, 6),
            "num_pages": page_count,
            "text_chars_total": len(markdown),
            "text_chars_sent": len(text_for_model),
            "text_truncated": truncated,
        }
    )
    return PipelineRunResult(output=call.parsed, metric=metric)


def render_pdf_images(pdf_path: Path, image_dir: Path, dpi: int) -> tuple[list[Path], float]:
    fitz = require_pymupdf()
    image_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    paths: list[Path] = []
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = image_dir / f"page-{page_index:03d}.png"
            pixmap.save(image_path)
            paths.append(image_path)
    return paths, time.perf_counter() - start


def image_content_item(image_path: Path, detail: str) -> dict[str, Any]:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{image_data}",
        "detail": detail,
    }


def chunks(items: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size <= 0:
        return [items]
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def reference_identity(reference: dict[str, Any]) -> str:
    for key in ("doi", "arxiv_id", "title", "raw_reference"):
        value = reference.get(key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value.strip().lower())
    return json.dumps(reference, sort_keys=True)


def merge_bibliographies(paper_id: str, bibliographies: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"paper_id": paper_id, "references": []}
    seen: set[str] = set()
    for bibliography in bibliographies:
        references = bibliography.get("references") if isinstance(bibliography, dict) else []
        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict):
                continue
            identity = reference_identity(reference)
            if identity in seen:
                continue
            seen.add(identity)
            item = copy.deepcopy(reference)
            item["ref_id"] = f"R{len(merged['references']) + 1:03d}"
            merged["references"].append(item)
    return merged


def run_pdf_images(pdf_path: Path, run_dir: Path, args: argparse.Namespace) -> PipelineRunResult:
    pipeline_id = "pdf_images"
    paper_id = safe_stem(pdf_path)
    start = time.perf_counter()

    image_dir = run_dir / "intermediate" / pipeline_id / paper_id
    image_paths, pdf_to_images_seconds = render_pdf_images(pdf_path, image_dir, args.image_dpi)
    if not image_paths:
        raise PipelineError(f"No pages were rendered from {pdf_path}.")
    image_batches = chunks(image_paths, args.image_max_pages_per_call)

    calls: list[LLMCallResult] = []
    parsed_batches: list[dict[str, Any]] = []
    validation_errors: list[str] = []

    for batch_index, batch in enumerate(image_batches, start=1):
        first_page = image_paths.index(batch[0]) + 1
        last_page = first_page + len(batch) - 1
        prompt = make_user_prompt(
            paper_id,
            f"page images {first_page}-{last_page} of {len(image_paths)}",
        )
        prompt += (
            "\n\nIf this page range does not contain bibliography/reference entries, "
            "return the schema with an empty references array."
        )
        content = [{"type": "input_text", "text": prompt}]
        content.extend(image_content_item(path, args.image_detail) for path in batch)

        call = call_llm_bibliography(content, args, paper_id)
        calls.append(call)
        parsed_batches.append(call.parsed)
        if call.validation_errors:
            validation_errors.extend(f"batch {batch_index}: {error}" for error in call.validation_errors)
        save_llm_artifacts(run_dir, pipeline_id, paper_id, args, call, suffix=f"batch-{batch_index:03d}")

    merged = merge_bibliographies(paper_id, parsed_batches)
    merged_errors = validate_bibliography(merged)
    validation_errors.extend(merged_errors)
    schema_valid = not validation_errors
    wall_clock_seconds = time.perf_counter() - start

    input_tokens = sum_optional_int([call.input_tokens for call in calls])
    output_tokens = sum_optional_int([call.output_tokens for call in calls])
    output_path = save_pipeline_output(run_dir, pipeline_id, paper_id, args, merged)

    metric = base_metric(
        pdf_path,
        pipeline_id,
        args,
        wall_clock_seconds,
        input_tokens,
        output_tokens,
        len(calls),
        schema_valid,
        "; ".join(validation_errors) if validation_errors else None,
    )
    metric.update(
        {
            "output_path": output_path.as_posix(),
            "image_dir": image_dir.as_posix(),
            "pdf_to_images_seconds": round(pdf_to_images_seconds, 6),
            "num_pages": len(image_paths),
            "image_dpi": args.image_dpi,
            "image_detail": args.image_detail,
            "image_max_pages_per_call": args.image_max_pages_per_call,
        }
    )
    return PipelineRunResult(output=merged, metric=metric)


def collect_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.pdf:
        pdfs = [Path(value) for value in args.pdf]
    else:
        pdfs = sorted(args.pdf_dir.glob("*.pdf"))

    missing = [path for path in pdfs if not path.exists()]
    if missing:
        formatted = ", ".join(path.as_posix() for path in missing)
        raise PipelineError(f"PDF not found: {formatted}")
    if not pdfs:
        raise PipelineError(f"No PDFs found in {args.pdf_dir}")
    if args.limit:
        pdfs = pdfs[: args.limit]
    return pdfs


def parse_pipeline_list(values: list[str]) -> list[str]:
    requested: list[str] = []
    for value in values:
        for part in value.split(","):
            pipeline = part.strip()
            if pipeline:
                requested.append(pipeline)
    unknown = [pipeline for pipeline in requested if pipeline not in PIPELINES]
    if unknown:
        raise PipelineError(
            f"Unknown pipeline(s): {', '.join(unknown)}. Choose from: {', '.join(PIPELINES)}"
        )
    return requested or list(PIPELINES)


def parse_model_list(values: list[str]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in value.split(","):
            model = part.strip()
            if not model:
                continue
            model_key = model.lower()
            if model_key in seen:
                continue
            models.append(model)
            seen.add(model_key)
    return models


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = args.output_root / (args.run_id or now_run_id())

    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise PipelineError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def pipeline_function(pipeline_id: str):
    return {
        "direct_pdf": run_direct_pdf,
        "pymupdf_markdown": run_pymupdf_markdown,
        "pdf_images": run_pdf_images,
    }[pipeline_id]


def error_metric(
    pdf_path: Path,
    pipeline_id: str,
    args: argparse.Namespace,
    start: float,
    error: str,
) -> dict[str, Any]:
    return base_metric(
        pdf_path,
        pipeline_id,
        args,
        time.perf_counter() - start,
        None,
        None,
        0,
        False,
        error,
    )


def write_run_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    pdfs: list[Path],
    pipelines: list[str],
    model_args: list[argparse.Namespace],
) -> None:
    pricing_by_model = {
        model_arg.model_id: {
            "model": model_arg.model,
            "price_source": model_arg.price_source,
            "price_currency": model_arg.price_currency,
            "input_price_per_million": model_arg.input_price_per_million,
            "output_price_per_million": model_arg.output_price_per_million,
            "price_record": model_arg.price_record,
        }
        for model_arg in model_args
    }
    metadata = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "provider": args.provider,
        "model": model_args[0].model if len(model_args) == 1 else None,
        "models": [model_arg.model for model_arg in model_args],
        "model_ids": [model_arg.model_id for model_arg in model_args],
        "pdfs": [path.as_posix() for path in pdfs],
        "pipelines": pipelines,
        "image_dpi": args.image_dpi,
        "image_detail": args.image_detail,
        "image_max_pages_per_call": args.image_max_pages_per_call,
        "max_text_chars": args.max_text_chars,
        "pricing": {
            "price_file": args.price_file.as_posix(),
            "models": pricing_by_model,
        },
    }
    write_json(run_dir / "run_metadata.json", metadata)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PDF bibliography extraction benchmark pipelines."
    )
    parser.add_argument(
        "--pdf",
        action="append",
        help="PDF path to benchmark. Repeatable. Defaults to all PDFs in --pdf-dir.",
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="PDF directory.")
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        help=f"Pipeline to run. Repeatable or comma-separated. Choices: {', '.join(PIPELINES)}.",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER),
        help=f"LLM provider (default: LLM_PROVIDER or {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model to use. Repeatable or comma-separated for same-provider multi-model runs. "
            "If omitted in a terminal, prompts for a selection. "
            "Otherwise defaults to LLM_MODEL, or provider default "
            f"({DEFAULT_GEMINI_MODEL} for Gemini, {DEFAULT_OPENAI_MODEL} for OpenAI)."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Provider API key. If missing in a terminal, prompts securely. "
            "Defaults to GEMINI_API_KEY/GOOGLE_API_KEY for Gemini or OPENAI_API_KEY for OpenAI."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for timestamped benchmark runs (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--run-id", help="Run ID directory name under --output-root.")
    parser.add_argument("--run-dir", type=Path, help="Exact output directory for this run.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty run dir.")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs, after sorting.")
    parser.add_argument("--timeout", type=float, default=300.0, help="LLM request timeout seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries for 429/5xx/network errors.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Retry backoff base seconds.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help="Optional provider max output token limit.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=0,
        help="Optional character cap for PyMuPDF text sent to the model. 0 means no cap.",
    )
    parser.add_argument(
        "--image-dpi",
        type=int,
        default=DEFAULT_IMAGE_DPI,
        help=f"PDF image render DPI (default: {DEFAULT_IMAGE_DPI}).",
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default="auto",
        help="OpenAI input_image detail level. Ignored by Gemini.",
    )
    parser.add_argument(
        "--image-max-pages-per-call",
        type=int,
        default=DEFAULT_IMAGE_BATCH_PAGES,
        help=(
            "Maximum rendered pages per image-pipeline LLM call. "
            "Use 0 to send all pages in one call."
        ),
    )
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        help=(
            "Override model input price per 1M tokens. "
            f"If omitted, defaults to --price-file ({DEFAULT_PRICE_FILE}) when available."
        ),
    )
    parser.add_argument(
        "--output-price-per-million",
        type=float,
        help=(
            "Override model output price per 1M tokens. "
            f"If omitted, defaults to --price-file ({DEFAULT_PRICE_FILE}) when available."
        ),
    )
    parser.add_argument(
        "--price-file",
        type=Path,
        default=DEFAULT_PRICE_FILE,
        help=(
            "JSONL catalog of model prices per 1M input/output tokens "
            f"(default: {DEFAULT_PRICE_FILE})."
        ),
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable terminal prompts for missing API keys and model selection.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip automatic evaluation after extraction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without API calls.")
    return parser.parse_args(argv)


def can_prompt(args: argparse.Namespace) -> bool:
    return (
        not args.no_interactive
        and not args.dry_run
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def provider_default_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    raise PipelineError(f"Unsupported provider: {provider}")


def provider_model_choices(provider: str) -> list[tuple[str, str]]:
    if provider == "gemini":
        return GEMINI_MODEL_CHOICES
    if provider == "openai":
        return OPENAI_MODEL_CHOICES
    raise PipelineError(f"Unsupported provider: {provider}")


def provider_env_model(provider: str) -> str | None:
    if provider == "gemini":
        return os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL")
    if provider == "openai":
        return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
    raise PipelineError(f"Unsupported provider: {provider}")


def provider_env_api_key(provider: str) -> str | None:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    raise PipelineError(f"Unsupported provider: {provider}")


def prompt_model(provider: str) -> str:
    choices = provider_model_choices(provider)
    default = provider_default_model(provider)
    print(f"\nSelect {provider} model:")
    for index, (model, description) in enumerate(choices, start=1):
        default_label = " [default]" if model == default else ""
        print(f"  {index}. {model}{default_label} - {description}")
    custom_index = len(choices) + 1
    print(f"  {custom_index}. Custom model name")

    while True:
        response = input(f"Model [{default}]: ").strip()
        if not response:
            return default
        if response.isdigit():
            selection = int(response)
            if 1 <= selection <= len(choices):
                return choices[selection - 1][0]
            if selection == custom_index:
                custom = input("Custom model: ").strip()
                if custom:
                    return custom
        else:
            return response
        print("Please enter a listed number or a model name.")


def prompt_api_key(provider: str) -> str:
    label = "Gemini" if provider == "gemini" else "OpenAI"
    value = getpass.getpass(f"{label} API key (input hidden, not saved): ").strip()
    return value


def apply_provider_defaults(args: argparse.Namespace) -> None:
    interactive = can_prompt(args)
    models = parse_model_list(args.model)
    if not models:
        env_model = provider_env_model(args.provider)
        models = parse_model_list([env_model]) if env_model else []
    if not models:
        models = [prompt_model(args.provider) if interactive else provider_default_model(args.provider)]

    unique_models: list[str] = []
    seen_model_ids: set[str] = set()
    for model in models:
        model_id = model_id_for(args.provider, model)
        if model_id in seen_model_ids:
            continue
        unique_models.append(model)
        seen_model_ids.add(model_id)
    args.models = unique_models
    args.model = unique_models[0]
    args.model_ids = [model_id_for(args.provider, model) for model in unique_models]

    args.api_key = args.api_key or provider_env_api_key(args.provider)
    if not args.api_key and interactive:
        args.api_key = prompt_api_key(args.provider)
        args.api_key = args.api_key or os.getenv("OPENAI_API_KEY")


def model_args_for_run(args: argparse.Namespace) -> list[argparse.Namespace]:
    multiple_models = len(args.models) > 1
    model_args: list[argparse.Namespace] = []
    for model in args.models:
        current_args = copy.copy(args)
        current_args.model = model
        current_args.model_id = model_id_for(args.provider, model)
        current_args.multi_model = multiple_models
        apply_price_defaults(current_args)
        model_args.append(current_args)
    return model_args


def run_auto_evaluation(run_dir: Path) -> None:
    try:
        from evaluate_outputs import main as evaluate_main
    except ImportError as exc:
        raise PipelineError(f"Automatic evaluation failed to import evaluator: {exc}") from exc

    exit_code = evaluate_main(["--run-dir", run_dir.as_posix(), "--no-match-details"])
    if exit_code:
        raise PipelineError(f"Automatic evaluation failed with exit code {exit_code}.")


def main(argv: list[str] | None = None) -> int:
    load_env_file(DEFAULT_ENV_FILE)
    args = parse_args(argv or sys.argv[1:])
    apply_provider_defaults(args)

    try:
        model_args = model_args_for_run(args)
        pipelines = parse_pipeline_list(args.pipeline) if args.pipeline else list(PIPELINES)
        pdfs = collect_pdfs(args)

        if args.dry_run:
            print(f"provider: {args.provider}")
            print(f"models: {', '.join(args.models)}")
            print(f"model_ids: {', '.join(args.model_ids)}")
            print(f"price_file: {args.price_file}")
            for model_arg in model_args:
                print(
                    "price: "
                    f"{model_arg.model_id} "
                    f"source={model_arg.price_source or 'none'} "
                    f"input={model_arg.input_price_per_million} "
                    f"output={model_arg.output_price_per_million}"
                )
            print(f"auto_evaluate: {not args.no_evaluate}")
            print(f"pipelines: {', '.join(pipelines)}")
            for pdf_path in pdfs:
                print(pdf_path.as_posix())
            return 0

        if not args.api_key:
            if args.provider == "gemini":
                raise PipelineError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY is required. Set it in .env or pass --api-key."
                )
            raise PipelineError("OPENAI_API_KEY is required. Set it in .env or pass --api-key.")
        if args.image_dpi <= 0:
            raise PipelineError("--image-dpi must be greater than 0.")
        if args.image_max_pages_per_call < 0:
            raise PipelineError("--image-max-pages-per-call must be 0 or greater.")

        run_dir = resolve_run_dir(args)
        write_run_metadata(run_dir, args, pdfs, pipelines, model_args)
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()

        for model_arg in model_args:
            for pdf_path in pdfs:
                for pipeline_id in pipelines:
                    start = time.perf_counter()
                    print(f"Running {model_arg.model_id}/{pipeline_id}: {pdf_path}", file=sys.stderr)
                    try:
                        result = pipeline_function(pipeline_id)(pdf_path, run_dir, model_arg)
                        append_jsonl(metrics_path, [result.metric])
                    except Exception as exc:
                        metric = error_metric(pdf_path, pipeline_id, model_arg, start, str(exc))
                        append_jsonl(metrics_path, [metric])
                        print(
                            f"error: {model_arg.model_id}/{pipeline_id} failed for {pdf_path}: {exc}",
                            file=sys.stderr,
                        )

        if not args.no_evaluate:
            run_auto_evaluation(run_dir)

        print(f"Wrote benchmark run to {run_dir}")
        print(f"Wrote metrics to {metrics_path}")
        return 0
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
