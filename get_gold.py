#!/usr/bin/env python3
"""Fetch ground-truth reference lists for local PDFs.

The script reads a manifest that maps each PDF to a Semantic Scholar paper
identifier, calls the Semantic Scholar Academic Graph API, and writes one JSON
object per source paper to JSONL.

Example manifest:

    pdf,doi,paper_id
    pdfs/1706.03762v7.pdf,10.48550/arXiv.1706.03762,
    pdfs/2020.emnlp-main.550.pdf,,DOI:10.18653/v1/2020.emnlp-main.550

You can use a raw DOI, a DOI: prefixed ID, an ARXIV: ID, an ACL: ID, a
CorpusId: ID, or a Semantic Scholar paperId hash.
"""

from __future__ import annotations

import argparse
import csv
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


BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_MANIFEST = Path("pdfs/paper_ids.csv")
DEFAULT_OUTPUT_DIR = Path("pdfs/gold_set")
DEFAULT_PAGE_SIZE = 100

SOURCE_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "title",
        "authors",
        "year",
        "venue",
        "publicationDate",
        "url",
        "citationCount",
        "referenceCount",
    ]
)

REFERENCE_FIELDS = [
    "isInfluential",
    "intents",
    "citedPaper.paperId",
    "citedPaper.externalIds",
    "citedPaper.title",
    "citedPaper.authors",
    "citedPaper.year",
    "citedPaper.venue",
    "citedPaper.publicationDate",
    "citedPaper.url",
    "citedPaper.citationCount",
]


class GoldBuildError(RuntimeError):
    """Raised when the gold-set build cannot continue safely."""


@dataclass(frozen=True)
class ManifestEntry:
    pdf: Path
    paper_id: str
    row_number: int


def normalize_doi(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    return f"DOI:{value}"


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


def normalize_arxiv(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^arxiv:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"v\d+$", "", value, flags=re.IGNORECASE)
    return f"ARXIV:{value}"


def normalize_acl(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^acl:\s*", "", value, flags=re.IGNORECASE)
    return f"ACL:{value}"


def normalize_paper_id(value: str) -> str:
    value = value.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered.startswith(("doi:", "arxiv:", "acl:", "corpusid:", "mag:")):
        if lowered.startswith("doi:"):
            return normalize_doi(value)
        if lowered.startswith("arxiv:"):
            return normalize_arxiv(value)
        if lowered.startswith("acl:"):
            return normalize_acl(value)
        if lowered.startswith("corpusid:"):
            return f"CorpusId:{value.split(':', 1)[1].strip()}"
        if lowered.startswith("mag:"):
            return f"MAG:{value.split(':', 1)[1].strip()}"
        return value

    if value.startswith("10."):
        return normalize_doi(value)
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", value, flags=re.IGNORECASE):
        return normalize_arxiv(value)
    if re.fullmatch(r"\d{4}\.[a-z]+-[a-z0-9-]+\.\d+", value, flags=re.IGNORECASE):
        return normalize_acl(value)
    return value


def infer_paper_id(pdf: Path) -> str:
    stem = pdf.stem

    arxiv_match = re.fullmatch(r"(\d{4}\.\d{4,5})(?:v\d+)?", stem, flags=re.IGNORECASE)
    if arxiv_match:
        return f"ARXIV:{arxiv_match.group(1)}"

    if re.fullmatch(r"\d{4}\.[a-z]+-[a-z0-9-]+\.\d+", stem, flags=re.IGNORECASE):
        return f"DOI:10.18653/v1/{stem}"

    return ""


def identifier_from_row(row: dict[str, str], pdf: Path, infer_ids: bool) -> str:
    for key in ("paper_id", "semantic_scholar_id", "s2_id"):
        if row.get(key, "").strip():
            return normalize_paper_id(row[key])

    if row.get("doi", "").strip():
        return normalize_doi(row["doi"])
    if row.get("arxiv", "").strip():
        return normalize_arxiv(row["arxiv"])
    if row.get("acl", "").strip():
        return normalize_acl(row["acl"])
    if row.get("corpus_id", "").strip():
        return f"CorpusId:{row['corpus_id'].strip()}"

    if infer_ids:
        return infer_paper_id(pdf)
    return ""


def load_manifest(path: Path, infer_ids: bool) -> list[ManifestEntry]:
    if not path.exists():
        raise GoldBuildError(
            f"Manifest not found: {path}. Run with --init-manifest first, or pass --manifest."
        )

    entries: list[ManifestEntry] = []
    missing: list[str] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise GoldBuildError(f"Manifest is empty: {path}")
        if "pdf" not in reader.fieldnames:
            raise GoldBuildError("Manifest must include a 'pdf' column.")

        for row_number, row in enumerate(reader, start=2):
            pdf_value = row.get("pdf", "").strip()
            if not pdf_value:
                missing.append(f"row {row_number}: missing pdf")
                continue

            pdf = Path(pdf_value)
            paper_id = identifier_from_row(row, pdf, infer_ids)
            if not paper_id:
                missing.append(f"row {row_number}: {pdf_value}")
                continue

            entries.append(ManifestEntry(pdf=pdf, paper_id=paper_id, row_number=row_number))

    if missing:
        formatted = "\n  - ".join(missing)
        raise GoldBuildError(
            "Every manifest row needs a DOI or Semantic Scholar-compatible ID.\n"
            f"Missing identifiers:\n  - {formatted}"
        )

    if not entries:
        raise GoldBuildError(f"No runnable entries found in {path}.")

    return entries


def create_manifest(path: Path, pdf_dir: Path, force: bool) -> None:
    if path.exists() and not force:
        raise GoldBuildError(f"Refusing to overwrite existing manifest: {path}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise GoldBuildError(f"No PDF files found in {pdf_dir}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdf", "doi", "paper_id"])
        writer.writeheader()
        for pdf in pdfs:
            writer.writerow(
                {
                    "pdf": pdf.as_posix(),
                    "doi": "",
                    "paper_id": infer_paper_id(pdf),
                }
            )


def retry_after_seconds(exc: HTTPError, fallback: float) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), fallback)
        except ValueError:
            return fallback
    return fallback


def api_get(
    path: str,
    params: dict[str, Any],
    api_key: str | None,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    query = urlencode(params)
    url = f"{BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"

    headers = {"User-Agent": "pdf-benchmark-gold-builder/1.0"}
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(retries + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {429, 500, 502, 503, 504}
            if retryable and attempt < retries:
                wait = retry_after_seconds(exc, sleep_seconds * (2**attempt))
                time.sleep(wait)
                continue
            raise GoldBuildError(f"Semantic Scholar request failed ({exc.code}): {body}") from exc
        except URLError as exc:
            if attempt < retries:
                time.sleep(sleep_seconds * (2**attempt))
                continue
            raise GoldBuildError(f"Semantic Scholar request failed: {exc}") from exc

    raise GoldBuildError("Semantic Scholar request failed after retries.")


def fetch_paper(
    paper_id: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    encoded_id = quote(paper_id, safe="")
    return api_get(
        f"/paper/{encoded_id}",
        {"fields": SOURCE_FIELDS},
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        sleep_seconds=sleep_seconds,
    )


def fetch_references(
    paper_id: str,
    fields: str,
    page_size: int,
    api_key: str | None,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    encoded_id = quote(paper_id, safe="")
    offset = 0
    references: list[dict[str, Any]] = []

    while True:
        payload = api_get(
            f"/paper/{encoded_id}/references",
            {"fields": fields, "limit": page_size, "offset": offset},
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
        page_data = payload.get("data") or []
        if not isinstance(page_data, list):
            raise GoldBuildError(
                "Semantic Scholar returned an unexpected references payload for "
                f"{paper_id}: expected a list or null for 'data'."
            )
        references.extend(page_data)

        next_offset = payload.get("next")
        if next_offset is None:
            break
        offset = int(next_offset)
        time.sleep(sleep_seconds)

    return references


def compact_authors(authors: Any) -> list[dict[str, str | None]]:
    if not isinstance(authors, list):
        return []
    return [
        {"authorId": author.get("authorId"), "name": author.get("name")}
        for author in authors
        if isinstance(author, dict)
    ]


def compact_paper(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": paper.get("paperId"),
        "externalIds": paper.get("externalIds") or {},
        "title": paper.get("title"),
        "authors": compact_authors(paper.get("authors")),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "publicationDate": paper.get("publicationDate"),
        "url": paper.get("url"),
        "citationCount": paper.get("citationCount"),
        "referenceCount": paper.get("referenceCount"),
    }


def compact_reference(item: dict[str, Any], index: int, include_contexts: bool) -> dict[str, Any]:
    cited_paper = compact_paper(item.get("citedPaper") or {})
    reference = {
        "index": index,
        "isInfluential": item.get("isInfluential"),
        "intents": item.get("intents") or [],
        "citedPaper": cited_paper,
    }
    if include_contexts:
        reference["contexts"] = item.get("contexts") or []
    return reference


def build_gold_record(
    entry: ManifestEntry,
    args: argparse.Namespace,
    fields: str,
) -> dict[str, Any]:
    print(f"Fetching {entry.pdf} ({entry.paper_id})", file=sys.stderr)
    source = fetch_paper(
        entry.paper_id,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep,
    )
    references = fetch_references(
        entry.paper_id,
        fields=fields,
        page_size=args.page_size,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep,
    )

    return {
        "pdf": entry.pdf.as_posix(),
        "queryId": entry.paper_id,
        "source": compact_paper(source),
        "referenceCount": len(references),
        "references": [
            compact_reference(item, index, include_contexts=args.include_contexts)
            for index, item in enumerate(references)
        ],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(path)


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def safe_output_stem(record: dict[str, Any], index: int) -> str:
    pdf = str(record.get("pdf") or "")
    query_id = str(record.get("queryId") or "")
    raw_stem = Path(pdf).stem if pdf else query_id
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_stem).strip("._-")
    return stem or f"paper-{index:02d}"


def write_per_paper_json(output_dir: Path, records: list[dict[str, Any]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    written: list[Path] = []

    for index, record in enumerate(records):
        stem = safe_output_stem(record, index)
        seen[stem] = seen.get(stem, 0) + 1
        if seen[stem] > 1:
            stem = f"{stem}-{seen[stem]}"

        output_path = output_dir / f"{stem}.json"
        write_json(output_path, record)
        written.append(output_path)

    return written


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldBuildError(
                    f"Could not parse {path}:{line_number} as JSONL: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise GoldBuildError(f"Expected object records in {path}:{line_number}.")
            records.append(record)
    return records


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL gold set of each PDF's reference list via Semantic Scholar."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"CSV with pdf plus doi/paper_id columns (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for pretty per-paper JSON files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--combined-output",
        "--output",
        dest="combined_output",
        type=Path,
        default=None,
        help="Optional combined JSONL output path. --output is kept as an alias.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        help="Semantic Scholar API key. Defaults to S2_API_KEY or SEMANTIC_SCHOLAR_API_KEY.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"References per API page (default: {DEFAULT_PAGE_SIZE}).",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for 429/5xx/network errors.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between reference pages and retry backoff base.",
    )
    parser.add_argument(
        "--include-contexts",
        action="store_true",
        help="Include Semantic Scholar citation context snippets for each reference.",
    )
    parser.add_argument(
        "--no-infer-ids",
        action="store_true",
        help="Do not infer ARXIV/ACL IDs from PDF filenames when manifest cells are blank.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print manifest entries without calling the API.",
    )
    parser.add_argument(
        "--init-manifest",
        action="store_true",
        help="Create a CSV manifest template from PDFs in --pdf-dir, then exit.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("pdfs"),
        help="PDF directory used with --init-manifest (default: pdfs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --init-manifest to overwrite an existing manifest.",
    )
    parser.add_argument(
        "--split-combined",
        type=Path,
        help="Split an existing combined JSONL file into pretty per-paper JSON files, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env_file(DEFAULT_ENV_FILE)
    args = parse_args(argv or sys.argv[1:])

    try:
        if args.init_manifest:
            create_manifest(args.manifest, args.pdf_dir, force=args.force)
            print(f"Wrote manifest template: {args.manifest}")
            return 0

        if args.split_combined:
            records = read_jsonl(args.split_combined)
            written = write_per_paper_json(args.output_dir, records)
            print(f"Wrote {len(written)} per-paper JSON files to {args.output_dir}")
            return 0

        if not (1 <= args.page_size <= 1000):
            raise GoldBuildError("--page-size must be between 1 and 1000.")

        entries = load_manifest(args.manifest, infer_ids=not args.no_infer_ids)
        if args.dry_run:
            for entry in entries:
                print(f"{entry.pdf}\t{entry.paper_id}")
            return 0

        fields = list(REFERENCE_FIELDS)
        if args.include_contexts:
            fields.append("contexts")

        records = [build_gold_record(entry, args, fields=",".join(fields)) for entry in entries]
        written = write_per_paper_json(args.output_dir, records)
        print(f"Wrote {len(written)} per-paper JSON files to {args.output_dir}")
        if args.combined_output:
            write_jsonl(args.combined_output, records)
            print(f"Wrote combined JSONL to {args.combined_output}")
        return 0
    except GoldBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
