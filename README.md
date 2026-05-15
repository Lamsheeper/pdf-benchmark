# pdf-benchmark
[AFRL] Benchmarking different LLM pipelines for extracting information from PDFs

## Building the reference gold set

`get_gold.py` uses the Semantic Scholar Academic Graph API to fetch each PDF's
reference list and writes one pretty JSON file per source paper in
`pdfs/gold_set/`.

Set up the project with `uv`:

```sh
uv sync
```

If you already synced before this project had its script shim, refresh the
installed command:

```sh
uv sync --reinstall-package pdf-benchmark
```

1. Check or edit the PDF identifier manifest:

   ```sh
   uv run get-gold --dry-run
   ```

   The default manifest is `pdfs/paper_ids.csv`. Rows can use `doi`,
   `paper_id`, `arxiv`, `acl`, or `corpus_id` columns.

   For API keys, either pass `--api-key`, export `S2_API_KEY`, or create a
   local `.env` file:

   ```sh
   S2_API_KEY=your_key_here
   ```

2. Fetch the gold references:

   ```sh
   uv run get-gold
   ```

   Each output file is a pretty JSON record named after the source PDF, for
   example `pdfs/gold_set/1706.03762v7.json`.

To also write a combined JSONL file:

```sh
uv run get-gold --combined-output pdfs/gold_set/00.jsonl
```

To split an existing combined JSONL file into pretty per-paper JSON files
without calling the API:

```sh
uv run get-gold --split-combined pdfs/gold_set/00.jsonl
```

To recreate the manifest template from the current PDFs:

```sh
uv run get-gold --init-manifest --force
```

## Running the extraction benchmark

`run_pipelines.py` implements the three extraction pipelines described in
`pipeline_instructions.md`:

- `direct_pdf`: PDF -> LLM -> JSON
- `pymupdf_markdown`: PDF -> PyMuPDF text -> LLM -> JSON
- `pdf_images`: PDF -> rendered page images -> LLM -> JSON

Gemini is the default provider. Set your Gemini API key in `.env`:

```sh
GEMINI_API_KEY=your_key_here
```

If no API key or model is configured and you run the command in a terminal,
`run-benchmark` will prompt for them. The prompted API key is hidden and is not
saved.

Install dependencies and check the planned work:

```sh
uv sync
uv run run-benchmark --dry-run
```

Run all three pipelines against all PDFs:

```sh
uv run run-benchmark
```

Use a specific Gemini model:

```sh
uv run run-benchmark --model gemini-3.1-pro-preview
```

Compare multiple Gemini models in the same benchmark run:

```sh
uv run run-benchmark \
  --model gemini-2.5-flash \
  --model gemini-2.5-pro \
  --model gemini-3.1-flash-lite \
  --model gemini-3.1-pro-preview
```

Repeated and comma-separated model values are both supported. Multi-model runs
use one provider and one API key.

Disable prompts for scripted runs:

```sh
uv run run-benchmark --no-interactive
```

OpenAI remains available as an alternate provider:

```sh
OPENAI_API_KEY=your_key_here uv run run-benchmark --provider openai --model gpt-4o-mini
```

Run one pipeline for one PDF:

```sh
uv run run-benchmark \
  --pipeline pymupdf_markdown \
  --pdf pdfs/1706.03762v7.pdf
```

Outputs are written under `benchmark_runs/<timestamp>/`:

- `outputs/<pipeline>/<paper>.json`: extracted bibliography JSON
- `outputs/<pipeline>/<model_id>/<paper>.json`: extracted JSON for multi-model runs
- `intermediate/pymupdf_markdown/<paper>.md`: PyMuPDF extracted text
- `intermediate/pdf_images/<paper>/page-001.png`: rendered page images
- `raw/<pipeline>/...`: raw model responses and output text
- `metrics.jsonl`: one metrics record per model, paper, and pipeline
- `scores.jsonl`: one accuracy record per model, paper, and pipeline
- `scores_summary.json`: aggregate scores overall, by pipeline, by model, and by pipeline/model
- `run_metadata.json`: run configuration

`run-benchmark` runs evaluation automatically after extraction. Skip that step:

```sh
uv run run-benchmark --no-evaluate
```

Token costs are estimated from `price.jsonl` when the selected provider/model is
listed there. The file stores input/output prices per 1M tokens and can be edited
as model prices change.

```sh
uv run run-benchmark --model gemini-3.1-flash-lite
```

Use a different pricing catalog:

```sh
uv run run-benchmark --price-file path/to/price.jsonl
```

Override catalog prices from the CLI:

```sh
uv run run-benchmark \
  --input-price-per-million INPUT_PRICE \
  --output-price-per-million OUTPUT_PRICE
```

If neither the catalog nor the CLI provides both prices, `estimated_cost_usd` is
left as `null`.

The benchmark runner does not use Semantic Scholar, Crossref, OpenAlex, DBLP, or
other citation databases during extraction.

## Evaluating accuracy

`run-benchmark` evaluates automatically by default. To re-run evaluation or
evaluate an older run, compare extracted outputs to `pdfs/gold_set/*.json`:

```sh
uv run evaluate-benchmark --run-dir benchmark_runs/<timestamp>
```

This writes:

- `scores.jsonl`: one accuracy record per model, paper, and pipeline
- `scores_summary.json`: aggregate scores overall, by pipeline, by model, and by pipeline/model

If `--run-dir` is omitted, the latest directory under `benchmark_runs/` is used.
Evaluate one pipeline or one model:

```sh
uv run evaluate-benchmark \
  --run-dir benchmark_runs/<timestamp> \
  --pipeline pymupdf_markdown

uv run evaluate-benchmark \
  --run-dir benchmark_runs/<timestamp> \
  --model gemini-3.1-flash-lite
```

The evaluator computes:

- Reference count error and absolute reference count error.
- Reference matching precision, recall, and F1.
- Matching method counts for DOI, arXiv ID, title similarity, and raw-reference similarity.
- Field-level precision, recall, and F1 for `title`, `authors`, `year`, `venue`, `doi`, `arxiv_id`, and `url`.
- Exact DOI and arXiv accuracy over matched references with those IDs in gold.
- Average title similarity for matched references.
- Average raw-reference similarity for matched references.
- Author precision, recall, and F1 through the `authors` field metrics.
- Schema validity rate.

Matching is deterministic and local: DOI/arXiv exact matches are preferred, then
title similarity, then raw-reference similarity. It does not use external
citation databases.

## Plotting benchmark results

After evaluation, generate static Matplotlib plots:

```sh
uv run plot-benchmark --run-dir benchmark_runs/<timestamp>
```

If `--run-dir` is omitted, the latest directory under `benchmark_runs/` is used.
Accuracy plots are paper-weighted: each attempted paper contributes equally, not
each reference. If a pipeline attempted a paper but did not produce a score, that
paper is counted as 0 for precision, recall, and F1, and the manifest records a
warning. In multi-model runs, bars are grouped by pipeline with one bar per
model inside each pipeline group.

Plots are written to `benchmark_runs/<timestamp>/plots/` by default:

- `average_f1_per_paper.png`: paper-weighted mean reference F1.
- `average_precision_per_paper.png`: paper-weighted mean reference precision.
- `average_recall_per_paper.png`: paper-weighted mean reference recall.
- `average_wall_clock_time_per_page.png`: mean wall-clock seconds per page.
- `average_error_rate.png`: percentage of attempted papers with an error.
- `average_input_output_tokens_per_page.png`: known input/output tokens per page.
- `average_cost_per_page.png`: estimated USD per page.
- `paper_heatmaps.png`: per-paper heatmaps for accuracy, runtime, tokens, cost, reference count error, and error status.
- `reference_counts_by_pipeline.png`: aggregate chart with one panel per pipeline.
- `reference_counts_by_model.png`: aggregate chart with one panel per model.
- `reference_counts_by_pipeline_<pipeline>.png`: correct reference counts by model for one pipeline.
- `reference_counts_by_model_<model_id>.png`: correct reference counts by pipeline for one model.
- `manifest.json`: source files, computed values, generated plots, and warnings.

Reference count plots include only correctly matched references. They are scoped
by source paper and keyed by matched gold reference. Each bar shows total
correct references; the orange bar segment is the count found only by that
pipeline/model within the chart. Correct-reference plots recompute the same
DOI/arXiv/title/raw-reference matching as `evaluate-benchmark`; use
`--title-threshold` or `--raw-threshold` if you evaluated with non-default
thresholds.

Plot one pipeline/model or change output format:

```sh
uv run plot-benchmark \
  --run-dir benchmark_runs/<timestamp> \
  --pipeline pdf_images \
  --model gemini-3.1-flash-lite \
  --format svg
```

Generate one dashboard per paper:

```sh
uv run plot-benchmark --run-dir benchmark_runs/<timestamp> --paper-plots
```

Runtime plots include failed or timed-out attempts because they consumed wall
time. Token plots only include rows where the provider reported both input and
output tokens. Cost plots only include rows where `estimated_cost_usd` is
available. Missing token or cost rows are listed in `manifest.json`.

## Gold Format

Each per-paper `.json` file contains one source-paper record. The combined
`.jsonl` file contains the same records, one compact JSON object per line:

```json
{
  "pdf": "pdfs/1706.03762v7.pdf",
  "queryId": "ARXIV:1706.03762",
  "source": {
    "paperId": "...",
    "externalIds": {},
    "title": "...",
    "authors": [{"authorId": "...", "name": "..."}],
    "year": 2017,
    "venue": "...",
    "publicationDate": "...",
    "url": "...",
    "citationCount": 0,
    "referenceCount": 0
  },
  "referenceCount": 0,
  "references": [
    {
      "index": 0,
      "isInfluential": false,
      "intents": [],
      "citedPaper": {
        "paperId": "...",
        "externalIds": {},
        "title": "...",
        "authors": [{"authorId": "...", "name": "..."}],
        "year": 2014,
        "venue": "...",
        "publicationDate": "...",
        "url": "...",
        "citationCount": 0,
        "referenceCount": null
      }
    }
  ]
}
```
