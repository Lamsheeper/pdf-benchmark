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
