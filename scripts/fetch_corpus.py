"""
Fetch full text for every document in the source catalog and write a corpus.

The source catalog (``data/catalog/sources.jsonl``) holds the CURATED metadata
for each document — collection, topics, source_organization, source_url,
publication_date, and (critically) the verified reuse ``license``. It does NOT
hold the full text.

This command fetches each ``source_url``, extracts the article's clean main text
(HTML via trafilatura), and writes ``data/catalog/corpus.jsonl`` — one JSON
object per line, ready for ``ingest_dataset``:

    python -m scripts.fetch_corpus                     # fetch everything
    python -m scripts.fetch_corpus --limit 10          # first 10 (a smoke test)
    python -m scripts.fetch_corpus --collection energy # just one collection

Licensing is enforced upstream (in the catalog): only documents whose reuse
rights have been checked belong in ``sources.jsonl``. This script records
``retrieved_at`` so provenance is attached to every document at ingest time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog" / "sources.jsonl"
DEFAULT_OUT = ROOT / "data" / "catalog" / "corpus.jsonl"

HEADERS = {
    "User-Agent": "EmbeddingLab/1.0 (educational retrieval demo; contact: smithar106@gmail.com)",
}


def _load_catalog(path: Path) -> list[dict]:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


def _fetch_text(url: str) -> str | None:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
    if not text or not text.strip():
        return None
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full text for the source catalog")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0, help="fetch only the first N docs")
    parser.add_argument("--collection", default=None, help="filter to one collection")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    catalog = _load_catalog(Path(args.catalog))
    if args.collection:
        catalog = [d for d in catalog if d.get("collection") == args.collection]
    if args.limit:
        catalog = catalog[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fetched, failed = 0, 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, doc in enumerate(catalog):
            url = doc.get("source_url")
            if not url:
                print(f"[skip] {doc['document_id']}: no source_url")
                failed += 1
                continue
            try:
                text = _fetch_text(url)
                if text is None:
                    print(f"[empty] {doc['document_id']}: {url}")
                    failed += 1
                    continue
                record = {**doc, "text": text,
                          "retrieved_at": datetime.now(timezone.utc).isoformat()}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[ ok ] {doc['document_id']}: {len(text):,} chars")
                fetched += 1
            except Exception as exc:
                print(f"[fail] {doc['document_id']}: {exc}")
                failed += 1
            if args.delay and i < len(catalog) - 1:
                time.sleep(args.delay)

    print(f"\nFetched {fetched}, failed {failed} of {len(catalog)}")
    if fetched == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
