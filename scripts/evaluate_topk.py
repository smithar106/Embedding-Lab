"""
Top-K sensitivity: rerun the same RAG evaluation at K=1,2,3,5 — nothing else changes.

This exposes the causal chain the earlier phases set up:

    embedding choice -> retrieval ranking -> context selection (top-K)
                     -> generation quality

At K=1 a model that ranks the relevant chunk second will have *no* relevant
evidence in its context, so the answer should differ (refuse or go wrong) —
while a model that ranks it first stays correct. Increasing K pulls the relevant
evidence back into the window and the differences vanish. This also measures the
production tradeoff: more K = more input tokens / latency for less marginal gain.

Usage (from the project root, after ingesting the corpus):

    python scripts/evaluate_topk.py
    python scripts/evaluate_topk.py --top-ks 1,2,3,5 --focus-question hq003
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from models.embedding_models import MODELS, embed_query  # noqa: E402
from generation.rag import run_rag  # noqa: E402
from evaluation.metrics import recall_at_k  # noqa: E402
from evaluation.answer_metrics import citation_validity, fact_coverage, is_refusal  # noqa: E402
from evaluation.dataset import load_golden_questions  # noqa: E402


def _document_ids(retrieved: list[dict]) -> list[str]:
    """Dedupe retrieved chunks to document ids, preserving rank order."""
    seen, out = set(), []
    for r in retrieved:
        d = r["document_id"]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _run_level(questions, models, k) -> dict:
    """Return {model: aggregate metrics} for one top_k value."""
    level = {}
    for model in models:
        recalls, covs, valids, refusals, in_tokens, total_ms = [], [], [], [], [], []
        for q in questions:
            r = run_rag(q["question"], model, top_k=k)
            relevant = q["relevant_document_ids"]
            recalls.append(recall_at_k(_document_ids(r["retrieved"]), relevant, k))
            fc = fact_coverage(r["answer"], q.get("expected_facts", []))
            covs.append(fc["coverage"] if fc["coverage"] is not None else 0.0)
            valids.append(citation_validity(r["citations"], r["retrieved_chunk_ids"])["ratio"])
            refusals.append(1.0 if is_refusal(r["answer"]) else 0.0)
            in_tokens.append(r["usage"]["input_tokens"] if r["usage"] else 0)
            total_ms.append(r["total_timing_ms"])
        n = len(questions)
        level[model] = {
            "recall_at_k": round(sum(recalls) / n, 3),
            "fact_coverage": round(sum(covs) / n, 3),
            "citation_validity": round(sum(valids) / n, 3),
            "refusal_rate": round(sum(refusals) / n, 3),
            "avg_input_tokens": round(sum(in_tokens) / n, 1),
            "avg_total_ms": round(sum(total_ms) / n, 1),
        }
    return level


def print_matrix(matrix: dict, models: list[str], top_ks: list[int]) -> None:
    rows = [
        ("Recall@K", "recall_at_k"),
        ("Fact coverage", "fact_coverage"),
        ("Citation validity", "citation_validity"),
        ("Refusal rate", "refusal_rate"),
        ("Avg input tokens", "avg_input_tokens"),
        ("Avg total ms", "avg_total_ms"),
    ]
    header = "                " + "".join(f"{k:>9}" for k in top_ks)
    print(header)
    for model in models:
        print(f"{model.upper():<14}" + "".join(f"{k:>9}" for k in top_ks))
        for label, key in rows:
            print(f"  {label:<12}" + "".join(f"{matrix[k][model][key]:>9}" for k in top_ks))
        print()


def print_focus(question: dict, models: list[str], top_ks: list[int]) -> None:
    print("=" * 72)
    print(f"FOCUS — {question['question_id']}: {question['question']}")
    print(f"Ground truth: {question['relevant_document_ids']}")
    print("=" * 72)
    for k in top_ks:
        print(f"\n--- top_k = {k} ---")
        for model in models:
            r = run_rag(question["question"], model, top_k=k)
            retrieved = [c["document_id"] for c in r["retrieved"]]
            print(f"  {model.upper():<8} retrieved: {retrieved}")
            print(f"           answer: {r['answer'][:140]}{'…' if r['answer'] and len(r['answer']) > 140 else ''}")
            print(f"           citations: {r['citations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-K sensitivity for the RAG pipeline")
    parser.add_argument("--questions", default="data/evaluation/golden_questions_hard.json")
    parser.add_argument("--top-ks", default="1,2,3,5")
    parser.add_argument("--focus-question", default="hq003")
    args = parser.parse_args()

    questions = load_golden_questions(args.questions)
    models = list(MODELS)
    top_ks = [int(k) for k in args.top_ks.split(",")]

    # Warm embedding models so latency is warm.
    for m in models:
        embed_query("warmup", m)

    matrix = {}
    for k in top_ks:
        matrix[k] = _run_level(questions, models, k)

    print("Top-K sensitivity matrix (identical corpus, questions, model, prompt, temp=0)\n")
    print_matrix(matrix, models, top_ks)

    focus = next((q for q in questions if q["question_id"] == args.focus_question), None)
    if focus:
        print_focus(focus, models, top_ks)

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "questions_file": args.questions,
        "top_ks": top_ks,
        "embedding_models": {m: MODELS[m].hf_id for m in models},
        "num_questions": len(questions),
        "matrix": {str(k): matrix[k] for k in top_ks},
    }
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"rag_topk_{ts}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
