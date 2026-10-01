"""
Evaluate the full RAG pipeline (retrieval + generation) against golden questions.

For every question and every embedding model it runs the controlled pipeline,
then measures answer quality: expected-fact coverage, citation validity,
citation relevance, and latency / token usage. It also runs the
insufficient-evidence questions to check the system abstains rather than
inventing an answer.

Results are saved to ``results/rag_evaluation_<timestamp>.json``.

Usage (from the project root, after ingesting the corpus):

    python scripts/evaluate_rag.py
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
from generation.generator import get_generation_model, get_generation_temperature  # noqa: E402
from generation.rag import run_rag  # noqa: E402
from evaluation.answer_metrics import (  # noqa: E402
    citation_relevance,
    citation_validity,
    fact_coverage,
    is_refusal,
)
from evaluation.dataset import load_golden_questions  # noqa: E402


def _load_simple_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"question_id": q.get("question_id", f"q{i}"), "question": q["question"]}
            for i, q in enumerate(data)]


def _answer_metrics(q, r) -> dict:
    facts = q.get("expected_facts", [])
    validity = citation_validity(r["citations"], r["retrieved_chunk_ids"])
    relevance = citation_relevance(r["citations"], q.get("relevant_document_ids", []))
    return {
        "fact_coverage": fact_coverage(r["answer"], facts),
        "citation_validity": validity,
        "citation_relevance": relevance,
        "refusal": is_refusal(r["answer"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers against golden questions")
    parser.add_argument("--questions", default="data/evaluation/golden_questions_hard.json")
    parser.add_argument("--insufficient", default="data/evaluation/insufficient_questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_golden_questions(args.questions)
    models = list(MODELS)

    # Warm the embedding models once so latency excludes model loading.
    for m in models:
        embed_query("warmup", m)

    # ---- Golden questions ---------------------------------------------------
    per_question = {}
    for q in questions:
        qid = q["question_id"]
        model_results = {}
        for model in models:
            r = run_rag(q["question"], model, top_k=args.top_k)
            r["answer_metrics"] = _answer_metrics(q, r)
            model_results[model] = r
        per_question[qid] = {"question": q["question"], "relevant": q["relevant_document_ids"],
                             "expected_facts": q.get("expected_facts", []), "models": model_results}

    # ---- Insufficient-evidence questions -----------------------------------
    insuff_path = Path(args.insufficient)
    insufficiency = []
    if insuff_path.exists():
        for q in _load_simple_questions(insuff_path):
            row = {"question": q["question"]}
            for model in models:
                r = run_rag(q["question"], model, top_k=args.top_k)
                row[model] = {"refused": is_refusal(r["answer"]), "answer": r["answer"],
                              "citations": r["citations"], "retrieved": r["retrieved_chunk_ids"]}
            insufficiency.append(row)

    # ---- Aggregate answer metrics per model ---------------------------------
    aggregate = {}
    for model in models:
        covs, valids, gens, toks = [], [], [], 0
        for q in questions:
            m = per_question[q["question_id"]]["models"][model]
            cov = m["answer_metrics"]["fact_coverage"]
            if cov["coverage"] is not None:
                covs.append(cov["coverage"])
            valids.append(m["answer_metrics"]["citation_validity"]["ratio"])
            gens.append(m["generation_timing_ms"])
            if m["usage"] and m["usage"].get("total_tokens"):
                toks += m["usage"]["total_tokens"]
        n = len(questions)
        aggregate[model] = {
            "avg_fact_coverage": round(sum(covs) / len(covs), 3) if covs else None,
            "avg_citation_validity": round(sum(valids) / n, 3),
            "avg_generation_ms": round(sum(gens) / n, 2),
            "total_tokens": toks,
        }

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generation_model": get_generation_model(),
        "temperature": get_generation_temperature(),
        "top_k": args.top_k,
        "embedding_models": {name: MODELS[name].hf_id for name in models},
        "num_questions": len(questions),
        "aggregate": aggregate,
        "per_question": per_question,
        "insufficiency": insufficiency,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"rag_evaluation_{ts}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ---- Report -------------------------------------------------------------
    print("=" * 72)
    print("RAG EVALUATION — aggregate answer metrics")
    print("=" * 72)
    print(f"Generation model : {payload['generation_model']}")
    print(f"Temperature      : {payload['temperature']}")
    print(f"Top-K            : {payload['top_k']}")
    print()
    header = f"{'MODEL':<8} {'Fact cov':>9} {'Cite valid':>11} {'Gen ms':>8} {'Tokens':>8}"
    print(header)
    print("-" * 72)
    for model in models:
        a = aggregate[model]
        print(f"{model.upper():<8} {str(a['avg_fact_coverage']):>9} {a['avg_citation_validity']:>11} "
              f"{a['avg_generation_ms']:>8} {a['total_tokens']:>8}")
    print()

    print("Insufficient-evidence questions (expected: refuse):")
    for row in insufficiency:
        print(f"  {row['question']}")
        for model in models:
            ok = "✓ refused" if row[model]["refused"] else "✗ DID NOT REFUSE"
            print(f"    {model.upper():<8} {ok}")
    print()

    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
