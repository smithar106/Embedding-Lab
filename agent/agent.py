"""A minimal agent with ONE tool: retrieval_search.

The mechanics are explicit and readable — no LangChain / LangGraph / CrewAI /
AutoGen. The loop is:

    1. LLM decision: answer directly, or request retrieval_search (function call)
    2. (if requested) DETERMINISTIC python execution of retrieval_search()
    3. tool result returned to the LLM
    4. LLM produces the final grounded answer

The separation to notice is **LLM decision** vs **deterministic python tool
execution**: the model only ever *asks for* evidence; our code (not the model)
actually runs the retrieval.
"""
from __future__ import annotations

import json
import time

from agent.prompts import AGENT_SYSTEM
from evaluation.answer_metrics import citation_validity, is_refusal
from generation.generator import get_client, get_generation_model
from generation.prompts import parse_citations
from tools.retrieval_tool import TOOL_SCHEMA, retrieval_search


def run_agent(
    user_query: str,
    *,
    default_top_k: int = 2,
    default_embedding_model: str = "bge",
) -> dict:
    """Run the agent for one user query and return a full structured trace."""
    client = get_client()
    model = get_generation_model()

    trace = {
        "user_query": user_query,
        "generation_model": model,
        "tool_requested": False,
        "tool_name": None,
        "tool_arguments": None,
        "tool_results": [],
        "final_answer": None,
        "citations": [],
        "citation_validity": None,
        "refused": False,
        "latency": {},
        "error": None,
    }

    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_query},
    ]

    try:
        # ---- 1. LLM decision -------------------------------------------------
        t0 = time.perf_counter()
        decision = client.chat.completions.create(
            model=model, messages=messages, tools=[TOOL_SCHEMA], temperature=0.0
        )
        decision_ms = (time.perf_counter() - t0) * 1000.0
        trace["latency"]["agent_decision_ms"] = round(decision_ms, 2)

        msg = decision.choices[0].message
        tool_calls = list(msg.tool_calls or [])

        if tool_calls:
            tc = tool_calls[0]
            trace["tool_requested"] = True
            trace["tool_name"] = tc.function.name
            trace["tool_arguments"] = json.loads(tc.function.arguments or "{}")

            # ---- 2. deterministic python tool execution ---------------------
            args = trace["tool_arguments"]
            t1 = time.perf_counter()
            result = retrieval_search(
                query=args.get("query", user_query),
                top_k=int(args.get("top_k") or default_top_k),
                embedding_model=default_embedding_model,
            )
            exec_ms = (time.perf_counter() - t1) * 1000.0
            trace["latency"]["tool_execution_ms"] = round(exec_ms, 2)
            trace["tool_results"] = result["results"]

            # ---- 3. return the tool result to the model ---------------------
            messages.append({
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

        # ---- 4. final grounded answer ---------------------------------------
        if tool_calls:
            # A tool was used: send the tool result back and ask for the answer.
            t2 = time.perf_counter()
            final = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0
            )
            gen_ms = (time.perf_counter() - t2) * 1000.0
            trace["latency"]["final_generation_ms"] = round(gen_ms, 2)
            final_answer = final.choices[0].message.content or ""
        else:
            # No tool: the decision message IS the final answer.
            final_answer = msg.content or ""
            trace["latency"]["final_generation_ms"] = 0.0

        trace["final_answer"] = final_answer
        trace["citations"] = parse_citations(final_answer)
        trace["refused"] = is_refusal(final_answer)

        # Validate citations against the chunks the tool actually returned.
        if trace["tool_requested"]:
            supplied = [r["chunk_id"] for r in trace["tool_results"]]
            trace["citation_validity"] = citation_validity(trace["citations"], supplied)
    except Exception as exc:  # noqa: BLE001
        trace["error"] = str(exc)

    decision_ms = trace["latency"].get("agent_decision_ms", 0.0)
    exec_ms = trace["latency"].get("tool_execution_ms", 0.0)
    gen_ms = trace["latency"].get("final_generation_ms", 0.0)
    trace["latency"]["total_ms"] = round(decision_ms + exec_ms + gen_ms, 2)
    return trace
