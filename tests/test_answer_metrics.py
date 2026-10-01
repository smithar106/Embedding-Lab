"""Tests for the answer-level evaluation metrics (pure functions)."""
from evaluation.answer_metrics import (
    citation_relevance,
    citation_validity,
    fact_coverage,
    is_refusal,
)
from generation.prompts import parse_citations


def test_parse_citations():
    answer = "The capacitor stores energy in an electric field [capacitor#0]. See [battery-housing#1]."
    assert parse_citations(answer) == ["capacitor#0", "battery-housing#1"]


def test_parse_citations_none():
    assert parse_citations("no citations here") == []


def test_fact_coverage_full():
    fc = fact_coverage("A capacitor stores energy in an electric field.", ["capacitor", "electric field"])
    assert fc["coverage"] == 1.0
    assert fc["missing_facts"] == []


def test_fact_coverage_partial():
    fc = fact_coverage("A capacitor stores energy.", ["capacitor", "electric field"])
    assert fc["coverage"] == 0.5
    assert fc["missing_facts"] == ["electric field"]


def test_fact_coverage_is_case_insensitive():
    fc = fact_coverage("It is a CAPACITOR.", ["capacitor"])
    assert fc["coverage"] == 1.0


def test_fact_coverage_no_expected_facts():
    fc = fact_coverage("anything", [])
    assert fc["coverage"] is None


def test_citation_validity_all_valid():
    cv = citation_validity(["a#0", "b#0"], ["a#0", "b#0", "c#0"])
    assert cv["ratio"] == 1.0
    assert cv["invalid"] == 0
    assert cv["invalid_citations"] == []


def test_citation_validity_detects_hallucinated():
    cv = citation_validity(["a#0", "zzz#9"], ["a#0", "b#0"])
    assert cv["ratio"] == 0.5
    assert cv["invalid_citations"] == ["zzz#9"]


def test_citation_validity_no_citations_is_valid():
    cv = citation_validity([], ["a#0"])
    assert cv["ratio"] == 1.0


def test_citation_relevance():
    cr = citation_relevance(["capacitor#0", "lithium-ion#0"], ["capacitor"])
    assert cr["ratio"] == 0.5
    assert cr["relevant_citations"] == ["capacitor#0"]


def test_is_refusal():
    assert is_refusal("The available evidence is insufficient to answer this question.")
    assert is_refusal("I cannot answer because the evidence does not contain the information.")
    assert not is_refusal("A capacitor stores energy in an electric field.")
