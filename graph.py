"""
Graph wiring for the Vendor Contract Review Copilot.

Same topology as the narrative-framing-comparator project (proven working
pattern), applied to contract clauses instead of news articles:

        START
          |
   dispatch_extraction  --Send--> extractor (parallel, one per clause category)
          |                              |
          '------------------------------'
                       |
                  risk_flagger
                    /      \\
        (sufficient)        (ambiguous categories found, rounds left)
                |                      |
           report_writer      dispatch_refinement --Send--> extractor_deeper
                |                                                |
               END                                          risk_flagger (loop back)

Bounded by `max_refinement_rounds` so the graph always terminates.
"""

from __future__ import annotations

import os

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from config import GROQ_MODEL
from clause_extractor import extract_clause, extract_clause_deeper
from risk_flagger import assess_risk
from report_writer import write_report
from judge import audit_report
from schemas import DEFAULT_PLAYBOOK, ReviewState, RiskAssessment, ClauseCategory

DEFAULT_MAX_REFINEMENT_ROUNDS = 1
ENABLE_AUDITOR = os.getenv("ENABLE_AUDITOR", "false").lower() == "true"
DEFAULT_CATEGORIES: list[ClauseCategory] = [
    "liability",
    "auto_renewal",
    "data_handling",
    "termination",
]


# ---- Nodes -----------------------------------------------------------------


def extractor_node(state: dict) -> dict:
    category = state["category"]
    contract_text = state["contract_text"]
    extraction = extract_clause(category, contract_text)
    return {"extractions": [extraction]}


def extractor_deeper_node(state: dict) -> dict:
    category = state["category"]
    contract_text = state["contract_text"]
    reason = state["ambiguity_reason"]
    extraction = extract_clause_deeper(category, contract_text, reason)
    return {"extractions": [extraction]}


def _latest_by_category(extractions: list) -> list:
    by_category: dict[str, object] = {}
    for e in extractions:
        by_category[e.category] = e
    return list(by_category.values())


def risk_flagger_node(state: ReviewState) -> dict:
    deduped = _latest_by_category(state["extractions"])
    playbook = state.get("playbook", DEFAULT_PLAYBOOK)
    result: RiskAssessment = assess_risk(playbook, deduped)

    round_num = state.get("refinement_round", 0)
    max_rounds = state.get("max_refinement_rounds", DEFAULT_MAX_REFINEMENT_ROUNDS)
    if not result.is_sufficient and round_num >= max_rounds:
        result.is_sufficient = True

    return {"assessment": result}


def report_writer_node(state: ReviewState) -> dict:
    deduped = _latest_by_category(state["extractions"])
    report = write_report(deduped, state["assessment"])
    return {"final_report": report}

def auditor_node(state: ReviewState) -> dict:
    deduped = _latest_by_category(state["extractions"])
    audit = audit_report(deduped, state["assessment"], state["final_report"])
    result = {"audited": True, "judge_model": GROQ_MODEL}
    if not audit.is_valid and audit.corrected_report is not None:
        result["final_report"] = audit.corrected_report
        result["report_corrected"] = True
    else:
        result["report_corrected"] = False
    return result

def increment_round_node(state: ReviewState) -> dict:
    return {"refinement_round": state.get("refinement_round", 0) + 1}


# ---- Conditional routing ----------------------------------------------------


def dispatch_extraction(state: ReviewState) -> list[Send]:
    categories = state.get("categories", DEFAULT_CATEGORIES)
    return [
        Send("extractor", {"category": c, "contract_text": state["contract_text"]})
        for c in categories
    ]


def route_after_risk_flagger(state: ReviewState) -> str:
    if state["assessment"].is_sufficient:
        return "report_writer"
    return "increment_round"


def dispatch_refinement(state: ReviewState) -> list[Send]:
    ambiguous = set(state["assessment"].ambiguous_categories)
    sends = []
    for category in ambiguous:
        sends.append(
            Send(
                "extractor_deeper",
                {
                    "category": category,
                    "contract_text": state["contract_text"],
                    "ambiguity_reason": "Risk flagger could not confidently assess this "
                    "category from the initial extraction.",
                },
            )
        )
    return sends


# ---- Graph assembly ----------------------------------------------------------


def build_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("extractor", extractor_node)
    graph.add_node("extractor_deeper", extractor_deeper_node)
    graph.add_node("risk_flagger", risk_flagger_node)
    graph.add_node("increment_round", increment_round_node)
    graph.add_node("report_writer", report_writer_node)

    graph.add_conditional_edges(START, dispatch_extraction, ["extractor"])
    graph.add_edge("extractor", "risk_flagger")

    graph.add_conditional_edges(
        "risk_flagger", route_after_risk_flagger, ["report_writer", "increment_round"]
    )
    graph.add_conditional_edges(
        "increment_round", dispatch_refinement, ["extractor_deeper"]
    )
    graph.add_edge("extractor_deeper", "risk_flagger")

    if ENABLE_AUDITOR:
        graph.add_node("auditor", auditor_node)
        graph.add_edge("report_writer", "auditor")
        graph.add_edge("auditor", END)
    else:
        graph.add_edge("report_writer", END)

    return graph.compile()


review_graph = build_graph()
