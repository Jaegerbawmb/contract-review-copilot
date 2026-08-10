"""
Report-writer agent.

Final node. Takes the (now-sufficient) risk assessment plus the raw
extractions and writes a coherent report a founder could actually act on:
what's fine, what needs negotiating, what's a dealbreaker, and what to do
next.
"""

from __future__ import annotations

import json
from config import get_llm
from llm_cache import cached_call
from schemas import ClauseExtraction, FinalReport, RiskAssessment
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm
from schemas import ClauseExtraction, FinalReport, RiskAssessment

_INSTRUCTIONS = """You are writing the final section of a vendor contract risk report \
for a startup founder who is not a lawyer.

Write:
- contract_summary: 2-3 plain-English sentences on what kind of contract this is and \
its overall shape
- categories_reviewed: list the categories that were assessed
- risk_flags: pass through the assessment's flags, tightening wording if needed
- overall_risk_level: "low", "medium", or "high" -- your holistic read given the flags \
(one high_risk category alone can justify "high" if it's serious enough)
- recommended_actions: 2-5 concrete, specific next steps (e.g. "negotiate the liability \
cap down to 12 months fees" not "review liability section")

Respond via the structured output tool only."""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _INSTRUCTIONS),
        (
            "human",
            "EXTRACTED CLAUSES (JSON):\n{extractions_json}\n\n"
            "RISK ASSESSMENT (JSON):\n{assessment_json}",
        ),
    ]
)


def write_report(extractions: list[ClauseExtraction], assessment: RiskAssessment) -> FinalReport:
    llm = get_llm().with_structured_output(FinalReport)
    chain = _prompt | llm
    inputs = {
        "extractions_json": json.dumps(
            [e.model_dump() for e in extractions], ensure_ascii=False, indent=2
        ),
        "assessment_json": json.dumps(assessment.model_dump(), ensure_ascii=False, indent=2),
    }
    return cached_call("write_report", inputs, lambda: chain.invoke(inputs), FinalReport)