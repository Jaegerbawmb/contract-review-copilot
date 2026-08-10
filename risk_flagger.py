"""
Risk-flagger agent.

Takes ALL per-category clause extractions plus the company's playbook
(their own risk thresholds) and assigns a risk level to each category:
compliant / non_standard / high_risk / not_addressed.

Also decides whether any category's extraction was too thin/ambiguous to
assess confidently -- if so, names those categories so the graph can route
back to the extractor for a deeper pass. This is what makes the graph
cyclic rather than a straight pipeline.
"""

from __future__ import annotations

import json
from config import get_llm
from llm_cache import cached_call
from schemas import ClauseExtraction, RiskAssessment
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm
from schemas import ClauseExtraction, RiskAssessment

_INSTRUCTIONS = """You are a contract risk analyst reviewing extracted clauses against \
a company's own risk playbook (their internal policy for what's acceptable).

For each extracted category, assign a risk_level:
- "compliant": clause terms are within the playbook's acceptable range
- "non_standard": clause deviates from playbook but isn't necessarily dangerous -- worth \
negotiating but not a blocker
- "high_risk": clause meaningfully exposes the company to harm per the playbook's own \
threshold (e.g. uncapped liability, no exit option, data resale rights)
- "not_addressed": the contract is silent on this category (this itself can be a risk \
depending on category -- silence on data deletion is worse than silence on liability caps)

For each flag, give a rationale grounded in the extracted key_terms, and cite which \
playbook rule it's being checked against.

Separately: if any category's extraction has confidence "low" or needs_deeper_pass is \
true, OR you personally find an extraction too thin to assess confidently, list that \
category in ambiguous_categories and set is_sufficient to false. Otherwise is_sufficient \
is true and ambiguous_categories is empty.

Respond via the structured output tool only."""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _INSTRUCTIONS),
        (
            "human",
            "PLAYBOOK (JSON):\n{playbook_json}\n\n"
            "EXTRACTED CLAUSES (JSON):\n{extractions_json}",
        ),
    ]
)


def assess_risk(playbook: dict[str, str], extractions: list[ClauseExtraction]) -> RiskAssessment:
    llm = get_llm().with_structured_output(RiskAssessment)
    chain = _prompt | llm
    inputs = {
        "playbook_json": json.dumps(playbook, ensure_ascii=False, indent=2),
        "extractions_json": json.dumps(
            [e.model_dump() for e in extractions], ensure_ascii=False, indent=2
        ),
    }
    return cached_call("assess_risk", inputs, lambda: chain.invoke(inputs), RiskAssessment)