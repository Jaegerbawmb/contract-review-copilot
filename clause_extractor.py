"""
Clause extractor agent.

Runs once per clause category (liability, auto_renewal, data_handling,
termination), fanned out in parallel via `Send`. For each category, it
searches the full contract text for relevant language and extracts a
structured summary + key terms (amounts, day counts, conditions).

A "deeper pass" variant is used when the risk-flagger can't confidently
assess a category from the first extraction -- same task, but with a
sharper prompt targeting exactly what was ambiguous.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from config import get_llm
from llm_cache import cached_call
from schemas import ClauseCategory, ClauseExtraction
from config import get_llm

_CATEGORY_HINTS = {
    "liability": "limitation of liability, liability caps, indemnification limits, damages exclusions",
    "auto_renewal": "renewal terms, auto-renewal, evergreen clauses, notice-to-cancel periods",
    "data_handling": "data ownership, data use rights, data deletion/retention, use of data for training or resale",
    "termination": "termination for convenience, termination for cause, notice periods, lock-in/minimum terms",
}

_BASE_INSTRUCTIONS = """You are a contract review analyst. You will be given the full \
text of a vendor contract and asked to focus on ONE clause category.

Category to analyze: {category}
What to look for: {hints}

1. found: true if the contract addresses this category anywhere, false if it's silent on it
2. summary: a plain-English paraphrase of what the relevant clause(s) say (do not quote \
more than a few words verbatim)
3. key_terms: extracted specifics relevant to risk assessment -- e.g. dollar amounts, \
day counts, percentages, named conditions. Short list, your own words.
4. confidence: how confident you are in this reading (low/medium/high)
5. needs_deeper_pass: true only if the language is genuinely ambiguous, contradictory, \
or split across multiple hard-to-reconcile sections
6. deeper_pass_reason: if needs_deeper_pass is true, say specifically what's ambiguous

Respond via the structured output tool only."""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_INSTRUCTIONS),
        ("human", "CONTRACT TEXT:\n{contract_text}"),
    ]
)

_deeper_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_INSTRUCTIONS),
        (
            "human",
            "CONTRACT TEXT:\n{contract_text}\n\n"
            "A prior pass flagged this specific ambiguity that needs resolving:\n"
            "{ambiguity_reason}\n\n"
            "Re-read the contract specifically with this question in mind and give "
            "your most decisive reading -- avoid defaulting back to needs_deeper_pass "
            "unless the contract genuinely contains zero signal on this point.",
        ),
    ]
)


def extract_clause(category: ClauseCategory, contract_text: str) -> ClauseExtraction:
    llm = get_llm().with_structured_output(ClauseExtraction)
    chain = _prompt | llm
    inputs = {
        "category": category,
        "hints": _CATEGORY_HINTS[category],
        "contract_text": contract_text,
    }
    result = cached_call("extract_clause", inputs, lambda: chain.invoke(inputs), ClauseExtraction)
    result.category = category
    return result


def extract_clause_deeper(
    category: ClauseCategory, contract_text: str, ambiguity_reason: str
) -> ClauseExtraction:
    llm = get_llm().with_structured_output(ClauseExtraction)
    chain = _deeper_prompt | llm
    inputs = {
        "category": category,
        "hints": _CATEGORY_HINTS[category],
        "contract_text": contract_text,
        "ambiguity_reason": ambiguity_reason,
    }
    result = cached_call(
        "extract_clause_deeper", inputs, lambda: chain.invoke(inputs), ClauseExtraction
    )
    result.category = category
    result.needs_deeper_pass = False  # deeper pass is terminal
    return result