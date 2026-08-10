from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import get_judge_llm
from schemas import ClauseExtraction, FinalReport, RiskAssessment


class AuditResult(BaseModel):
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    corrected_report: FinalReport | None = None


_INSTRUCTIONS = """You are auditing a contract risk report for internal consistency \
before it reaches a founder. Check: does each risk_flag's risk_level match its own \
rationale; does the rationale reference real key_terms from the extraction; is \
overall_risk_level consistent with individual flags; are recommended_actions concrete.

If everything checks out, set is_valid=true and leave issues/corrected_report empty.
If you find real problems, set is_valid=false, list the issues, and provide a
corrected_report with the same structure but the issues fixed.

Respond via the structured output tool only."""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _INSTRUCTIONS),
        (
            "human",
            "EXTRACTED CLAUSES (JSON):\n{extractions_json}\n\n"
            "RISK ASSESSMENT (JSON):\n{assessment_json}\n\n"
            "DRAFT REPORT (JSON):\n{report_json}",
        ),
    ]
)


def audit_report(
    extractions: list[ClauseExtraction], assessment: RiskAssessment, report: FinalReport
) -> AuditResult:
    llm = get_judge_llm().with_structured_output(AuditResult)
    chain = _prompt | llm
    return chain.invoke(
        {
            "extractions_json": json.dumps([e.model_dump() for e in extractions], ensure_ascii=False, indent=2),
            "assessment_json": json.dumps(assessment.model_dump(), ensure_ascii=False, indent=2),
            "report_json": json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        }
    )