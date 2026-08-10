from __future__ import annotations

from typing import Annotated, Literal, TypedDict
import operator

from pydantic import BaseModel, Field

ClauseCategory = Literal["liability", "auto_renewal", "data_handling", "termination"]

DEFAULT_PLAYBOOK: dict[str, str] = {
    "liability": "Liability should be capped at 12 months of fees paid. Uncapped or unusually high liability caps are high risk.",
    "auto_renewal": "Auto-renewal terms must include at least 30 days advance notice before renewal. Silent/evergreen renewal with no notice is high risk.",
    "data_handling": "Vendor must not use company data for their own model training or resell it, and must support data deletion within 30 days of termination.",
    "termination": "Company must be able to terminate for convenience with 60 days notice or less. Long lock-in periods are high risk.",
}


class ClauseExtraction(BaseModel):
    category: ClauseCategory
    found: bool = Field(..., description="Whether the contract addresses this category at all")
    summary: str = Field(default="", description="Paraphrased summary of what the clause says")
    key_terms: list[str] = Field(default_factory=list, description="Extracted specifics: amounts, day counts, conditions")
    confidence: Literal["low", "medium", "high"] = "medium"
    needs_deeper_pass: bool = False
    deeper_pass_reason: str = ""


class RiskFlag(BaseModel):
    category: ClauseCategory
    risk_level: Literal["compliant", "non_standard", "high_risk", "not_addressed"]
    rationale: str
    playbook_reference: str


class RiskAssessment(BaseModel):
    flags: list[RiskFlag] = Field(default_factory=list)
    ambiguous_categories: list[str] = Field(default_factory=list)
    is_sufficient: bool = True


class FinalReport(BaseModel):
    contract_summary: str
    categories_reviewed: list[ClauseCategory]
    risk_flags: list[RiskFlag]
    overall_risk_level: Literal["low", "medium", "high"]
    recommended_actions: list[str]


class ReviewState(TypedDict, total=False):
    contract_text: str
    playbook: dict[str, str]
    categories: list[ClauseCategory]

    extractions: Annotated[list[ClauseExtraction], operator.add]

    assessment: RiskAssessment
    refinement_round: int
    max_refinement_rounds: int
    final_report: FinalReport
    audited: bool
    report_corrected: bool
