"""
Exercises the compiled graph end-to-end WITHOUT calling a real Gemini API,
by monkeypatching the agent functions. Proves fan-out, risk-flagger
routing, refinement cycle, and termination are correct.

Run: python test_graph_flow.py
"""

from unittest.mock import patch

from schemas import ClauseExtraction, RiskAssessment, RiskFlag, FinalReport


def fake_extract_clause(category, contract_text):
    low_conf = category == "data_handling"
    return ClauseExtraction(
        category=category,
        found=True,
        summary=f"{category} summary",
        key_terms=[f"{category} term"],
        confidence="low" if low_conf else "high",
        needs_deeper_pass=low_conf,
        deeper_pass_reason="split across sections" if low_conf else "",
    )


def fake_extract_clause_deeper(category, contract_text, ambiguity_reason):
    return ClauseExtraction(
        category=category,
        found=True,
        summary=f"{category} summary (refined)",
        key_terms=[f"{category} term (clarified)"],
        confidence="high",
        needs_deeper_pass=False,
    )


_call_count = {"assess": 0}


def fake_assess_risk(playbook, extractions):
    _call_count["assess"] += 1
    categories = [e.category for e in extractions]
    still_low = [e.category for e in extractions if e.confidence == "low"]
    if still_low and _call_count["assess"] == 1:
        return RiskAssessment(flags=[], ambiguous_categories=still_low, is_sufficient=False)
    return RiskAssessment(
        flags=[
            RiskFlag(
                category=c,
                risk_level="compliant",
                rationale="test rationale",
                playbook_reference="test playbook ref",
            )
            for c in categories
        ],
        ambiguous_categories=[],
        is_sufficient=True,
    )


def fake_write_report(extractions, assessment):
    return FinalReport(
        contract_summary="A test contract.",
        categories_reviewed=[e.category for e in extractions],
        risk_flags=assessment.flags,
        overall_risk_level="low",
        recommended_actions=["Test action"],
    )


def main():
    with (
        patch("graph.extract_clause", side_effect=fake_extract_clause),
        patch("graph.extract_clause_deeper", side_effect=fake_extract_clause_deeper),
        patch("graph.assess_risk", side_effect=fake_assess_risk),
        patch("graph.write_report", side_effect=fake_write_report),
    ):
        from graph import build_graph

        graph = build_graph()

        initial_state = {
            "contract_text": "This is a placeholder contract body.",
            "categories": ["liability", "auto_renewal", "data_handling", "termination"],
            "extractions": [],
            "refinement_round": 0,
            "max_refinement_rounds": 1,
        }

        final_state = graph.invoke(initial_state)

        assert _call_count["assess"] == 2, (
            f"expected risk_flagger to run twice (initial + post-refinement), "
            f"got {_call_count['assess']}"
        )

        report = final_state["final_report"]
        assert set(report.categories_reviewed) == {
            "liability", "auto_renewal", "data_handling", "termination"
        }
        assert len(report.risk_flags) == 4

        deduped = final_state["extractions"]
        by_category = {e.category: e for e in deduped}
        assert by_category["data_handling"].confidence == "high"
        assert "refined" in by_category["data_handling"].summary

        print("ALL ASSERTIONS PASSED")
        print(f"- Risk flagger ran {_call_count['assess']} times (fan-out -> ambiguity -> refine -> resolve)")
        print(f"- Categories reviewed: {report.categories_reviewed}")
        print(f"- data_handling after refinement: confidence={by_category['data_handling'].confidence}")


if __name__ == "__main__":
    main()
