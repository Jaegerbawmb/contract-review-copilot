"""
Runs the graph on a synthetic sample vendor contract using the actual
Gemini API. Requires GEMINI_API_KEY in .env.

The contract text below is entirely made up for demo purposes -- it
intentionally mixes one clearly risky clause (uncapped liability), one
borderline clause (15-day auto-renewal notice, vs. the 30-day playbook
standard), and two reasonably standard clauses, so the risk levels in
the output should look meaningfully different across categories.

Usage:
    python sample_run.py
"""

from graph import review_graph
from schemas import DEFAULT_PLAYBOOK

SAMPLE_CONTRACT = """
MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into between
Acme Cloud Analytics Inc. ("Vendor") and Customer.

1. SERVICES. Vendor shall provide cloud-based data analytics services
as described in the applicable Order Form.

2. TERM AND RENEWAL. This Agreement shall commence on the Effective
Date and continue for an initial term of twelve (12) months. Thereafter,
this Agreement shall automatically renew for successive twelve (12) month
terms unless either party provides written notice of non-renewal at
least fifteen (15) days prior to the end of the then-current term.

3. TERMINATION. Either party may terminate this Agreement for material
breach if such breach is not cured within thirty (30) days of written
notice. Customer may not terminate this Agreement for convenience during
the initial or any renewal term.

4. DATA. Vendor may process Customer Data solely to provide the Services.
Vendor may also use aggregated, de-identified Customer Data to improve
Vendor's products and services, including for training Vendor's internal
machine learning models. Upon termination, Vendor will delete Customer
Data within ninety (90) days upon Customer's written request.

5. LIMITATION OF LIABILITY. IN NO EVENT SHALL VENDOR'S TOTAL LIABILITY
ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE FEES PAID BY
CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM, EXCEPT FOR
CLAIMS ARISING FROM VENDOR'S GROSS NEGLIGENCE, WILLFUL MISCONDUCT, OR
BREACH OF CONFIDENTIALITY OBLIGATIONS, FOR WHICH VENDOR'S LIABILITY
SHALL BE UNCAPPED.

6. GENERAL. This Agreement is governed by the laws of the State of
Delaware.
""".strip()


def main():
    initial_state = {
        "contract_text": SAMPLE_CONTRACT,
        "playbook": DEFAULT_PLAYBOOK,
        "categories": ["liability", "auto_renewal", "data_handling", "termination"],
        "extractions": [],
        "refinement_round": 0,
        "max_refinement_rounds": 1,
    }

    result = review_graph.invoke(initial_state)
    report = result["final_report"]

    print("\n=== CONTRACT SUMMARY ===")
    print(report.contract_summary)

    print("\n=== CATEGORIES REVIEWED ===")
    print(", ".join(report.categories_reviewed))

    print("\n=== RISK FLAGS ===")
    for f in report.risk_flags:
        print(f"\n[{f.risk_level.upper()}] {f.category}")
        print(f.rationale)
        print(f"  playbook ref: {f.playbook_reference}")

    print(f"\n=== OVERALL RISK LEVEL: {report.overall_risk_level.upper()} ===")

    print("\n=== RECOMMENDED ACTIONS ===")
    for a in report.recommended_actions:
        print(f"- {a}")

    print(f"\n(refinement rounds used: {result.get('refinement_round', 0)})")


if __name__ == "__main__":
    main()
