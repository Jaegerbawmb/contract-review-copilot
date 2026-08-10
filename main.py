from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from graph import DEFAULT_MAX_REFINEMENT_ROUNDS, DEFAULT_CATEGORIES, review_graph
from schemas import DEFAULT_PLAYBOOK, FinalReport, ClauseCategory
from llm_cache import reset_stats, get_stats

app = FastAPI(title="Vendor Contract Review Copilot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    contract_text: str
    playbook: dict[str, str] = DEFAULT_PLAYBOOK
    categories: list[ClauseCategory] = DEFAULT_CATEGORIES
    max_refinement_rounds: int = DEFAULT_MAX_REFINEMENT_ROUNDS


class ReviewResponse(BaseModel):
    report: FinalReport
    refinement_rounds_used: int
    audited: bool
    report_corrected: bool
    cache_hits: int
    cache_misses: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    if len(req.contract_text.strip()) < 200:
        raise HTTPException(400, "Contract text looks too short to review meaningfully.")

    initial_state = {
        "contract_text": req.contract_text,
        "playbook": req.playbook,
        "categories": req.categories,
        "extractions": [],
        "refinement_round": 0,
        "max_refinement_rounds": req.max_refinement_rounds,
    }

    reset_stats()
    final_state = review_graph.invoke(initial_state)
    stats = get_stats()

    return ReviewResponse(
        report=final_state["final_report"],
        refinement_rounds_used=final_state.get("refinement_round", 0),
        audited=final_state.get("audited", False),
        report_corrected=final_state.get("report_corrected", False),
        cache_hits=stats["hits"],
        cache_misses=stats["misses"],
    )