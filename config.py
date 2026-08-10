import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Single factory for the main pipeline agents (extractor, risk_flagger,
    report_writer). Low temperature by default -- contract review should be
    consistent, not creative."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


def get_judge_llm(temperature: float = 0.0) -> ChatGroq:
    """Groq-backed judge model -- separate provider from the main Gemini
    pipeline, so an outage or quota issue on one doesn't take out the
    audit step too. Temperature 0 by default: judging consistency should
    be deterministic, not creative."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to .env to use the judge."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_retries=4,
    )