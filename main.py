"""
Logos Christian AI Assistant — FastAPI Backend
Demonstrates: FastAPI, LangChain, RAG, ChromaDB, Anthropic API, safety moderation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import re
import json

from rag_pipeline import ChristianRAGPipeline

app = FastAPI(
    title="Logos Christian AI Assistant",
    description="Scripture-grounded, denomination-aware Christian AI backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline once at startup
rag = ChristianRAGPipeline()

# ── REQUEST / RESPONSE MODELS ──

class ChatRequest(BaseModel):
    message: str
    denomination: str = "general"
    conversation_history: List[dict] = []
    image_mode: bool = False

class ChatResponse(BaseModel):
    reply: str
    scripture_references: List[str] = []
    hallucination_warning: Optional[str] = None
    denomination_context: str = "general"
    safety_flagged: bool = False
    rag_sources: List[str] = []

class VerifyVerseRequest(BaseModel):
    reference: str  # e.g. "John 3:16"
    claimed_text: Optional[str] = None

class ModerationResult(BaseModel):
    blocked: bool
    reason: Optional[str] = None
    soft_warning: Optional[str] = None

# ── SAFETY / MODERATION ──

HARD_BLOCK_PATTERNS = [
    r"rewrite.*bible.*verse.*to support",
    r"bible.*verse.*justify.*(?:hate|violence|racism|slavery as good)",
    r"satanic.*prayer|prayer to satan",
    r"(?:kill|murder|harm).*(?:christian|church|pastor|priest)",
    r"jesus.*(?:sex|porn|nude|naked)",
    r"heresy generator",
    r"antichrist.*manifesto",
]

SOFT_WARN_PATTERNS = [
    (r"prosperity gospel", "Prosperity gospel is theologically contested. Multiple scholarly perspectives will be presented."),
    (r"flat earth.*bible", "Flat earth biblical claims are not supported by mainstream biblical scholarship."),
    (r"covid.*mark of the beast", "This interpretation is not supported by mainstream biblical scholars."),
]

FAKE_VERSES = [
    ("god helps those who help themselves", "NOT in the Bible. Often misattributed; originates from Algernon Sidney (1698). See Philippians 4:13, Proverbs 3:5-6."),
    ("cleanliness is next to godliness", "NOT a Bible verse. From John Wesley's 1778 sermon, not scripture."),
    ("this too shall pass", "NOT a Bible verse. Persian proverb. See 2 Corinthians 4:17 for similar comfort."),
    ("money is the root of all evil", "Misquotation. 1 Timothy 6:10 says the LOVE of money is 'a root of all kinds of evil' — an important distinction."),
    ("spare the rod spoil the child", "Not a direct quote. The actual verse is Proverbs 13:24."),
]

def check_moderation(text: str) -> ModerationResult:
    text_lower = text.lower()
    for pattern in HARD_BLOCK_PATTERNS:
        if re.search(pattern, text_lower):
            return ModerationResult(blocked=True, reason="Request violates content safety guidelines.")
    for pattern, warning in SOFT_WARN_PATTERNS:
        if re.search(pattern, text_lower):
            return ModerationResult(blocked=False, soft_warning=warning)
    return ModerationResult(blocked=False)

def detect_hallucination(text: str) -> Optional[str]:
    text_lower = text.lower()
    for phrase, note in FAKE_VERSES:
        if phrase in text_lower:
            return note
    # Detect references to nonexistent books
    fake_books = ["hezekiah", "jasher", "enoch 2", "3 maccabees"]
    for book in fake_books:
        if book in text_lower:
            return f"Note: '{book.title()}' is not part of the standard biblical canon. Please verify the reference."
    return None

# ── DENOMINATION SYSTEM PROMPTS ──

DENOMINATION_CONTEXTS = {
    "catholic": "The user is Catholic. Reference the Catechism (CCC), Magisterium, Sacred Tradition, seven sacraments, role of Mary and saints, and papal authority.",
    "protestant": "The user is Protestant/Evangelical. Emphasize sola scriptura, justification by faith alone, and cite Protestant theologians (Luther, Calvin, Wesley).",
    "orthodox": "The user is Eastern Orthodox. Reference Holy Tradition, Church Fathers, theosis, the Divine Liturgy, and Orthodox councils.",
    "anglican": "The user is Anglican. Reference the Book of Common Prayer, the via media, and Anglican theologians.",
    "pentecostal": "The user is Pentecostal/Charismatic. Emphasize gifts of the Holy Spirit, personal experience, and the ongoing work of the Spirit.",
    "lutheran": "The user is Lutheran. Emphasize Law/Gospel distinction, justification by grace through faith, and the Lutheran confessions.",
    "reformed": "The user is Reformed/Presbyterian. Reference TULIP, covenant theology, and the Westminster Confession.",
    "methodist": "The user is Methodist. Reference Wesley's quadrilateral (Scripture, Tradition, Reason, Experience) and Wesleyan sanctification.",
    "baptist": "The user is Baptist. Emphasize believers' baptism, local church autonomy, and the priesthood of all believers.",
    "general": "The user has selected General Christian. Present mainstream Christian teaching while noting significant denominational differences.",
}

def build_system_prompt(denomination: str) -> str:
    denom_context = DENOMINATION_CONTEXTS.get(denomination, DENOMINATION_CONTEXTS["general"])
    return f"""You are Logos, a deeply knowledgeable Christian AI assistant. Your purpose is to help users explore the Christian faith with accuracy, reverence, and pastoral wisdom.

DENOMINATION CONTEXT:
{denom_context}

CORE RULES:
1. SCRIPTURE ACCURACY: Only cite verses you are confident exist. If unsure, write "Paraphrasing: [Ref]" rather than quoting. Never invent references.
2. HALLUCINATION PREVENTION: If a user cites a verse that doesn't exist (e.g. "God helps those who help themselves"), correct them gently.
3. SAFETY: Never rewrite scripture to support harmful ideologies. Handle difficult questions pastorally.
4. TONE: Warm, intellectually honest, accessible to seekers but deep for scholars.
5. FORMAT: Cite verses as "Book Chapter:Verse (Translation)". Use ✝ for section breaks in longer responses.

You have access to a RAG system with verified Bible verses. Always prefer RAG-grounded responses over memory for scripture quotes."""


# ── API ROUTES ──

@app.get("/")
def root():
    return {"message": "Logos Christian AI Backend is running ✝", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG grounding, moderation, and hallucination detection."""

    # Step 1: Moderation check
    mod = check_moderation(request.message)
    if mod.blocked:
        return ChatResponse(
            reply="I'm unable to process that request. Logos is designed to support and strengthen faith, not undermine it. I'd be happy to help with a different question.",
            safety_flagged=True,
            denomination_context=request.denomination
        )

    # Step 2: Pre-flight hallucination detection
    halluc_warning = detect_hallucination(request.message)

    # Step 3: RAG retrieval — find relevant Bible verses
    rag_results = rag.retrieve(request.message, k=3)
    rag_context = "\n".join([f"- {r['reference']}: \"{r['text']}\"" for r in rag_results])
    rag_sources = [r['reference'] for r in rag_results]

    # Step 4: Build messages with RAG context injected
    system_prompt = build_system_prompt(request.denomination)
    if rag_context:
        system_prompt += f"\n\nRELEVANT SCRIPTURE (from verified Bible corpus):\n{rag_context}\nPrefer citing these verified verses in your response."

    messages = request.conversation_history.copy()
    messages.append({"role": "user", "content": request.message})

    # Step 5: Call Claude API
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=messages
        )
        reply = response.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")

    # Step 6: Post-response hallucination scan
    post_halluc = detect_hallucination(reply)
    final_halluc_warning = halluc_warning or post_halluc

    # Step 7: Extract scripture references from reply
    refs = re.findall(r'\b\d?\s?[A-Z][a-z]+\.?\s\d+:\d+(?:-\d+)?(?:\s\([A-Z]+\))?', reply)

    return ChatResponse(
        reply=reply,
        scripture_references=list(set(refs)),
        hallucination_warning=final_halluc_warning,
        denomination_context=request.denomination,
        safety_flagged=False,
        rag_sources=rag_sources,
        **{"soft_warning": mod.soft_warning} if mod.soft_warning else {}
    )


@app.post("/verify-verse")
async def verify_verse(request: VerifyVerseRequest):
    """Check if a Bible verse reference exists and return the actual text."""
    result = rag.verify_verse(request.reference)
    return {
        "reference": request.reference,
        "exists": result["exists"],
        "actual_text": result.get("text"),
        "note": result.get("note"),
        "claimed_text_matches": None  # In production: compare with embeddings
    }


@app.post("/moderate")
async def moderate(text: str):
    """Standalone moderation endpoint."""
    return check_moderation(text)


@app.get("/denominations")
async def list_denominations():
    """List supported denominations."""
    return {"denominations": list(DENOMINATION_CONTEXTS.keys())}


@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": rag.is_ready()}
