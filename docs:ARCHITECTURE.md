# Logos — Christian AI Assistant
## Architecture & Engineering Decisions

---

## Overview

Logos is a Christianity-focused AI assistant built around four core engineering concerns:
**hallucination prevention**, **scripture grounding**, **safety moderation**, and **denomination-aware context**.
It is intentionally NOT a generic chatbot with a cross icon — every architectural decision traces back to a specific failure mode in faith-domain AI.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        USER INPUT                        │
└─────────────────────────┬────────────────────────────────┘
                          │
                ┌─────────▼──────────┐
                │   PRE-FLIGHT LAYER  │
                │  ① Moderation Check │  ← Hard-block regex patterns
                │  ② Halluc Detection │  ← Known fake-verse DB
                │  ③ Intent Classify  │  ← Chat vs Image mode
                └─────────┬──────────┘
                          │ Pass
          ┌───────────────▼───────────────┐
          │       CONTEXT ASSEMBLY         │
          │  • Denomination system prompt  │
          │  • Conversation history (k=20) │
          │  • Safety instructions embed   │
          └───────────────┬───────────────┘
                          │
              ┌───────────▼───────────┐
              │      CLAUDE API        │
              │  claude-sonnet-4       │
              │  (grounded system      │
              │   prompt + history)    │
              └───────────┬───────────┘
                          │
            ┌─────────────▼─────────────┐
            │    POST-RESPONSE LAYER     │
            │  ① Scripture citation fmt  │
            │  ② Halluc scan on output   │
            │  ③ Denom badge display     │
            └─────────────┬─────────────┘
                          │
                ┌─────────▼────────┐
                │   RENDERED UI    │
                └──────────────────┘
```

---

## Key Engineering Decisions

### 1. Hallucination Prevention Strategy

**Problem:** LLMs frequently invent Bible verse references or misquote scripture.

**Solution (multi-layer):**

- **Pre-flight fake-verse DB:** A curated list of the most commonly misattributed phrases ("God helps those who help themselves," "This too shall pass," etc.) is checked *before* the API call. If detected in the user's input, a warning is appended to the response — the user is corrected even if the LLM doesn't catch it.

- **System prompt constraint:** The system prompt explicitly instructs: *"NEVER invent, paraphrase-as-quote, or slightly alter Bible verses. If unsure, use 'Paraphrasing: [Book Chapter:Verse]' rather than quoting."* This shifts the model toward conservative behavior.

- **Post-response scan:** The same fake-verse detector runs on the model's output. If a misattribution slips through, it's flagged in the UI.

- **What this does NOT solve:** Subtle hallucination of obscure verses (e.g., fabricating a plausible-sounding Proverbs verse). Production-grade solution would add a RAG layer with a verified Bible corpus, checking every cited verse against a ground truth database.

**Production extension:** Vector DB (ChromaDB/Pinecone) with all 31,102 Bible verses embedded. Every verse reference in the response is looked up via semantic + exact match. If no match found within cosine similarity threshold, it's flagged as potentially hallucinated.

---

### 2. Denomination-Aware Context System

**Problem:** "What is Communion?" has fundamentally different correct answers depending on whether someone is Catholic (transubstantiation), Lutheran (consubstantiation), Baptist (memorial only), or Orthodox (mystical presence).

**Solution:** A denomination selector drives the system prompt. Each denomination has a tailored context block that tells the model:
- Which theological sources to prioritize (Catechism vs. Westminster Confession vs. Book of Common Prayer)
- Which theologians to reference (Aquinas, Luther, Calvin, Wesley, Palamas)
- Which sacramental theology to apply
- Language appropriate to the tradition

This is not merely stylistic — it changes the *content* of theologically accurate responses.

**10 denominations supported:** General Christian, Catholic, Protestant/Evangelical, Eastern Orthodox, Anglican, Pentecostal, Lutheran, Reformed/Presbyterian, Methodist, Baptist.

---

### 3. Safety Architecture (Layered Moderation)

Rather than a single content filter, Logos uses three tiers:

**Tier 1 — Hard Block (pre-API, regex-based):**
Patterns that should NEVER reach the model:
- Scripture corruption requests ("rewrite verse to support X hatred")
- Satanic/blasphemous content generation
- Sexual content involving religious figures
- Incitement to violence using scriptural framing

*Rationale: Sending these to the API wastes tokens and risks partial compliance. Block before the call.*

**Tier 2 — Soft Warning (pre-API, redirect):**
Theologically sensitive but legitimate questions that need flagging:
- Prosperity gospel questions (present multiple scholarly views)
- Conspiracy theology (flat earth, vaccine = Mark of the Beast)
- Inter-religious comparisons

*Rationale: Don't block legitimate theological inquiry. Contextualize it.*

**Tier 3 — System Prompt Safety:**
The model's own guardrails via system prompt:
- Refuse to validate heretical teachings as mainstream
- Handle theodicy and controversial topics with pastoral care
- Not pick winners in genuinely contested denominational debates

**Image Safety:** A separate image moderation layer checks image prompts for violence, sexual content, and disrespectful portrayals before generating a DALL-E 3 prompt.

---

### 4. Conversation Memory

**Current implementation:** In-memory sliding window (last 20 turns). Passed as full `messages` array on each API call, providing conversational continuity.

**Production extension:**
- Persistent storage: user sessions saved to a vector DB (conversation summaries embedded)
- Long-term memory: summarize older context and inject as a system-level "memory" block
- Cross-session topics: "You previously asked about the Sermon on the Mount — would you like to continue that thread?"

---

### 5. Image Generation Pipeline

**Architecture:**
```
User prompt → Safety check → Claude prompt refinement → Image API (DALL-E 3) → Theological caption
```

**Why two-step?** Raw user prompts often need theological/artistic refinement. "Show me God" is theologically fraught (divine invisibility) — the refinement step converts it into an appropriate artistic representation (light, doves, a hand from a cloud in classical style).

**Grounding:** The caption for each image is generated by Claude using the same scripture-grounded system prompt, ensuring the spiritual reflection is theologically accurate.

---

## What Would Complete the Production System

| Feature | Current State | Production Approach |
|---|---|---|
| Bible verse grounding | System prompt constraints | RAG over full Bible corpus (ChromaDB + OpenAI embeddings) |
| Image generation | Refined prompt + placeholder | DALL-E 3 API with Christian art style guide |
| Conversation persistence | In-memory | PostgreSQL + session tokens |
| Hallucination detection | Curated fake-verse list | Semantic search against verified verse DB |
| User accounts | None | Auth (Clerk/Supabase) + personal history |
| Moderation | Client-side regex | OpenAI Moderation API + custom fine-tuned classifier |
| Multilingual | English only | i18n + multilingual scripture databases |

---

## Evaluation Dataset

`eval/evaluation_dataset.json` contains 42 test cases across 5 categories:
- **Hallucination tests** (H001–H008): Fake verses, nonexistent books, historical misattributions
- **Adversarial prompts** (A001–A008): Scripture corruption, jailbreaks, incitement attempts
- **Theological edge cases** (T001–T007): Theodicy, LGBTQ inclusion, faith/science tension
- **Denomination-aware tests** (D001–D003): Same question, different correct answers by tradition
- **Content generation tests** (C001–C005): Devotionals, images, sermons, creative content

Scoring rubric weights: Scripture accuracy (25%), Hallucination prevention (25%), Safety (20%), Theological accuracy (15%), Pastoral quality (15%).

---

## Technical Stack

```
Frontend:    Vanilla HTML/CSS/JS (no framework overhead)
Fonts:       Cinzel (headers) + EB Garamond (body) — period-appropriate serif
API:         Anthropic Claude claude-sonnet-4 (claude-sonnet-4-20250514)
State:       In-memory conversation array
Moderation:  Client-side regex (Tier 1) + LLM system prompt (Tier 2+3)
Deployment:  Any static host (Vercel, Netlify, GitHub Pages)
```

---

## Tradeoffs & Honest Limitations

1. **No real RAG yet:** The most robust scripture grounding requires a vector DB. The current approach relies on LLM knowledge + system prompt constraints — effective but not verifiable.

2. **Client-side safety has limits:** Regex patterns are bypassable by creative rephrasing. Production systems should use the Anthropic Moderation API or a fine-tuned classifier.

3. **Denomination context is a heuristic:** The denomination selector improves responses but doesn't guarantee theological accuracy for every tradition — Orthodox theology in particular requires specialized knowledge.

4. **Image generation is mocked:** DALL-E 3 / Stability AI API integration is architected but not live-connected in this demo (requires separate API key management).

---

*Built for the AI Engineer role evaluation — demonstrating prompt engineering, grounding strategies, multimodal architecture, safety thinking, and product judgment.*
