# ✝ Logos — Christian AI Assistant

> A scripture-grounded, denomination-aware, safety-moderated AI assistant for the Christian faith.

![Logos Screenshot](docs/screenshot.png)

## Quick Start

```bash
# Just open index.html in a browser — no build step required
# The app uses the Anthropic API (key handled by the Claude.ai environment)
open index.html
```

---

## Features

### ✦ Core Capabilities
| Feature | Implementation |
|---|---|
| 💬 Chat interface | Conversation with memory (20-turn window) |
| 📖 Scripture grounding | Verse citations formatted + validated |
| 🔍 Hallucination detection | Pre/post-flight fake verse detection |
| 🎨 Image generation | Christian art prompt refinement + DALL-E 3 pipeline |
| 🏛 Denomination awareness | 10 traditions with tailored system prompts |
| 🛡 Safety moderation | 3-tier: hard block / soft warn / LLM-level |
| 🕊 Theological accuracy | Church history, councils, theological debates |

### ✦ Denomination Support
- General Christian
- Catholic
- Protestant / Evangelical
- Eastern Orthodox
- Anglican / Episcopal
- Pentecostal / Charismatic
- Lutheran
- Reformed / Presbyterian
- Methodist
- Baptist

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full engineering decisions.

```
User Input
    → Pre-flight moderation (regex hard-blocks)
    → Hallucination pre-check (fake verse DB)
    → Denomination context assembly
    → Claude API (scripture-grounded system prompt)
    → Post-response hallucination scan
    → Citation formatting
    → Rendered UI
```

---

## Evaluation Dataset

`eval/evaluation_dataset.json` — 42 test cases:

- **H-series**: Hallucination tests (fake verses, nonexistent books)
- **A-series**: Adversarial prompts (scripture corruption, jailbreaks)
- **T-series**: Theological edge cases (theodicy, sensitive questions)
- **D-series**: Denomination-variant questions
- **C-series**: Content generation tests

---

## Safety Layer

### Hard Blocks (never reach API)
- Rewriting scripture to support hate/violence/racism
- Generating satanic/blasphemous content
- Sexual content involving religious figures
- Incitement via scriptural framing

### Soft Redirects
- Conspiracy theology (addressed with scholarly context)
- Prosperity gospel (multiple views presented)
- Inter-religious comparisons (handled respectfully)

### LLM-Level Safety (system prompt)
- Refusal to endorse heretical teachings as mainstream
- Pastoral handling of sensitive topics
- Conservative scripture quotation policy

---

## Key Engineering Decisions

### Why conservative scripture quoting?
LLMs hallucinate Bible verses with high confidence. The system prompt mandates:
> "If unsure of exact wording, use 'Paraphrasing: [Ref]' rather than quoting."

This trades completeness for accuracy — the right tradeoff in a faith context.

### Why denomination-specific system prompts?
"Is infant baptism valid?" has a *different correct answer* depending on tradition.
A single generic prompt either picks a winner (wrong) or gives a wishy-washy non-answer (useless).
10 denomination contexts means 10 actually accurate answers.

### Why pre-flight moderation?
Sending adversarial prompts to the API costs tokens and risks partial compliance.
Regex-based hard blocks are instant, free, and deterministic for known patterns.

### Why post-response hallucination scanning?
The model sometimes ignores system prompt constraints for edge cases.
A second-pass check on the output catches escapes.

---

## File Structure

```
christian-ai/
├── index.html              ← Complete single-file app
├── eval/
│   └── evaluation_dataset.json  ← 42 test cases + rubric
└── docs/
    └── ARCHITECTURE.md     ← Full engineering decisions
```

---

## Production Roadmap

1. **RAG layer**: Embed all 31,102 Bible verses → ChromaDB → validate every citation
2. **Live image generation**: DALL-E 3 with Christian art style guide
3. **User accounts**: Session persistence, conversation history
4. **Fine-tuned moderation**: Custom classifier for theological adversarial patterns
5. **Church father corpus**: Include Patristic writings for deeper theological grounding
6. **Prayer mode**: Guided intercession and liturgical prayer generation

---

## Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS (Cinzel + EB Garamond fonts)
- **LLM**: Anthropic Claude claude-sonnet-4
- **Safety**: Client-side regex + LLM system prompt
- **State**: In-memory conversation array
- **Deploy**: Any static host

---

*Built as an AI Engineer evaluation — demonstrating prompt engineering, grounding, hallucination prevention, safety architecture, and product thinking.*
