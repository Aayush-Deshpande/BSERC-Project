# Offline RAG, Local LLM & Voice Engine

A fully offline, air-gapped conversational AI stack: document ingestion and semantic
retrieval (RAG), a local Qwen3-4B reasoning engine served through Ollama, and an
on-device speech interface (whisper.cpp STT + Kokoro-82M TTS). No cloud calls, no
API keys.

## Modules

| Path | Purpose |
| --- | --- |
| `backend/knowledge/loaders/` | PDF / DOCX / PPTX / text ingestion and chunking |
| `backend/knowledge/retrieval/` | Local knowledge store, embedding index, FlashRank reranker |
| `backend/agent/llm_engine.py` | Local Qwen3-4B engine (Ollama / llama.cpp, GPU-offloaded, lazy-loaded) |
| `backend/agent/copilot.py` | RAG + LLM + voice orchestrator with safety guardrails |
| `backend/voice/stt_engine.py` | whisper.cpp speech-to-text (`Voice/ggml-tiny.en.bin`) |
| `backend/voice/tts_engine.py` | Kokoro-82M StyleTTS2 speech synthesis |
| `backend/voice/conversation.py` | Multi-turn voice session state |
| `backend/voice/thinking_stream.py` | Incremental reasoning stream for spoken replies |

## Retrieval tiers

`LocalKnowledgeStore` degrades gracefully: sentence-transformers semantic embedding
first, FlashRank cross-encoder second, keyword overlap last. Missing optional
dependencies reduce quality but never raise.

## Setup

```bash
pip install -r requirements.txt

# The LLM is served by Ollama, not pip:
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Speech models live in `Voice/`. Run the tests with `pytest`.

## Design constraints

- **Lazy loading** — the LLM is pulled into VRAM on first real use, never at import.
- **Fails soft** — connection/generation errors surface as status on the engine, never crash the caller.
- **Blocking by design** — `generate()` blocks on GPU inference; call it from a worker thread.
