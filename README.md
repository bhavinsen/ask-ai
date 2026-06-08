# Ask Bhavin

A simple **Retrieval-Augmented Generation (RAG)** chat app that answers questions about **Bhavin Sen** using a curated knowledge base built from public professional profiles.

![Stack](https://img.shields.io/badge/RAG-TF--IDF-blue) ![UI](https://img.shields.io/badge/UI-Streamlit-red)

## Quick start

```bash
cd ask-ai
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Index knowledge (first run also happens automatically in the UI)
python -m src.ingest

# Copy .env.example → .env and add your Sarvam API key (https://dashboard.sarvam.ai)
cp .env.example .env

streamlit run app.py
```

Open **http://localhost:8501** and ask questions like:

- *What AI tools has Bhavin used in production?*
- *What's his Freelancer rating?*
- *Has he worked on HIPAA projects?*

## Architecture

```
data/*.md          →  chunk + TF-IDF index    →  .index/ on disk
User question      →  retrieve top-5 chunks   →  Sarvam sarvam-105b (optional)
                                              →  grounded answer + source citations
```

| Component | Choice | Why |
|-----------|--------|-----|
| **Knowledge** | 5 markdown files in `data/` | Transparent, editable, versionable |
| **Retrieval** | scikit-learn TF-IDF + cosine similarity | Zero model downloads; great for small corpora |
| **Chunking** | 500 chars, 80 overlap, header-aware | Keeps sections coherent |
| **Generation** | Sarvam `sarvam-105b` if key set | Indian LLM; strong instruction-following for grounded QA |
| **Fallback** | Show retrieved passages | App still demos retrieval without Sarvam |

## Data sources

See **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** for full provenance, limitations, and design decisions.

Primary public sources:

1. [GitHub @bhavinsen](https://github.com/bhavinsen)
2. [Freelancer @senbhavin](https://www.freelancer.com/u/senbhavin)

## Loom walkthrough

I cannot record Loom videos from this environment. Use **[docs/LOOM_SCRIPT.md](docs/LOOM_SCRIPT.md)** — a ~5–7 minute narrated script covering process, data choices, and accuracy safeguards.

## Project layout

```
ask-ai/
├── app.py                 # Streamlit UI
├── data/                  # Knowledge base (markdown)
├── src/
│   ├── config.py
│   ├── ingest.py          # Build Chroma index
│   └── rag.py             # Retrieve + generate
├── docs/
│   ├── DATA_SOURCES.md
│   └── LOOM_SCRIPT.md
└── requirements.txt
```

## Accuracy notes

- Answers are **grounded** in `data/` only; the system prompt forbids inventing facts.
- Negative client feedback is included for balance (see `05_client_reviews_and_reputation.md`).
- For production use, add resume/LinkedIn PDFs you approve, and re-run `python -m src.ingest`.

## License

MIT — built as a portfolio / evaluation demo.
