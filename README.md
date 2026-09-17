# KG-RAG: Knowledge Graph-Augmented Retrieval-Augmented Generation for ISRO Domain Question Answering

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Backend](https://img.shields.io/badge/Backend-Complete-brightgreen)
![Frontend](https://img.shields.io/badge/Frontend-Pending-orange)

> A hybrid RAG system that automatically constructs a domain-specific knowledge graph
> from unstructured ISRO mission documents and integrates it with FAISS dense retrieval
> for accurate, hallucination-reduced question answering — running entirely on local
> consumer-grade hardware with zero cloud dependency.

The complete Python backend is implemented and has produced evaluated results, including
scraping, preprocessing, knowledge graph construction, dense indexing, hybrid retrieval,
generation, evaluation, ablation studies, and unit tests. The React frontend and analysis
notebooks are the remaining implementation tasks.

---

## Research Context

This project is developed as part of the **ISRO Bharatiya Antariksh Hackathon 2025 (BAH-02)**
and serves as the Capstone Project for the M.Tech in Artificial Intelligence and Data Science
programme at Alliance School of Advanced Computing, Alliance University (2025–2027).

**Target Publication:** ICNLP 2027 (IEEE Xplore / Scopus)

---

## System Architecture

```
                    ┌─────────────────────┐
                    │   ISRO Public Docs  │
                    │   (isro.gov.in)     │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Firecrawl API     │
                    │  Web + PDF scrape   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    Text Chunks      │
                    │ 512 tok, stride 128 │
                    └────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │     spaCy NER       │      │   MiniLM-L6-v2      │
   │  + Dep. Parsing     │      │  (384-dim encoder)  │
   └──────────┬──────────┘      └──────────┬──────────┘
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │    NetworkX KG      │      │    FAISS Index      │
   │ 4.2K nodes/11.5K   │      │     Flat L2         │
   │      edges          │      │                     │
   └──────────┬──────────┘      └──────────┬──────────┘
              │                             │
              │      ┌──────────────┐       │
              │      │  User Query  │       │
              │      └──────┬───────┘       │
              │             │               │
              │      ┌──────┴───────┐       │
              │      │  Query NER   │       │
              │      └──────┬───────┘       │
              │             │               │
              ▼             ▼               ▼
   ┌─────────────────────────────────────────────┐
   │            Hybrid Retrieval                 │
   │   1-hop KG neighbours + top-5 FAISS chunks  │
   └─────────────────────┬───────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │  Mistral-7B-Instruct    │
           │     Q4_K_M via Ollama   │
           │   (local, zero-cost)    │
           └─────────────┬───────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │     Generated Answer    │
           └─────────────────────────┘
```

---

## Key Features

- Automatic KG construction from raw unstructured ISRO documents — no pre-built structured KB required
- Hybrid retrieval combining structured KG one-hop neighbourhood expansion with dense FAISS passage retrieval
- Fully local inference using Mistral-7B-Instruct Q4\_K\_M via Ollama — zero cloud dependency
- ISRO-QA: a curated benchmark of 200 domain-specific question-answer pairs across three difficulty tiers
- Evaluated using RAGAS metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Deployable React frontend for interactive question answering

---

## Results (Preliminary)

| System | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| BM25 + LLM | 0.62 | 0.59 | — | — |
| Vanilla RAG | 0.71 | 0.68 | — | — |
| GraphRAG | — | — | — | — |
| **KG-RAG (ours)** | **0.84** | **0.81** | — | — |

The evaluation pipeline reports ROUGE-L, answer coverage, exact match, and abstention
rate from the generated result files.

---

## Project Structure

```
KG-RAG-ISRO/
│
├── data/
│   ├── raw/                    # Scraped markdown files from isro.gov.in
│   ├── cleaned/                # Cleaned documents (output of clean.py)
│   ├── chunks/                 # Preprocessed chunks — chunks.json
│   ├── index/                  # FAISS index files
│   ├── kg/                     # NetworkX graph (pickle + JSON)
│   ├── results/                # Evaluation outputs (ragas_scores.json, etc.)
│   └── benchmark/              # ISRO-QA benchmark — isro_qa.json
│
├── src/
│   ├── scraper/
│   │   └── crawl.py            # Firecrawl + HTTP-fallback data collection
│   │
│   ├── preprocessing/
│   │   ├── clean.py            # Noise removal, deduplication
│   │   └── chunk.py            # 512-token chunker with 128-token stride
│   │
│   ├── kg_builder/
│   │   ├── ner.py              # spaCy NER entity extraction
│   │   ├── entity_ruler.py     # ISRO-domain EntityRuler patterns
│   │   ├── relations.py        # Stub — triple logic lives in build_kg.py
│   │   └── build_kg.py         # NetworkX graph construction + dep. parsing
│   │
│   ├── indexer/
│   │   ├── encode.py           # MiniLM-L6-v2 chunk encoding
│   │   └── build_index.py      # FAISS index builder
│   │
│   ├── retriever/
│   │   ├── kg_retriever.py     # One-hop KG neighbourhood expansion
│   │   ├── faiss_retriever.py  # Dense passage retrieval
│   │   ├── hybrid.py           # Context merging pipeline
│   │   └── query.py            # CLI query entrypoint
│   │
│   ├── generator/
│   │   ├── prompt.py           # Prompt templates
│   │   └── ollama_api.py       # Ollama REST API integration
│   │
│   ├── baselines/
│   │   ├── bm25_llm.py         # BM25 + Mistral baseline
│   │   ├── vanilla_rag.py      # FAISS-only RAG baseline
│   │   ├── graphrag.py         # GraphRAG baseline
│   │   └── run_baselines.py    # Batch baseline runner
│   │
│   └── evaluation/
│       ├── evaluate.py         # ROUGE-L / coverage / exact-match scorer
│       ├── ablation.py         # Ablation study scripts
│       └── plot_results.py     # Result visualisation (matplotlib)
│
├── frontend/                   # ⚠ Placeholder — React UI not yet implemented
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx     # Placeholder
│   │   │   ├── QueryInput.jsx  # Placeholder
│   │   │   └── Answer.jsx      # Placeholder
│   │   ├── App.jsx             # Placeholder
│   │   └── index.jsx
│   └── package.json
│
├── notebooks/                  # ⚠ Placeholder — content not yet written
│   ├── kg_analysis.ipynb
│   ├── retrieval_analysis.ipynb
│   └── results_analysis.ipynb
│
├── paper/
│   ├── main.tex                # IEEE paper LaTeX source
│   ├── references.bib          # Bibliography (WIP)
│   └── figures/
│       ├── results_comparison.png
│       ├── ablation_results.png
│       └── idk_per_tier.png
│
├── tests/
│   ├── test_graph_directory.py
│   ├── test_indexer.py
│   ├── test_kg_builder.py
│   └── test_preprocessing.py
│
├── app.py                      # Minimal Flask/CLI app wrapper
├── app_v2.py                   # Extended app with streaming support
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM
- [Ollama](https://ollama.ai) installed and running
- Node.js 18+ (for frontend)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/KG-RAG-ISRO.git
cd KG-RAG-ISRO
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# Windows PowerShell
venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 4. Pull Mistral model via Ollama

```bash
ollama pull mistral:7b-instruct-q4_K_M
```

### 5. Set up environment variables (optional for the HTTP fallback)

```bash
cp .env.example .env
# Add FIRECRAWL_API_KEY to .env to use Firecrawl scraping.
# Without a key, the scraper uses its local HTTP fallback.
```

### 6. Install frontend dependencies (once the React UI is implemented)

```bash
cd frontend
npm install
npm run dev
```

> **Note:** The frontend is not yet implemented. The `npm start` script is a placeholder.

---

## Usage

### Step 1 — Collect documents

```bash
python src/scraper/crawl.py
```

The crawler writes Markdown files to `data/raw/`. Use `python src/scraper/crawl.py --help`
to inspect crawl limits and seed options.

### Step 2 — Clean and chunk

```bash
python src/preprocessing/clean.py --input-dir data/raw --output-dir data/cleaned
python src/preprocessing/chunk.py --input-dir data/cleaned --output-dir data/chunks
```

The commands write cleaned documents to `data/cleaned/` and chunk JSON files to
`data/chunks/`.

### Step 3 — Build knowledge graph

```bash
python src/kg_builder/build_kg.py
```

### Step 4 — Build FAISS index

The retriever expects `data/index/faiss_index.index` and the matching
`data/chunks/chunks.json`. The indexer utilities encode the project chunks and write the
FAISS index. They can be invoked from Python as follows:

```python
from src.indexer.encode import encode_chunks
from src.indexer.build_index import build_faiss_index

vectors = encode_chunks("data/chunks")
build_faiss_index(vectors, "data/index")
```

### Step 5 — Run the QA system

```bash
python src/retriever/query.py --question "What is the primary payload of Chandrayaan-2?"
```

The query command expects the generated KG and FAISS index and a running local Ollama
model. It prints the generated answer to the terminal.

### Step 6 — Run all baselines

```bash
python src/baselines/run_baselines.py
```

Runs BM25 + LLM, Vanilla RAG, and GraphRAG baselines and writes results to
`data/results/baseline_results.json`.

### Step 7 — Evaluate saved results

```bash
python src/evaluation/evaluate.py
```

Reads `data/results/baseline_results.json` and the benchmark, then writes scores to
`data/results/ragas_scores.json`. Reports ROUGE-L, answer coverage, exact match, and
abstention rate.

### Step 8 — Run ablation study

```bash
python src/evaluation/ablation.py
```

### Step 9 — Plot results

```bash
python src/evaluation/plot_results.py
```

Generates comparison charts to `paper/figures/`.

### Step 10 — Run tests

```bash
pytest -q
```

---

## ISRO-QA Benchmark

ISRO-QA is a curated benchmark of 200 domain-specific question-answer pairs across three
difficulty tiers, stored at `data/benchmark/isro_qa.json`:

| Tier | Type | Count |
|---|---|---|
| 1 | Factoid | 100 |
| 2 | Multi-hop relational | 60 |
| 3 | Timeline reasoning | 40 |
| **Total** | | **200** |

---

## Tech Stack

| Component | Tool |
|---|---|
| Web scraping | Firecrawl API |
| NER + parsing | spaCy `en_core_web_lg` |
| Knowledge graph | NetworkX 3.x |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector index | FAISS-CPU |
| LLM | Mistral-7B-Instruct Q4\_K\_M |
| LLM serving | Ollama |
| Evaluation | RAGAS |
| Frontend | React (Vite) — pending |

---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| Web scraper | ✅ Complete | Firecrawl API + HTTP fallback |
| Preprocessing | ✅ Complete | Cleaning, deduplication, chunking |
| NER + entity ruler | ✅ Complete | spaCy + ISRO-domain patterns |
| KG construction | ✅ Complete | NetworkX, dep. parsing, triple extraction |
| FAISS indexer | ✅ Complete | MiniLM-L6-v2 + Flat L2 index |
| KG retriever | ✅ Complete | 1-hop neighbourhood expansion |
| Hybrid retriever | ✅ Complete | KG + FAISS context merging |
| Generator | ✅ Complete | Ollama REST + prompt templates |
| Baselines | ✅ Complete | BM25, Vanilla RAG, GraphRAG |
| Evaluation | ✅ Complete | ROUGE-L, coverage, exact match, abstention |
| Ablation study | ✅ Complete | Results in `data/results/ablation_results.json` |
| Unit tests | ✅ Complete | 4 test modules, `pytest -q` |
| ISRO-QA benchmark | ✅ Complete | 200 QA pairs, `data/benchmark/isro_qa.json` |
| Result figures | ✅ Complete | 3 charts in `paper/figures/` |
| Paper (LaTeX) | 🔄 In progress | `paper/main.tex` drafted; bibliography WIP |
| React frontend | ❌ Pending | All components are placeholder stubs |
| Analysis notebooks | ❌ Pending | `.ipynb` files are empty stubs |

---

## Team

| Name | Role | Institution |
|---|---|---|
| Nandana Narayan Das | KG pipeline, retrieval, evaluation, paper | Alliance School of Advanced Computing, Alliance University |
| Gowri Kannan | Data collection, generation, baselines, frontend | VJCET |

---

## Citation

If you use this work or the ISRO-QA benchmark, please cite:

```bibtex
@inproceedings{narayandas2027kgrag,
  title     = {KG-RAG: Knowledge Graph-Augmented Retrieval-Augmented Generation
               for ISRO Domain Question Answering on Resource-Constrained Hardware},
  author    = {Narayan Das, Nandana and Kannan, Gowri},
  booktitle = {Proceedings of the International Conference on Natural Language Processing (ICNLP)},
  year      = {2027}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

This work is conducted under the ISRO Bharatiya Antariksh Hackathon 2025 (BAH-02) framework.
We thank Alliance School of Advanced Computing, Alliance University for academic support.
