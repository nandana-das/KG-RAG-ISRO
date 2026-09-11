# KG-RAG: Knowledge Graph-Augmented Retrieval-Augmented Generation for ISRO Domain Question Answering

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Backend%20Complete-orange)

> A hybrid RAG system that automatically constructs a domain-specific knowledge graph
> from unstructured ISRO mission documents and integrates it with FAISS dense retrieval
> for accurate, hallucination-reduced question answering — running entirely on local
> consumer-grade hardware with zero cloud dependency.

The complete Python backend is implemented, including scraping, preprocessing, knowledge
graph construction, dense indexing, hybrid retrieval, generation, evaluation, and tests.
The React frontend is the remaining project component.

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
│   ├── chunks/                 # Preprocessed chunks (JSON)
│   ├── kg/                     # NetworkX graph files
│   └── benchmark/              # ISRO-QA benchmark (JSON)
│
├── src/
│   ├── scraper/
│   │   └── crawl.py            # Firecrawl data collection
│   │
│   ├── preprocessing/
│   │   ├── clean.py            # Noise removal, deduplication
│   │   └── chunk.py            # 512-token chunker with stride
│   │
│   ├── kg_builder/
│   │   ├── ner.py              # spaCy NER entity extraction
│   │   ├── relations.py        # Dependency-based triple extraction
│   │   └── build_kg.py         # NetworkX graph construction
│   │
│   ├── indexer/
│   │   ├── encode.py           # MiniLM-L6-v2 chunk encoding
│   │   └── build_index.py      # FAISS index builder
│   │
│   ├── retriever/
│   │   ├── kg_retriever.py     # One-hop KG neighbourhood expansion
│   │   ├── faiss_retriever.py  # Dense passage retrieval
│   │   ├── hybrid.py           # Context merging pipeline
│   │   └── query.py            # Main query entrypoint
│   │
│   ├── generator/
│   │   ├── prompt.py           # Prompt templates
│   │   └── ollama_api.py       # Ollama REST API integration
│   │
│   ├── baselines/
│   │   ├── bm25_llm.py         # BM25 + Mistral baseline
│   │   ├── vanilla_rag.py      # FAISS-only RAG baseline
│   │   └── graphrag.py         # GraphRAG baseline
│   │
│   └── evaluation/
│       ├── evaluate.py         # RAGAS scoring pipeline
│       └── ablation.py         # Ablation study scripts
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx     # Main chat interface
│   │   │   ├── QueryInput.jsx  # Question input component
│   │   │   └── Answer.jsx      # Answer display component
│   │   ├── App.jsx
│   │   └── index.jsx
│   └── package.json
│
├── notebooks/
│   ├── kg_analysis.ipynb       # KG statistics and visualization
│   ├── retrieval_analysis.ipynb# Retrieval quality analysis
│   └── results_analysis.ipynb  # Experiment results analysis
│
├── paper/
│   ├── main.tex                # IEEE paper LaTeX source
│   ├── references.bib          # Bibliography
│   └── figures/
│       └── system_architecture.png
│
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

### 6. Install frontend dependencies (when the UI is implemented)

```bash
cd frontend
npm install
```

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

### Step 6 — Run tests

```bash
pytest -q
```

### Step 7 — Evaluate saved results

```bash
python src/evaluation/evaluate.py
```

The evaluator reads `data/results/baseline_results.json` and the benchmark, then writes
scores to `data/results/ragas_scores.json`. It currently reports ROUGE-L, answer
coverage, exact match, and abstention rate.

### Frontend status

The backend and research pipeline are complete. The React interface in `frontend/` is the
remaining implementation task; its current `npm start` command is a placeholder.

---

## ISRO-QA Benchmark

The ISRO-QA benchmark is intended to contain 200 manually curated question-answer pairs:

| Tier | Type | Count |
|---|---|---|
| 1 | Factoid | 100 |
| 2 | Multi-hop relational | 60 |
| 3 | Timeline reasoning | 40 |
| **Total** | | **200** |

The benchmark JSON is located at `data/benchmark/isro_qa.json`.

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
| Frontend | React |

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
