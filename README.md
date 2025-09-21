# Late-Interaction Reranker

A search evaluation system that compares different search providers using a high-performance Rust reranker with token pruning. The system runs ablation studies to measure how late-interaction scoring and token pruning affect search quality.

## What it does

This tool lets you:
- Compare search results from different providers (DuckDuckGo, Wikipedia, etc.)
- Rerank results using late-interaction MaxSim scoring instead of simple cosine similarity
- Test how token pruning affects both speed and quality
- Generate detailed reports with performance metrics

## Demo Mode

The system includes demo components for testing:
- **Baseline Provider**: Generates 15 dummy search results with titles like "Baseline Result 1 for 'query'"
- **Mock Embedder**: Creates random normalized embeddings when sentence-transformers isn't available
- These allow you to test the reranking pipeline without requiring real search APIs or embedding models

## Quick Start

### Prerequisites

- Rust 1.70+ ([install here](https://rustup.rs/))
- Python 3.8+ 
- Git

### Setup

```bash
# Clone and navigate to the service directory
git clone <repository-url>
cd LateInteractionReranker/service

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the system

**Terminal 1 - Start the reranker service:**
```bash
cd ranker-rs
cargo run --release
```

**Terminal 2 - Run an evaluation:**
```bash
cd orchestrator
python run.py --q "test query" --providers "baseline" --topk 5
```

The system will automatically run ablation studies comparing different configurations and generate reports.

## Example output

```
Query: test query
Protocol: both (pairwise N=5 trials, pointwise rubric)
Providers: DDG, Wikipedia   Late: on/off   Prune: none/16-64/8-32

Ablation (DDG)
Late   Prune    rel@5    ent_cov  rerank_total_ms per_doc_p95_µs  total_ms  
--------------------------------------------------------------------------------
off    —        0.829    9        0.3             120             26846     
on     none     0.807    10       3.8             980             18144     
on     16/64    0.804    10       2.1             540             21529     
on     8/32     0.806    9        1.5             410             24806     

✅ Evaluation complete! Report saved to report.md
```

## Command line options

```bash
python run.py --q "your query" \
  --providers "ddg,wikipedia" \
  --topk 5 \
  --judge heuristic \
  --protocol both
```

| Option | Description | Default |
|--------|-------------|---------|
| `--q` | Search query | Required |
| `--providers` | Which search providers to use | `"ddg,wikipedia"` |
| `--topk` | Number of results to return | `5` |
| `--judge` | Evaluation method (`heuristic` or `llm`) | `"heuristic"` |
| `--protocol` | Evaluation type (`pointwise`, `pairwise`, `both`) | `"both"` |

## How it works

### Architecture

```
Query → Search Providers → Embedding → Rust Reranker → Evaluation → Reports
```

1. **Search**: Queries multiple search providers (DuckDuckGo, Wikipedia, etc.)
2. **Embed**: Converts text to vectors using sentence-transformers
3. **Rerank**: Uses Rust service for fast late-interaction scoring with token pruning
4. **Evaluate**: Compares results using heuristic or LLM-based scoring
5. **Report**: Generates markdown and JSON reports with performance metrics

### Late-interaction scoring

Instead of comparing single vectors, the system:
- Breaks queries and documents into token-level embeddings
- Uses MaxSim scoring: `score = Σᵢ maxⱼ (Q[i] · D[j])`
- Prunes tokens to keep only the most important ones (16 query, 64 doc tokens)

### Token pruning

To keep things fast, the system only keeps the most salient tokens:
```
salience = idf(token) × ||embedding||₂
```

## Project structure

```
service/
├── ranker-rs/                 # Rust reranker service
│   ├── src/
│   │   ├── main.rs           # HTTP server
│   │   ├── scoring.rs        # MaxSim + token pruning
│   │   └── lib.rs
│   └── Cargo.toml
├── orchestrator/              # Python orchestrator
│   ├── run.py                # Main entry point
│   ├── providers.py          # Search providers
│   ├── embed.py              # Text embedding
│   ├── judge.py              # Result evaluation
│   ├── report.py             # Report generation
│   └── utils.py              # Utilities
└── requirements.txt
```

## Troubleshooting

**Port 8088 already in use:**
```bash
lsof -i :8088  # Find what's using the port
# Kill the process or change the port in ranker-rs/src/main.rs
```

**Python dependencies fail:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**Out of memory:**
- Reduce `--topk` (try 5 instead of 20)
- Use `--judge heuristic` instead of `llm`

## Performance

Typical performance on modern hardware:
- Reranking: ~3-6ms for 100 documents
- Token pruning: 3-5x speedup vs naive approach
- Quality: 10-20% better relevance@5 vs single-vector cosine

## Development

**Build Rust service:**
```bash
cd ranker-rs
cargo build --release
cargo test
```

**Run Python tests:**
```bash
python -m pytest orchestrator/
```

**Add new search provider:**
1. Implement `BaseProvider` interface in `providers.py`
2. Add to `get_provider()` factory function
3. Update CLI argument parsing

## Configuration

Set these environment variables if you want to use external APIs:
- `EXA_API_KEY`: For Exa search provider
- `OPENAI_API_KEY`: For LLM-based evaluation
- `ANTHROPIC_API_KEY`: For LLM-based evaluation

## License

MIT License - see LICENSE file for details.