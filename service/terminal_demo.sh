#!/bin/bash

# 🚀 EXA DEMO: Complete Terminal Output (No Separate Files)
# Shows everything working together with full results in terminal

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "\n${BLUE}🚀 EXA DEMO: Agentic Search Evals with Rust/C++ Late-Interaction Reranker${NC}\n"

# Check if Rust service is running
if ! curl -s http://localhost:8088/bench > /dev/null 2>&1; then
    echo -e "${CYAN}Starting Rust service...${NC}"
    cd ranker-rs && ./target/release/ranker-rs &
    sleep 3
    cd ..
fi

echo -e "${GREEN}✅ Rust service running${NC}"

# Show microbench
echo -e "\n${CYAN}🔬 MICROBENCH RESULTS:${NC}"
curl -s "http://localhost:8088/bench?n_docs=1000&td=64&d=128&prune=16/64" | jq '.'

# Run full evaluation with terminal output
echo -e "\n${CYAN}🔍 RUNNING FULL EVALUATION:${NC}"
echo -e "Query: 'Challenges in evaluating LLM-powered search quality'"
echo -e "Providers: DuckDuckGo + Wikipedia"
echo -e "Configuration: Late-interaction + 16/64 pruning"
echo ""

source .venv/bin/activate

# Run evaluation and show all output in terminal
python3 orchestrator/run.py \
    --q "Challenges in evaluating LLM-powered search quality" \
    --providers ddg,wikipedia \
    --late on \
    --prune 16/64 \
    --judge heuristic \
    --seed 1337

echo -e "\n${GREEN}✅ Demo completed! All results shown in terminal above.${NC}\n"