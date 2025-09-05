"""Prompt templates for evaluation."""

POINTWISE_RUBRIC = """You are a strict search evaluator.

Score this ranked list for query: "{query}".

For the TOP-5 items, rate:
A) Direct Relevance (0-1)
B) Coverage of distinct aspects (0-1)  
C) Clarity/Trustworthiness (0-1)

Return JSON: {{"A":float,"B":float,"C":float,"notes": "2 lines max"}}."""

PAIRWISE_RUBRIC = """You are a comparison judge.

Two ranked lists (A,B) for query: "{query}".

Pick the winner and margin (+0.0..+1.0). Ignore formatting/SEO fluff.

Return JSON: {{"winner":"A|B","margin":float,"rationale":"1-2 lines"}}."""

ATTRIBUTION_CHECK = """You verify citation support.

Does the following passage substantively support the claim? 

Claim: "{sentence}"
Passage: "{passage}"

Answer yes/no and give one-line rationale. 

JSON: {{"ok": true|false, "why": "..."}}."""

AGENT_JUDGE = """You evaluate the effectiveness of a multi-step research agent.

Given this trace (JSON), score:
- Breadth (distinct relevant subtopics covered, 0-1),
- Redundancy (duplication across top-K; lower is better; return as 0-1 where higher=less redundancy),
- Budget (efficiency wrt total latency; return 0-1 where higher=more efficient).

Return JSON: {{"breadth":..,"redundancy":..,"budget":..,"notes":"1-2 lines"}}."""

def get_judge_prompt(query: str, provider1: str, results1: list, provider2: str, results2: list) -> str:
    """Generate judge prompt for comparing two providers."""
    results1_text = "\n".join([
        f"{i+1}. {r.title}\n   {r.snippet}\n   {r.url}\n"
        for i, r in enumerate(results1[:5])
    ])
    
    results2_text = "\n".join([
        f"{i+1}. {r.title}\n   {r.snippet}\n   {r.url}\n"
        for i, r in enumerate(results2[:5])
    ])
    
    return f"""Compare these two search result sets for query: "{query}"

{provider1.upper()} Results:
{results1_text}

{provider2.upper()} Results:
{results2_text}

Which provider gives better results? Consider relevance, coverage, and quality.
Return JSON: {{"winner": "{provider1}|{provider2}", "margin": 0.0-1.0, "rationale": "brief explanation"}}"""

def heuristic_evaluate(provider: str, results: list, query: str) -> float:
    """Heuristic evaluation of search results."""
    if not results:
        return 0.0
    
    # Simple scoring based on title length, snippet length, and URL quality
    scores = []
    for result in results[:5]:  # Top 5 results
        title_score = min(len(result.title) / 50, 1.0)
        snippet_score = min(len(result.snippet) / 200, 1.0)
        url_score = 0.5
        
        # Bonus for certain domains
        if any(domain in result.url.lower() for domain in ['wikipedia', 'github', 'stackoverflow']):
            url_score += 0.3
        
        total_score = (title_score + snippet_score + url_score) / 3
        scores.append(total_score)
    
    return sum(scores) / len(scores) if scores else 0.0

def get_synthesis_prompt(query: str, results: dict) -> str:
    """Generate synthesis prompt for combining results."""
    return f"""Synthesize the following search results for query: "{query}"

Results:
{results}

Provide a comprehensive summary that combines the best information from all sources.
Focus on key insights, facts, and actionable information.
Keep it concise but informative (2-3 paragraphs max)."""