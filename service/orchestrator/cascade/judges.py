"""Specialized judges for the cascade system."""

import logging
import time
import re
from typing import Dict, List, Any, Optional
from .cascade import BaseJudge, JudgeResult

logger = logging.getLogger(__name__)

class HeuristicJudge(BaseJudge):
    """Fast heuristic-based judge."""
    
    def __init__(self):
        super().__init__("heuristic")
    
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> JudgeResult:
        """Evaluate results using heuristics."""
        start_time = time.time()
        
        if not results:
            return JudgeResult(0.0, 0.0, self.name, 0.0)
        
        # Calculate relevance scores
        relevance_scores = []
        for result in results:
            score = self._calculate_heuristic_score(query, result)
            relevance_scores.append(score)
        
        # Calculate overall metrics
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        max_relevance = max(relevance_scores)
        min_relevance = min(relevance_scores)
        
        # Calculate confidence based on score distribution
        score_variance = sum((s - avg_relevance) ** 2 for s in relevance_scores) / len(relevance_scores)
        confidence = min(1.0, max(0.0, 1.0 - score_variance))
        
        # Boost confidence for high-quality results
        if max_relevance > 0.8:
            confidence = min(1.0, confidence + 0.2)
        
        processing_time = (time.time() - start_time) * 1000
        
        return JudgeResult(
            relevance_score=avg_relevance,
            confidence=confidence,
            judge_type=self.name,
            processing_time_ms=processing_time,
            metadata={
                "max_relevance": max_relevance,
                "min_relevance": min_relevance,
                "score_variance": score_variance,
                "num_results": len(results)
            }
        )
    
    def _calculate_heuristic_score(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate heuristic relevance score for a single result."""
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        query_lower = query.lower()
        
        # Query term matches
        query_terms = set(query_lower.split())
        title_terms = set(title.split())
        snippet_terms = set(snippet.split())
        
        # Title relevance (higher weight)
        title_matches = query_terms.intersection(title_terms)
        title_score = len(title_matches) / max(1, len(query_terms)) * 0.6
        
        # Snippet relevance
        snippet_matches = query_terms.intersection(snippet_terms)
        snippet_score = len(snippet_matches) / max(1, len(query_terms)) * 0.4
        
        # Exact phrase bonus
        phrase_bonus = 0.0
        if query_lower in title:
            phrase_bonus += 0.2
        if query_lower in snippet:
            phrase_bonus += 0.1
        
        # Length penalty for very short results
        length_penalty = 0.0
        if len(title) < 10 or len(snippet) < 20:
            length_penalty = 0.1
        
        # Spam penalty
        spam_penalty = 0.0
        spam_indicators = ["click here", "read more", "learn more", "free", "cheap"]
        text = f"{title} {snippet}"
        if any(indicator in text for indicator in spam_indicators):
            spam_penalty = 0.2
        
        score = title_score + snippet_score + phrase_bonus - length_penalty - spam_penalty
        return max(0.0, min(1.0, score))

class LLMJudge(BaseJudge):
    """LLM-based judge for high-quality evaluation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__("llm")
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client."""
        try:
            import openai
            self.client = openai.OpenAI()
            logger.info(f"LLM judge initialized with {self.model_name}")
        except ImportError:
            logger.warning("OpenAI not available, LLM judge will use fallback")
            self.client = None
    
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> JudgeResult:
        """Evaluate results using LLM."""
        start_time = time.time()
        
        if not results:
            return JudgeResult(0.0, 0.0, self.name, 0.0)
        
        if not self.client:
            # Fallback to heuristic if LLM not available
            heuristic_judge = HeuristicJudge()
            result = heuristic_judge.evaluate(query, results)
            result.judge_type = self.name
            return result
        
        try:
            # Prepare results for LLM evaluation
            results_text = self._format_results_for_llm(query, results)
            
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(query, results_text)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert search quality evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse response
            evaluation = self._parse_llm_response(response.choices[0].message.content)
            
            processing_time = (time.time() - start_time) * 1000
            
            return JudgeResult(
                relevance_score=evaluation["relevance_score"],
                confidence=evaluation["confidence"],
                judge_type=self.name,
                processing_time_ms=processing_time,
                metadata=evaluation.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback to heuristic
            heuristic_judge = HeuristicJudge()
            result = heuristic_judge.evaluate(query, results)
            result.judge_type = self.name
            return result
    
    def _format_results_for_llm(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format results for LLM evaluation."""
        formatted = []
        for i, result in enumerate(results[:10]):  # Limit to top 10 for LLM
            formatted.append(f"{i+1}. {result.get('title', '')}\n   {result.get('snippet', '')[:200]}...")
        
        return "\n\n".join(formatted)
    
    def _create_evaluation_prompt(self, query: str, results_text: str) -> str:
        """Create evaluation prompt for LLM."""
        return f"""
Evaluate the relevance of these search results for the query: "{query}"

Results:
{results_text}

Please provide:
1. Overall relevance score (0.0 to 1.0)
2. Confidence level (0.0 to 1.0)
3. Brief explanation

Format your response as:
RELEVANCE: [score]
CONFIDENCE: [score]
EXPLANATION: [brief explanation]
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response."""
        try:
            lines = response.strip().split('\n')
            relevance_score = 0.5
            confidence = 0.5
            explanation = ""
            
            for line in lines:
                if line.startswith("RELEVANCE:"):
                    relevance_score = float(line.split(":", 1)[1].strip())
                elif line.startswith("CONFIDENCE:"):
                    confidence = float(line.split(":", 1)[1].strip())
                elif line.startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()
            
            return {
                "relevance_score": max(0.0, min(1.0, relevance_score)),
                "confidence": max(0.0, min(1.0, confidence)),
                "metadata": {"explanation": explanation}
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "relevance_score": 0.5,
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

class ConfidenceJudge(BaseJudge):
    """Judge that combines multiple evaluation methods."""
    
    def __init__(self):
        super().__init__("confidence")
        self.heuristic_judge = HeuristicJudge()
        self.llm_judge = LLMJudge()
    
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> JudgeResult:
        """Evaluate using multiple methods and combine results."""
        start_time = time.time()
        
        if not results:
            return JudgeResult(0.0, 0.0, self.name, 0.0)
        
        # Get heuristic evaluation
        heuristic_result = self.heuristic_judge.evaluate(query, results)
        
        # Get LLM evaluation if heuristic confidence is low
        llm_result = None
        if heuristic_result.confidence < 0.6:
            llm_result = self.llm_judge.evaluate(query, results)
        
        # Combine results
        if llm_result and llm_result.confidence > heuristic_result.confidence:
            # Use LLM result if it's more confident
            final_result = llm_result
            final_result.judge_type = self.name
        else:
            # Use heuristic result
            final_result = heuristic_result
            final_result.judge_type = self.name
        
        # Update metadata
        final_result.metadata.update({
            "heuristic_score": heuristic_result.relevance_score,
            "heuristic_confidence": heuristic_result.confidence,
            "llm_used": llm_result is not None,
            "llm_score": llm_result.relevance_score if llm_result else None,
            "llm_confidence": llm_result.confidence if llm_result else None
        })
        
        processing_time = (time.time() - start_time) * 1000
        final_result.processing_time_ms = processing_time
        
        return final_result
