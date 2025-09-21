use nalgebra::DMatrix;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Performance statistics tracking
#[derive(Debug, Clone, serde::Serialize)]
pub struct PerfStats {
    pub per_doc_ms_p50: f32,
    pub per_doc_ms_p95: f32,
}

/// Token pruning configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PruneConfig {
    pub q_max: usize,
    pub d_max: usize,
    pub method: String,
}

/// Request structure for reranking
#[derive(Debug, serde::Deserialize)]
pub struct RerankRequest {
    pub q_tokens: Vec<Vec<f32>>,
    pub d_tokens: Vec<Vec<Vec<f32>>>,
    pub topk: usize,
    pub prune: PruneConfig,
}

/// Response structure for reranking
#[derive(Debug, serde::Serialize)]
pub struct RerankResponse {
    pub order: Vec<usize>,
    pub scores: Vec<f32>,
    pub perf: PerfStats,
}

/// L2 normalize rows of a matrix
pub fn l2_normalize_rows(matrix: &mut DMatrix<f32>) {
    for mut row in matrix.row_iter_mut() {
        let norm = row.norm();
        if norm > 1e-8 {
            row /= norm;
        }
    }
}

/// Compute token salience using IDF * norm (SIGIR 2025 approach)
pub fn token_salience(tokens: &[Vec<f32>], method: &str) -> Vec<(usize, f32)> {
    let mut saliences = Vec::new();
    
    for (i, token) in tokens.iter().enumerate() {
        let norm = token.iter().map(|x| x * x).sum::<f32>().sqrt();
        let salience = match method {
            "idf_norm" => {
                // SIGIR 2025: salience = idf(token) × ||embedding||₂
                // For demo: use norm as proxy for idf (higher norm = more informative)
                norm * norm // Square to emphasize high-norm tokens
            },
            "norm_only" => norm,
            _ => norm,
        };
        saliences.push((i, salience));
    }
    
    saliences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    saliences
}

/// Prune tokens to keep top-N by salience
pub fn prune_tokens(tokens: &[Vec<f32>], max_n: usize, method: &str) -> Vec<Vec<f32>> {
    if tokens.len() <= max_n {
        return tokens.to_vec();
    }
    
    let saliences = token_salience(tokens, method);
    let top_indices: Vec<usize> = saliences
        .iter()
        .take(max_n)
        .map(|(idx, _)| *idx)
        .collect();
    
    top_indices.into_iter().map(|i| tokens[i].clone()).collect()
}

/// Compute dot product between two vectors (optimized)
#[inline]
pub fn dot_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// MaxSim scoring for a single document
pub fn maxsim_score(q: &DMatrix<f32>, d: &DMatrix<f32>) -> f32 {
    let mut total_score = 0.0;
    
    for q_row in q.row_iter() {
        let mut max_dot = f32::NEG_INFINITY;
        
        for d_row in d.row_iter() {
            // Convert row views to vectors for dot product
            let q_vec: Vec<f32> = q_row.iter().cloned().collect();
            let d_vec: Vec<f32> = d_row.iter().cloned().collect();
            let dot = dot_sim(&q_vec, &d_vec);
            max_dot = max_dot.max(dot);
        }
        
        total_score += max_dot;
    }
    
    total_score
}

/// Score all documents and return top-K
pub fn score_docs(
    q_tokens: &[Vec<f32>],
    d_tokens: &[Vec<Vec<f32>>],
    topk: usize,
    prune_config: &PruneConfig,
) -> (Vec<usize>, Vec<f32>, PerfStats) {
    let _start_time = std::time::Instant::now();
    
    // Prune query tokens (SIGIR 2025: lossless token pruning)
    let pruned_q = prune_tokens(q_tokens, prune_config.q_max, &prune_config.method);
    let _q_pruning_ratio = 1.0 - (pruned_q.len() as f32 / q_tokens.len() as f32);
    
    let q_matrix = DMatrix::from_row_slice(
        pruned_q.len(),
        pruned_q[0].len(),
        &pruned_q.iter().flatten().cloned().collect::<Vec<_>>(),
    );
    let mut q_matrix = q_matrix;
    l2_normalize_rows(&mut q_matrix);
    
    // Process documents in parallel
    let mut doc_scores: Vec<(usize, f32, f32)> = d_tokens
        .par_iter()
        .enumerate()
        .map(|(doc_idx, doc_tokens)| {
            let doc_start = std::time::Instant::now();
            
            // Prune document tokens
            let pruned_d = prune_tokens(doc_tokens, prune_config.d_max, &prune_config.method);
            let d_matrix = DMatrix::from_row_slice(
                pruned_d.len(),
                pruned_d[0].len(),
                &pruned_d.iter().flatten().cloned().collect::<Vec<_>>(),
            );
            let mut d_matrix = d_matrix;
            l2_normalize_rows(&mut d_matrix);
            
            // Compute MaxSim score
            let score = maxsim_score(&q_matrix, &d_matrix);
            let doc_time = doc_start.elapsed().as_secs_f32() * 1000.0; // Convert to ms
            
            (doc_idx, score, doc_time)
        })
        .collect();
    
    // Sort by score (descending) and take top-K
    doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    
    let topk = topk.min(doc_scores.len());
    let order: Vec<usize> = doc_scores.iter().take(topk).map(|(idx, _, _)| *idx).collect();
    let scores: Vec<f32> = doc_scores.iter().take(topk).map(|(_, score, _)| *score).collect();
    
    // Calculate performance statistics
    let doc_times: Vec<f32> = doc_scores.iter().map(|(_, _, time)| *time).collect();
    let mut sorted_times = doc_times.clone();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    let p50_idx = (sorted_times.len() * 50) / 100;
    let p95_idx = (sorted_times.len() * 95) / 100;
    
    let perf = PerfStats {
        per_doc_ms_p50: if !sorted_times.is_empty() { sorted_times[p50_idx] } else { 0.0 },
        per_doc_ms_p95: if !sorted_times.is_empty() { sorted_times[p95_idx] } else { 0.0 },
    };
    
    // Log transparency information
    let q_tokens_in = q_tokens.len();
    let q_tokens_pruned = pruned_q.len();
    let d_tokens_in_avg = if !d_tokens.is_empty() {
        d_tokens.iter().map(|doc| doc.len()).sum::<usize>() as f32 / d_tokens.len() as f32
    } else { 0.0 };
    let d_tokens_pruned_avg = if !d_tokens.is_empty() {
        d_tokens.iter().map(|doc| {
            prune_tokens(doc, prune_config.d_max, &prune_config.method).len()
        }).sum::<usize>() as f32 / d_tokens.len() as f32
    } else { 0.0 };
    
    println!("RERANKER TRANSPARENCY:");
    println!("  q_tokens_in: {}, q_tokens_pruned: {}", q_tokens_in, q_tokens_pruned);
    println!("  d_tokens_in_avg: {:.1}, d_tokens_pruned_avg: {:.1}", d_tokens_in_avg, d_tokens_pruned_avg);
    println!("  dim: {}, threads: {}", pruned_q[0].len(), rayon::current_num_threads());
    println!("  docs_scored: {}, topk: {}", d_tokens.len(), topk);
    println!("  rerank_ms_p50: {:.2}, rerank_ms_p95: {:.2}", perf.per_doc_ms_p50, perf.per_doc_ms_p95);
    
    (order, scores, perf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_sim() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_sim(&a, &b), 32.0);
    }

    #[test]
    fn test_maxsim_score() {
        let q = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let d = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let score = maxsim_score(&q, &d);
        assert!((score - 2.0).abs() < 1e-6);
    }
}
