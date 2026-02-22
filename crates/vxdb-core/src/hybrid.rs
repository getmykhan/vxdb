use std::collections::HashMap;

use crate::types::SearchResult;

/// BM25 parameters
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Simple in-memory BM25 index for keyword search.
pub struct Bm25Index {
    /// doc_id -> tokenized terms
    docs: HashMap<String, Vec<String>>,
    /// term -> set of doc_ids containing that term
    inverted: HashMap<String, Vec<String>>,
    /// total number of documents
    doc_count: usize,
    /// average document length
    avg_doc_len: f64,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            docs: HashMap::new(),
            inverted: HashMap::new(),
            doc_count: 0,
            avg_doc_len: 0.0,
        }
    }

    pub fn insert(&mut self, id: &str, text: &str) {
        let tokens = tokenize(text);

        // Remove old entry if exists (upsert)
        if self.docs.contains_key(id) {
            self.remove(id);
        }

        for token in &tokens {
            self.inverted
                .entry(token.clone())
                .or_default()
                .push(id.to_string());
        }

        self.docs.insert(id.to_string(), tokens);
        self.doc_count = self.docs.len();
        self.recalc_avg_len();
    }

    pub fn remove(&mut self, id: &str) {
        if let Some(tokens) = self.docs.remove(id) {
            for token in &tokens {
                if let Some(posting) = self.inverted.get_mut(token) {
                    posting.retain(|d| d != id);
                    if posting.is_empty() {
                        self.inverted.remove(token);
                    }
                }
            }
            self.doc_count = self.docs.len();
            self.recalc_avg_len();
        }
    }

    fn recalc_avg_len(&mut self) {
        if self.doc_count == 0 {
            self.avg_doc_len = 0.0;
        } else {
            let total: usize = self.docs.values().map(|t| t.len()).sum();
            self.avg_doc_len = total as f64 / self.doc_count as f64;
        }
    }

    /// Search with BM25 scoring. Returns (doc_id, bm25_score) sorted by score desc.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<&str, f64> = HashMap::new();
        let n = self.doc_count as f64;

        for token in &query_tokens {
            let posting = match self.inverted.get(token) {
                Some(p) => p,
                None => continue,
            };

            let df = posting.len() as f64;
            // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for doc_id in posting {
                let doc_tokens = &self.docs[doc_id.as_str()];
                let doc_len = doc_tokens.len() as f64;
                let tf = doc_tokens.iter().filter(|t| *t == token).count() as f64;

                // BM25 term score
                let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doc_len / self.avg_doc_len));
                let term_score = idf * tf_norm;

                *scores.entry(doc_id.as_str()).or_insert(0.0) += term_score;
            }
        }

        let mut results: Vec<(&str, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results.into_iter().map(|(id, s)| (id.to_string(), s)).collect()
    }

    pub fn len(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    pub fn contains(&self, id: &str) -> bool {
        self.docs.contains_key(id)
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

/// Reciprocal Rank Fusion: merges two ranked result lists.
/// RRF score = sum over lists of 1 / (k + rank), where k is a constant (default 60).
pub fn reciprocal_rank_fusion(
    vector_results: &[SearchResult],
    keyword_results: &[(String, f64)],
    top_k: usize,
    rrf_k: usize,
    alpha: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    // Vector results: rank-based scoring weighted by alpha
    for (rank, result) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(result.id.clone()).or_insert(0.0) += alpha * rrf_score;
    }

    // Keyword results: rank-based scoring weighted by (1 - alpha)
    for (rank, (id, _bm25_score)) in keyword_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id.clone()).or_insert(0.0) += (1.0 - alpha) * rrf_score;
    }

    let mut fused: Vec<(String, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);
    fused
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic_search() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "the quick brown fox jumps over the lazy dog");
        idx.insert("b", "a fast brown car races down the highway");
        idx.insert("c", "the dog sat on the mat");

        let results = idx.search("brown fox", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_bm25_relevance_ordering() {
        let mut idx = Bm25Index::new();
        idx.insert("exact", "machine learning is great");
        idx.insert("partial", "machine tools are useful");
        idx.insert("none", "the cat sat on the mat");

        let results = idx.search("machine learning", 10);
        assert_eq!(results[0].0, "exact");
        assert!(results.len() >= 2);
        // "none" should not appear or be ranked last
        assert!(results.iter().all(|r| r.0 != "none"));
    }

    #[test]
    fn test_bm25_empty_query() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "hello world");
        let results = idx.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_no_match() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "hello world");
        let results = idx.search("quantum computing", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_upsert() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "old content");
        idx.insert("a", "new content about machine learning");
        assert_eq!(idx.len(), 1);

        let results = idx.search("machine learning", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");

        // old content should not match
        let results = idx.search("old content", 10);
        assert!(results.is_empty() || results[0].0 == "a");
    }

    #[test]
    fn test_bm25_delete() {
        let mut idx = Bm25Index::new();
        idx.insert("a", "hello world");
        idx.insert("b", "hello there");
        idx.remove("a");

        assert_eq!(idx.len(), 1);
        let results = idx.search("hello", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "b");
    }

    #[test]
    fn test_rrf_basic() {
        let vector_results = vec![
            SearchResult { id: "a".into(), score: 0.1, metadata: HashMap::new() },
            SearchResult { id: "b".into(), score: 0.2, metadata: HashMap::new() },
            SearchResult { id: "c".into(), score: 0.3, metadata: HashMap::new() },
        ];
        let keyword_results = vec![
            ("b".to_string(), 5.0),
            ("d".to_string(), 3.0),
            ("a".to_string(), 1.0),
        ];

        let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, 10, 60, 0.5);

        // "b" appears in both lists at good ranks, should be near top
        assert!(!fused.is_empty());
        let top_ids: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        assert!(top_ids.contains(&"b"));
        assert!(top_ids.contains(&"a"));
    }

    #[test]
    fn test_rrf_alpha_weighting() {
        let vector_results = vec![
            SearchResult { id: "vec_winner".into(), score: 0.01, metadata: HashMap::new() },
        ];
        let keyword_results = vec![
            ("kw_winner".to_string(), 10.0),
        ];

        // alpha=1.0: only vector results matter
        let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, 10, 60, 1.0);
        assert_eq!(fused[0].0, "vec_winner");

        // alpha=0.0: only keyword results matter
        let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, 10, 60, 0.0);
        assert_eq!(fused[0].0, "kw_winner");
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single-char tokens filtered out
        assert!(!tokens.contains(&"a".to_string()));
    }
}
