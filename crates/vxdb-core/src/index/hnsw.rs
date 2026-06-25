use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

use rand::Rng;

use crate::distance::Metric;
use crate::error::{VexError, VexResult};
use crate::types::{Metadata, SearchResult, VectorData};

use super::VectorIndex;

#[derive(Clone, Copy)]
struct Candidate {
    idx: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap ordering (closest first when popped from BinaryHeap used as max-heap of negated,
/// or used with Reverse). We use a wrapper instead.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-distance-first candidate for the "worst" tracking heap.
struct FarthestCandidate(Candidate);

impl PartialEq for FarthestCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}
impl Eq for FarthestCandidate {}
impl PartialOrd for FarthestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FarthestCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Reused "visited" set using version stamps instead of a `HashSet`.
///
/// `search_layer` is called millions of times during a build; allocating and
/// hashing into a fresh `HashSet` each time dominates the non-distance cost.
/// Here marking visited is an array write and "clearing" between searches is a
/// single counter bump (`start`), with a rare full clear only on u32 wraparound.
#[derive(Default)]
struct VisitedList {
    marks: Vec<u32>,
    cur: u32,
}

impl VisitedList {
    /// Begin a fresh logical visited-set covering `n` nodes (O(1) amortized).
    fn start(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.cur = 1;
        }
    }

    /// Mark `idx` visited; returns true if it was not already visited.
    #[inline]
    fn visit(&mut self, idx: usize) -> bool {
        if self.marks[idx] == self.cur {
            false
        } else {
            self.marks[idx] = self.cur;
            true
        }
    }
}

pub struct HnswConfig {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 150,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            m_max0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

struct Node {
    id: String,
    vector: VectorData,
    metadata: Metadata,
    /// neighbors[layer] = list of neighbor indices
    neighbors: Vec<Vec<usize>>,
}

pub struct HnswIndex {
    dimension: usize,
    metric: Metric,
    config: HnswConfig,
    nodes: Vec<Node>,
    id_to_idx: HashMap<String, usize>,
    entry_point: Option<usize>,
    max_layer: usize,
    /// Reused across inserts on the (single-threaded) build path.
    scratch_visited: VisitedList,
}

impl HnswIndex {
    pub fn new(dimension: usize, metric: Metric, config: HnswConfig) -> Self {
        Self {
            dimension,
            metric,
            config,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            scratch_visited: VisitedList::default(),
        }
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.config.ml).floor() as usize
    }

    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        self.metric.distance(a, b)
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<usize>,
        ef: usize,
        layer: usize,
        visited: &mut VisitedList,
    ) -> Vec<Candidate> {
        visited.start(self.nodes.len());
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<FarthestCandidate> = BinaryHeap::new();

        for &ep in &entry_points {
            visited.visit(ep);
            let d = self.dist(query, &self.nodes[ep].vector);
            let c = Candidate { idx: ep, distance: d };
            candidates.push(c);
            results.push(FarthestCandidate(c));
        }

        while let Some(closest) = candidates.pop() {
            let farthest_dist = results.peek().map(|f| f.0.distance).unwrap_or(f32::MAX);

            if closest.distance > farthest_dist {
                break;
            }

            let neighbors = &self.nodes[closest.idx].neighbors;
            if layer < neighbors.len() {
                for &neighbor_idx in &neighbors[layer] {
                    if !visited.visit(neighbor_idx) {
                        continue;
                    }

                    let d = self.dist(query, &self.nodes[neighbor_idx].vector);
                    let farthest_dist = results.peek().map(|f| f.0.distance).unwrap_or(f32::MAX);

                    if results.len() < ef || d < farthest_dist {
                        let c = Candidate {
                            idx: neighbor_idx,
                            distance: d,
                        };
                        candidates.push(c);
                        results.push(FarthestCandidate(c));

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|fc| fc.0)
            .collect()
    }

    fn select_neighbors_simple(&self, candidates: &[Candidate], m: usize) -> Vec<usize> {
        let mut sorted: Vec<&Candidate> = candidates.iter().collect();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        sorted.iter().take(m).map(|c| c.idx).collect()
    }

    fn connect_neighbors(&mut self, node_idx: usize, neighbors: &[usize], layer: usize) {
        let m_max = if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m
        };

        // Set forward links
        if layer < self.nodes[node_idx].neighbors.len() {
            self.nodes[node_idx].neighbors[layer] = neighbors.to_vec();
        }

        // Set reverse links and prune if needed
        for &neighbor_idx in neighbors {
            if layer < self.nodes[neighbor_idx].neighbors.len() {
                let has_link = self.nodes[neighbor_idx].neighbors[layer].contains(&node_idx);
                if !has_link {
                    self.nodes[neighbor_idx].neighbors[layer].push(node_idx);

                    if self.nodes[neighbor_idx].neighbors[layer].len() > m_max {
                        // Prune: keep closest m_max neighbors
                        let neighbor_vec = &self.nodes[neighbor_idx].vector;
                        let mut scored: Vec<Candidate> = self.nodes[neighbor_idx].neighbors[layer]
                            .iter()
                            .map(|&idx| Candidate {
                                idx,
                                distance: self.dist(neighbor_vec, &self.nodes[idx].vector),
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
                        });
                        self.nodes[neighbor_idx].neighbors[layer] =
                            scored.iter().take(m_max).map(|c| c.idx).collect();
                    }
                }
            }
        }
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, id: String, vector: VectorData, metadata: Metadata) -> VexResult<()> {
        if vector.len() != self.dimension {
            return Err(VexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if vector.is_empty() {
            return Err(VexError::EmptyVector);
        }

        // Handle upsert: delete old, then re-insert
        if self.id_to_idx.contains_key(&id) {
            self.delete(&id)?;
        }

        let node_level = self.random_level();
        let node_idx = self.nodes.len();

        let mut neighbors = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            neighbors.push(Vec::new());
        }

        self.nodes.push(Node {
            id: id.clone(),
            vector,
            metadata,
            neighbors,
        });
        self.id_to_idx.insert(id, node_idx);

        if self.nodes.len() == 1 {
            self.entry_point = Some(node_idx);
            self.max_layer = node_level;
            return Ok(());
        }

        let mut ep = self.entry_point.unwrap();
        let query = &self.nodes[node_idx].vector.clone();

        // Reuse the visited buffer across this insert's layer searches (and
        // across inserts, since we move it back). `mem::take` sidesteps the
        // borrow conflict between `&self` search_layer and `&mut self`.
        let mut visited = std::mem::take(&mut self.scratch_visited);

        // Traverse from top layer down to node_level + 1 (greedy search)
        let mut current_layer = self.max_layer;
        while current_layer > node_level {
            let results = self.search_layer(query, vec![ep], 1, current_layer, &mut visited);
            if let Some(nearest) = results.first() {
                ep = nearest.idx;
            }
            if current_layer == 0 {
                break;
            }
            current_layer -= 1;
        }

        // For each layer from min(node_level, max_layer) down to 0, find and connect neighbors
        let start_layer = node_level.min(self.max_layer);
        for layer in (0..=start_layer).rev() {
            let candidates =
                self.search_layer(query, vec![ep], self.config.ef_construction, layer, &mut visited);

            let neighbors = self.select_neighbors_simple(&candidates, self.config.m);
            self.connect_neighbors(node_idx, &neighbors, layer);

            if let Some(nearest) = candidates.first() {
                ep = nearest.idx;
            }
        }

        self.scratch_visited = visited;

        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_idx);
        }

        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> VexResult<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(VexError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let mut ep = self.entry_point.unwrap();

        // Queries can run concurrently under a read lock, so the buffer is local
        // (one allocation per query) rather than the shared build-path scratch.
        let mut visited = VisitedList::default();

        // Greedy descent from top layer to layer 1
        let mut current_layer = self.max_layer;
        while current_layer > 0 {
            let results = self.search_layer(query, vec![ep], 1, current_layer, &mut visited);
            if let Some(nearest) = results.first() {
                ep = nearest.idx;
            }
            current_layer -= 1;
        }

        // Search layer 0 with ef_search
        let ef = self.config.ef_search.max(top_k);
        let candidates = self.search_layer(query, vec![ep], ef, 0, &mut visited);

        let results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|c| self.id_to_idx.contains_key(&self.nodes[c.idx].id))
            .take(top_k)
            .map(|c| SearchResult {
                id: self.nodes[c.idx].id.clone(),
                score: c.distance,
                metadata: self.nodes[c.idx].metadata.clone(),
                document: None,
            })
            .collect();

        Ok(results)
    }

    fn delete(&mut self, id: &str) -> VexResult<bool> {
        // Soft-delete approach: mark as deleted by removing from id_to_idx.
        // The node remains in the graph but won't appear in results.
        // Full compaction would require rebuilding, kept simple for now.
        let _idx = match self.id_to_idx.remove(id) {
            Some(idx) => idx,
            None => return Ok(false),
        };
        Ok(true)
    }

    fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    fn contains(&self, id: &str) -> bool {
        self.id_to_idx.contains_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::flat::FlatIndex;
    use std::collections::{HashMap, HashSet};

    fn make_hnsw(dim: usize) -> HnswIndex {
        HnswIndex::new(
            dim,
            Metric::Cosine,
            HnswConfig::new(16, 200, 50),
        )
    }

    #[test]
    fn test_insert_and_search_basic() {
        let mut idx = make_hnsw(3);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();
        idx.insert("c".into(), vec![1.0, 0.1, 0.0], HashMap::new()).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_empty_search() {
        let idx = make_hnsw(3);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut idx = make_hnsw(3);
        let result = idx.insert("a".into(), vec![1.0, 0.0], HashMap::new());
        assert!(matches!(result, Err(VexError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_delete() {
        let mut idx = make_hnsw(3);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();

        assert!(idx.delete("a").unwrap());
        assert_eq!(idx.len(), 1);
        assert!(!idx.contains("a"));

        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn test_recall_against_flat() {
        let dim = 32;
        let n = 1000;
        let n_queries = 50;
        let top_k = 10;

        let mut rng = rand::thread_rng();
        let mut hnsw = HnswIndex::new(
            dim,
            Metric::Euclidean,
            HnswConfig::new(16, 200, 100),
        );
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);

        // Insert same vectors into both
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let id = format!("v{}", i);
            hnsw.insert(id.clone(), v.clone(), HashMap::new()).unwrap();
            flat.insert(id, v.clone(), HashMap::new()).unwrap();
            vectors.push(v);
        }

        // Test recall
        let mut total_recall = 0.0f64;
        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

            let hnsw_results = hnsw.search(&query, top_k).unwrap();
            let flat_results = flat.search(&query, top_k).unwrap();

            let hnsw_ids: HashSet<String> = hnsw_results.iter().map(|r| r.id.clone()).collect();
            let flat_ids: HashSet<String> = flat_results.iter().map(|r| r.id.clone()).collect();

            let overlap = hnsw_ids.intersection(&flat_ids).count();
            total_recall += overlap as f64 / top_k as f64;
        }

        let avg_recall = total_recall / n_queries as f64;
        eprintln!("HNSW recall@{}: {:.3}", top_k, avg_recall);
        assert!(
            avg_recall > 0.90,
            "HNSW recall@{} = {:.3}, expected > 0.90",
            top_k,
            avg_recall
        );
    }

    #[test]
    fn test_default_ef_search_lifts_recall() {
        // Motivates the default ef_search = 150 (raised from 50). On a harder,
        // higher-dimensional regime than `test_recall_against_flat`, the old
        // default left a lot of recall on the table; raising ef_search trades
        // some of vxdb's large query-latency headroom for materially better
        // recall. This guards against regressing the default back down.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let dim = 128;
        let n = 3000;
        let n_queries = 100;
        let top_k = 10;

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut hnsw = HnswIndex::new(
            dim,
            Metric::Euclidean,
            HnswConfig::new(16, 200, 50),
        );
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);

        let mut queries: Vec<Vec<f32>> = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let id = format!("v{}", i);
            hnsw.insert(id.clone(), v.clone(), HashMap::new()).unwrap();
            flat.insert(id, v.clone(), HashMap::new()).unwrap();
        }
        for _ in 0..n_queries {
            queries.push((0..dim).map(|_| rng.gen::<f32>()).collect());
        }

        let recall_at = |hnsw: &HnswIndex| -> f64 {
            let mut total = 0.0f64;
            for q in &queries {
                let h: HashSet<String> =
                    hnsw.search(q, top_k).unwrap().into_iter().map(|r| r.id).collect();
                let f: HashSet<String> =
                    flat.search(q, top_k).unwrap().into_iter().map(|r| r.id).collect();
                total += h.intersection(&f).count() as f64 / top_k as f64;
            }
            total / n_queries as f64
        };

        hnsw.config.ef_search = 50;
        let recall_old = recall_at(&hnsw);
        hnsw.config.ef_search = HnswConfig::default().ef_search; // 150
        let recall_new = recall_at(&hnsw);
        eprintln!(
            "recall@{} dim={}: ef=50 -> {:.3}, ef={} -> {:.3}",
            top_k, dim, recall_old, HnswConfig::default().ef_search, recall_new
        );

        // The new default must be a clear improvement and a high-quality result.
        assert!(
            recall_new > recall_old + 0.05,
            "raising ef_search should lift recall materially: {:.3} -> {:.3}",
            recall_old,
            recall_new
        );
        assert!(
            recall_new > 0.85,
            "recall at default ef_search = {:.3}, expected > 0.85",
            recall_new
        );
    }

    #[test]
    fn test_single_element() {
        let mut idx = make_hnsw(2);
        idx.insert("only".into(), vec![1.0, 0.0], HashMap::new()).unwrap();

        let results = idx.search(&[1.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "only");
    }

    #[test]
    fn test_upsert() {
        let mut idx = make_hnsw(3);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], HashMap::new()).unwrap();
        idx.insert("a".into(), vec![0.0, 1.0, 0.0], HashMap::new()).unwrap();

        // After upsert, there should only be one logical entry
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_metadata_preserved() {
        let mut idx = make_hnsw(3);
        let meta = HashMap::from([("tag".into(), serde_json::json!("test"))]);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], meta).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].metadata["tag"], "test");
    }
}
