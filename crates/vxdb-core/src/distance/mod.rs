mod cosine;
mod dot;
mod euclidean;
mod simd;

pub use cosine::CosineDistance;
pub use dot::DotProductDistance;
pub use euclidean::EuclideanDistance;

use crate::types::DistanceMetricKind;

/// All distance functions return a *distance* (lower = more similar).
/// For cosine: distance = 1 - cosine_similarity
/// For dot product: distance = -dot_product (negate so lower = higher similarity)
/// For euclidean: distance = L2 distance
pub trait DistanceMetric: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Monomorphized distance dispatch for the hot path.
///
/// Unlike `Box<dyn DistanceMetric>`, this `Copy` enum lets the compiler inline
/// `distance` into the index inner loops (a vtable call cannot be inlined),
/// which in turn lets LLVM auto-vectorize the per-dimension reduction. Same
/// math as the trait impls — both share the `#[inline]` free functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl Metric {
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => cosine::cosine(a, b),
            Metric::Euclidean => euclidean::euclidean(a, b),
            Metric::DotProduct => dot::dot(a, b),
        }
    }
}

pub fn metric_for_kind(kind: DistanceMetricKind) -> Metric {
    match kind {
        DistanceMetricKind::Cosine => Metric::Cosine,
        DistanceMetricKind::Euclidean => Metric::Euclidean,
        DistanceMetricKind::DotProduct => Metric::DotProduct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not a correctness test. Run manually:
    /// `cargo test -p vxdb-core --release -- --ignored bench_distance --nocapture`
    #[test]
    #[ignore = "micro-benchmark, run explicitly in release mode"]
    fn bench_distance_throughput() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::time::Instant;
        const DIM: usize = 768;
        const ROWS: usize = 20_000;
        const REPS: usize = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..DIM * ROWS)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        let q: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let mut acc = 0.0f32;
        let t = Instant::now();
        for _ in 0..REPS {
            for row in data.chunks_exact(DIM) {
                acc += cosine::cosine_scalar(row, &q);
            }
        }
        let scalar = t.elapsed();

        let mut acc2 = 0.0f32;
        let t = Instant::now();
        for _ in 0..REPS {
            for row in data.chunks_exact(DIM) {
                acc2 += Metric::Cosine.distance(row, &q);
            }
        }
        let dispatch = t.elapsed();

        let per = |d: std::time::Duration| d.as_nanos() as f64 / (ROWS * REPS) as f64;
        eprintln!(
            "cosine 768d  scalar:   {:>8.1} ns/vec  (acc={acc})",
            per(scalar)
        );
        eprintln!(
            "cosine 768d  dispatch: {:>8.1} ns/vec  (acc={acc2})",
            per(dispatch)
        );
        eprintln!("speedup: {:.2}x", per(scalar) / per(dispatch));
    }
}
