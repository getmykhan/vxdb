mod cosine;
mod dot;
mod euclidean;

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
