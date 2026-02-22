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

pub fn metric_for_kind(kind: DistanceMetricKind) -> Box<dyn DistanceMetric> {
    match kind {
        DistanceMetricKind::Cosine => Box::new(CosineDistance),
        DistanceMetricKind::Euclidean => Box::new(EuclideanDistance),
        DistanceMetricKind::DotProduct => Box::new(DotProductDistance),
    }
}
