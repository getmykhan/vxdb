use super::DistanceMetric;

/// Cosine distance (1 - cosine similarity). Free function shared by the trait
/// impl and the monomorphized `Metric` enum. Multiple accumulators let LLVM
/// auto-vectorize the three reductions (see `euclidean` for the rationale).
#[inline]
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut adot = [0.0f32; LANES];
    let mut anorm_a = [0.0f32; LANES];
    let mut anorm_b = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let x: &[f32; LANES] = x.try_into().unwrap();
        let y: &[f32; LANES] = y.try_into().unwrap();
        for j in 0..LANES {
            adot[j] += x[j] * y[j];
            anorm_a[j] += x[j] * x[j];
            anorm_b[j] += y[j] * y[j];
        }
    }
    let mut dot: f32 = adot.iter().sum();
    let mut norm_a: f32 = anorm_a.iter().sum();
    let mut norm_b: f32 = anorm_b.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot / denom)
}

pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_naive_across_dims() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(11);
        for dim in [1usize, 7, 8, 9, 16, 100, 384, 769] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let got = cosine(&a, &b);
            let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let naive = 1.0 - dot / (na * nb);
            assert!(
                (got - naive).abs() <= 1e-3,
                "dim={dim}: {got} vs naive {naive}"
            );
        }
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn test_identical_vectors() {
        let d = CosineDistance;
        assert!((d.distance(&[1.0, 0.0], &[1.0, 0.0])).abs() < EPS);
        assert!((d.distance(&[3.0, 4.0], &[3.0, 4.0])).abs() < EPS);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let d = CosineDistance;
        // cos(90°) = 0, distance = 1 - 0 = 1
        assert!((d.distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_opposite_vectors() {
        let d = CosineDistance;
        // cos(180°) = -1, distance = 1 - (-1) = 2
        assert!((d.distance(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_zero_vector() {
        let d = CosineDistance;
        assert!((d.distance(&[0.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_known_value() {
        let d = CosineDistance;
        // a=[1,2,3], b=[4,5,6]
        // dot = 4+10+18 = 32
        // |a| = sqrt(14), |b| = sqrt(77)
        // cos = 32 / sqrt(14*77) = 32 / sqrt(1078) ≈ 0.97463
        let expected = 1.0 - 32.0 / (14.0f32.sqrt() * 77.0f32.sqrt());
        assert!((d.distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - expected).abs() < EPS);
    }

    #[test]
    fn test_scaling_invariance() {
        let d = CosineDistance;
        let d1 = d.distance(&[1.0, 2.0], &[3.0, 4.0]);
        let d2 = d.distance(&[2.0, 4.0], &[6.0, 8.0]);
        assert!((d1 - d2).abs() < EPS);
    }
}
