use super::DistanceMetric;

/// L2 distance. Free function so both the `DistanceMetric` trait impl and the
/// monomorphized `Metric` enum share one `#[inline]` code path.
///
/// Uses `LANES` independent accumulators so LLVM can auto-vectorize the
/// reduction (NEON/SSE are baseline on arm64/x86-64) — a single running sum
/// can't vectorize because IEEE f32 addition isn't associative. Zero deps, no
/// `unsafe`, no intrinsics. Vectors shorter than `LANES` fall to the scalar
/// remainder and match the naive result exactly.
#[inline]
pub(crate) fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let x: &[f32; LANES] = x.try_into().unwrap();
        let y: &[f32; LANES] = y.try_into().unwrap();
        for j in 0..LANES {
            let d = x[j] - y[j];
            acc[j] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum.sqrt()
}

pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        euclidean(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_naive_across_dims() {
        // Exercises the multi-accumulator/chunked path at dims around and well
        // past the 8-wide block (the small-vector tests only hit the remainder).
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        for dim in [1usize, 7, 8, 9, 16, 100, 384, 769] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let got = euclidean(&a, &b);
            let naive = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            assert!(
                (got - naive).abs() <= 1e-3 * naive.max(1.0),
                "dim={dim}: {got} vs naive {naive}"
            );
        }
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn test_identical_vectors() {
        let d = EuclideanDistance;
        assert!((d.distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < EPS);
    }

    #[test]
    fn test_unit_distance() {
        let d = EuclideanDistance;
        // sqrt((1-0)^2) = 1
        assert!((d.distance(&[0.0], &[1.0]) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_3_4_5_triangle() {
        let d = EuclideanDistance;
        // sqrt(3^2 + 4^2) = 5
        assert!((d.distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_known_3d() {
        let d = EuclideanDistance;
        // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9+9+9) = sqrt(27) ≈ 5.196
        let expected = 27.0f32.sqrt();
        assert!((d.distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - expected).abs() < EPS);
    }

    #[test]
    fn test_symmetry() {
        let d = EuclideanDistance;
        let d1 = d.distance(&[1.0, 2.0], &[3.0, 4.0]);
        let d2 = d.distance(&[3.0, 4.0], &[1.0, 2.0]);
        assert!((d1 - d2).abs() < EPS);
    }

    #[test]
    fn test_triangle_inequality() {
        let d = EuclideanDistance;
        let a = &[0.0, 0.0];
        let b = &[1.0, 0.0];
        let c = &[0.0, 1.0];
        let ab = d.distance(a, b);
        let bc = d.distance(b, c);
        let ac = d.distance(a, c);
        assert!(ac <= ab + bc + EPS);
    }
}
