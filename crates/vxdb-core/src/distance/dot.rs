use super::DistanceMetric;

/// Negative dot product (so lower = more similar). Free function shared by the
/// trait impl and the monomorphized `Metric` enum. Dispatches to a SIMD kernel
/// when the CPU supports one; `dot_scalar` is the reference and the fallback.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every Rust aarch64 std target.
        return unsafe { super::simd::neon::dot(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if super::simd::avx2_available() {
            // SAFETY: AVX2 and FMA support verified at runtime just above.
            return unsafe { super::simd::avx2::dot(a, b) };
        }
    }
    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

/// Scalar reference. Multiple accumulators let LLVM auto-vectorize the
/// reduction (see `euclidean` for the rationale).
#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let x: &[f32; LANES] = x.try_into().unwrap();
        let y: &[f32; LANES] = y.try_into().unwrap();
        for j in 0..LANES {
            acc[j] += x[j] * y[j];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    -sum
}

pub struct DotProductDistance;

impl DistanceMetric for DotProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        dot(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_naive_across_dims() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(13);
        for dim in [1usize, 7, 8, 9, 16, 100, 384, 769] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let got = dot(&a, &b);
            let naive: f32 = -a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>();
            assert!(
                (got - naive).abs() <= 1e-3 * naive.abs().max(1.0),
                "dim={dim}: {got} vs naive {naive}"
            );
        }
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn test_identical_unit_vectors() {
        let d = DotProductDistance;
        // dot([1,0],[1,0]) = 1, distance = -1
        assert!((d.distance(&[1.0, 0.0], &[1.0, 0.0]) - (-1.0)).abs() < EPS);
    }

    #[test]
    fn test_orthogonal() {
        let d = DotProductDistance;
        // dot([1,0],[0,1]) = 0, distance = 0
        assert!((d.distance(&[1.0, 0.0], &[0.0, 1.0])).abs() < EPS);
    }

    #[test]
    fn test_opposite() {
        let d = DotProductDistance;
        // dot([1,0],[-1,0]) = -1, distance = 1
        assert!((d.distance(&[1.0, 0.0], &[-1.0, 0.0]) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_known_value() {
        let d = DotProductDistance;
        // dot([1,2,3],[4,5,6]) = 4+10+18 = 32, distance = -32
        assert!((d.distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - (-32.0)).abs() < EPS);
    }

    #[test]
    fn test_higher_similarity_is_lower_distance() {
        let d = DotProductDistance;
        let aligned = d.distance(&[1.0, 0.0], &[1.0, 0.0]);
        let orthogonal = d.distance(&[1.0, 0.0], &[0.0, 1.0]);
        // aligned should have lower distance than orthogonal
        assert!(aligned < orthogonal);
    }

    #[test]
    fn test_commutativity() {
        let d = DotProductDistance;
        let d1 = d.distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let d2 = d.distance(&[4.0, 5.0, 6.0], &[1.0, 2.0, 3.0]);
        assert!((d1 - d2).abs() < EPS);
    }

    #[test]
    fn test_dispatch_matches_scalar_across_dims() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(101);
        for dim in [
            0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100, 384, 768, 769, 1536,
        ] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let got = dot(&a, &b);
            let scalar = dot_scalar(&a, &b);
            let naive = -(a
                .iter()
                .zip(&b)
                .map(|(x, y)| (*x as f64) * (*y as f64))
                .sum::<f64>()) as f32;
            assert!(
                (got - naive).abs() <= 1e-3 * naive.abs().max(1.0),
                "dim={dim}: dispatch {got} vs f64 naive {naive}"
            );
            assert!(
                (scalar - naive).abs() <= 1e-3 * naive.abs().max(1.0),
                "dim={dim}: scalar {scalar} vs f64 naive {naive}"
            );
        }
    }
}
