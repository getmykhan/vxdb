use super::DistanceMetric;

pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
