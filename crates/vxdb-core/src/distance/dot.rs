use super::DistanceMetric;

pub struct DotProductDistance;

impl DistanceMetric for DotProductDistance {
    /// Returns negative dot product so that lower = more similar.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
        }
        -dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
