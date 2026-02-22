use super::DistanceMetric;

pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
