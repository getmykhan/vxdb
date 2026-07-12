//! Explicit SIMD kernels for the distance hot path.
//!
//! All functions are `unsafe` because of `#[target_feature]`. Dispatch rules:
//! on aarch64, NEON (AdvSIMD) is part of the baseline Armv8-A ISA that every
//! Rust aarch64 std target assumes, so kernels run unconditionally; on x86_64,
//! callers must check `avx2_available()` first (the result is cached by std).
//! Kernels clamp to `min(a.len(), b.len())` so a length mismatch can never
//! read out of bounds, matching the scalar zip semantics.

#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        -sum
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
            i += 16;
        }
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc0 = vfmaq_f32(acc0, d, d);
            i += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum.sqrt()
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut dot0 = vdupq_n_f32(0.0);
        let mut dot1 = vdupq_n_f32(0.0);
        let mut na0 = vdupq_n_f32(0.0);
        let mut na1 = vdupq_n_f32(0.0);
        let mut nb0 = vdupq_n_f32(0.0);
        let mut nb1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= n {
            let x0 = vld1q_f32(pa.add(i));
            let y0 = vld1q_f32(pb.add(i));
            let x1 = vld1q_f32(pa.add(i + 4));
            let y1 = vld1q_f32(pb.add(i + 4));
            dot0 = vfmaq_f32(dot0, x0, y0);
            dot1 = vfmaq_f32(dot1, x1, y1);
            na0 = vfmaq_f32(na0, x0, x0);
            na1 = vfmaq_f32(na1, x1, x1);
            nb0 = vfmaq_f32(nb0, y0, y0);
            nb1 = vfmaq_f32(nb1, y1, y1);
            i += 8;
        }
        while i + 4 <= n {
            let x = vld1q_f32(pa.add(i));
            let y = vld1q_f32(pb.add(i));
            dot0 = vfmaq_f32(dot0, x, y);
            na0 = vfmaq_f32(na0, x, x);
            nb0 = vfmaq_f32(nb0, y, y);
            i += 4;
        }
        let mut dot = vaddvq_f32(vaddq_f32(dot0, dot1));
        let mut norm_a = vaddvq_f32(vaddq_f32(na0, na1));
        let mut norm_b = vaddvq_f32(vaddq_f32(nb0, nb1));
        while i < n {
            let (x, y) = (*pa.add(i), *pb.add(i));
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
            i += 1;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use core::arch::x86_64::*;

    /// Horizontal sum of one 256-bit register.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            i += 16;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let mut sum = hsum(_mm256_add_ps(acc0, acc1));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        -sum
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            i += 16;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc0 = _mm256_fmadd_ps(d, d, acc0);
            i += 8;
        }
        let mut sum = hsum(_mm256_add_ps(acc0, acc1));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum.sqrt()
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut dot_acc = _mm256_setzero_ps();
        let mut na_acc = _mm256_setzero_ps();
        let mut nb_acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(pa.add(i));
            let y = _mm256_loadu_ps(pb.add(i));
            dot_acc = _mm256_fmadd_ps(x, y, dot_acc);
            na_acc = _mm256_fmadd_ps(x, x, na_acc);
            nb_acc = _mm256_fmadd_ps(y, y, nb_acc);
            i += 8;
        }
        let mut dot = hsum(dot_acc);
        let mut norm_a = hsum(na_acc);
        let mut norm_b = hsum(nb_acc);
        while i < n {
            let (x, y) = (*pa.add(i), *pb.add(i));
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
            i += 1;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }
}
