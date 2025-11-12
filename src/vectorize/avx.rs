use core::ops::*;
use std::{arch::x86_64::*, fmt::Debug};

use crate::vectorize::NumPar;

#[derive(Clone, Copy)]
pub struct AVXf32x8(pub __m256);

impl<Rhs: Into<AVXf32x8>> Add<Rhs> for AVXf32x8 {
    type Output = AVXf32x8;
    #[inline]
    fn add(self, rhs: Rhs) -> Self::Output {
        AVXf32x8(unsafe { _mm256_add_ps(self.0, rhs.into().0) })
    }
}
impl<Rhs: Into<AVXf32x8>> Sub<Rhs> for AVXf32x8 {
    type Output = AVXf32x8;
    #[inline]
    fn sub(self, rhs: Rhs) -> Self::Output {
        AVXf32x8(unsafe { _mm256_sub_ps(self.0, rhs.into().0) })
    }
}
impl<Rhs: Into<AVXf32x8>> Mul<Rhs> for AVXf32x8 {
    type Output = AVXf32x8;
    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        AVXf32x8(unsafe { _mm256_mul_ps(self.0, rhs.into().0) })
    }
}
impl<Rhs: Into<AVXf32x8>> Div<Rhs> for AVXf32x8 {
    type Output = AVXf32x8;
    #[inline]
    fn div(self, rhs: Rhs) -> Self::Output {
        AVXf32x8(unsafe { _mm256_div_ps(self.0, rhs.into().0) })
    }
}
impl<Rhs: Into<AVXf32x8>> Rem<Rhs> for AVXf32x8 {
    type Output = AVXf32x8;
    /// NOTE: this is actuall an implementation of mod, which rounds towards negative infinity
    /// instead of zero, because that's more useful in this application.
    #[inline]
    fn rem(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        // AVXf32x8(unsafe { _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(self.0, rhs.0)), rhs.0) })
        AVXf32x8(unsafe {
            _mm256_mul_ps(
                {
                    let x = _mm256_div_ps(self.0, rhs.0);
                    _mm256_sub_ps(x, _mm256_floor_ps(x))
                },
                rhs.0,
            )
        })
    }
}
impl AVXf32x8 {
    #[inline]
    pub fn reduce_sum(self) -> f32 {
        // https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
        unsafe {
            let sum_quad = _mm_add_ps(
                _mm256_castps256_ps128(self.0),
                _mm256_extractf128_ps::<1>(self.0),
            );
            let sum_dual = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
            let sum: __m128 = _mm_add_ss(sum_dual, _mm_shuffle_ps::<0x1>(sum_dual, sum_dual));
            return _mm_cvtss_f32(sum);
        }
    }
    #[inline]
    pub fn new(x0: f32, x1: f32, x2: f32, x3: f32, x4: f32, x5: f32, x6: f32, x7: f32) -> Self {
        Self(unsafe { _mm256_set_ps(x0, x1, x2, x3, x4, x5, x6, x7) })
    }
    #[inline]
    pub fn from_slice__sstride<const S: usize>(data: &[f32]) -> Self {
        Self::from_slice__dstride(data, S)
    }
    #[inline]
    pub fn from_slice__dstride(data: &[f32], s: usize) -> Self {
        Self(unsafe {
            _mm256_set_ps(
                data[0],
                data[1 * s],
                data[2 * s],
                data[3 * s],
                data[4 * s],
                data[5 * s],
                data[6 * s],
                data[7 * s],
            )
        })
    }
    #[inline]
    pub fn unpack(self) -> [f32; 8] {
        unsafe {
            let l4 = _mm256_castps256_ps128(self.0);
            let h4 = _mm256_extractf128_ps::<1>(self.0);
            return [
                f32::from_bits(_mm_extract_ps::<3>(h4) as u32),
                f32::from_bits(_mm_extract_ps::<2>(h4) as u32),
                f32::from_bits(_mm_extract_ps::<1>(h4) as u32),
                _mm_cvtss_f32(h4),
                f32::from_bits(_mm_extract_ps::<3>(l4) as u32),
                f32::from_bits(_mm_extract_ps::<2>(l4) as u32),
                f32::from_bits(_mm_extract_ps::<1>(l4) as u32),
                _mm_cvtss_f32(l4),
            ];
        }
    }
    #[inline]
    pub fn unpack_into<const S: usize>(self, data: &mut [f32]) {
        self.unpack_into__dstride(data, S);
    }
    #[inline]
    pub fn unpack_into__sstride<const S: usize>(self, data: &mut [f32]) {
        self.unpack_into__dstride(data, S);
    }
    #[inline]
    pub fn unpack_into__dstride(self, data: &mut [f32], s: usize) {
        unsafe {
            let l4 = _mm256_castps256_ps128(self.0);
            let h4 = _mm256_extractf128_ps::<1>(self.0);

            data[0 * s] = f32::from_bits(_mm_extract_ps::<3>(h4) as u32);
            data[1 * s] = f32::from_bits(_mm_extract_ps::<2>(h4) as u32);
            data[2 * s] = f32::from_bits(_mm_extract_ps::<1>(h4) as u32);
            data[3 * s] = _mm_cvtss_f32(h4);
            data[4 * s] = f32::from_bits(_mm_extract_ps::<3>(l4) as u32);
            data[5 * s] = f32::from_bits(_mm_extract_ps::<2>(l4) as u32);
            data[6 * s] = f32::from_bits(_mm_extract_ps::<1>(l4) as u32);
            data[7 * s] = _mm_cvtss_f32(l4);
        }
    }

    #[inline]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm256_floor_ps(self.0) })
    }
    #[inline]
    pub fn frac(self) -> Self {
        Self(unsafe { _mm256_sub_ps(self.0, _mm256_floor_ps(self.0)) })
    }
    #[inline]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }
    #[inline]
    pub fn div_floor_mul(self, rhs: impl Into<Self>) -> Self {
        let rhs = rhs.into();
        Self(unsafe { _mm256_mul_ps(_mm256_floor_ps(_mm256_div_ps(self.0, rhs.0)), rhs.0) })
    }
}
impl From<[f32; 8]> for AVXf32x8 {
    fn from(value: [f32; 8]) -> Self {
        Self(unsafe {
            _mm256_set_ps(
                value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
            )
        })
    }
}
impl From<&[f32]> for AVXf32x8 {
    fn from(value: &[f32]) -> Self {
        Self(unsafe {
            _mm256_set_ps(
                value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
            )
        })
    }
}
impl From<f32> for AVXf32x8 {
    fn from(value: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(value) })
    }
}

impl Debug for AVXf32x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [x0, x1, x2, x3, x4, x5, x6, x7] = self.unpack();
        write!(f, "[avx: {x0}, {x1}, {x2}, {x3}, {x4}, {x5}, {x6}, {x7}]")
    }
}

impl NumPar for AVXf32x8 {}

#[test]
fn avxf32x8_shape() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(a, AVXf32x8::from(a).unpack());
}
#[test]
fn avxf32x8_ops() {
    let a_raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = AVXf32x8::from(a_raw);
    assert_eq!([1.0; 8], (a / a).unpack());
    assert_eq!(
        a_raw.map(|v| v * (v - 1.0) + 2.0),
        (a * (a - 1.0) + 2.0).unpack()
    );
    assert_eq!(
        a_raw.map(|v| ((v % 2.25) * 1000.0).floor()),
        ((a % 2.25) * 1000.0).floor().unpack()
    );
}
