#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub mod avx;

pub trait NumPar: num::traits::NumOps + From<f32> {}

impl NumPar for f32 {}
