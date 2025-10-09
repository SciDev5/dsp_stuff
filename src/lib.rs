use num::Complex;

pub mod conv;
pub mod fft;
pub mod wavetable;
pub use nih_plug;

#[cfg(feature = "prefer_f32")]
pub type NFloat = f32;
#[cfg(not(feature = "prefer_f32"))]
pub type NFloat = f64;
pub type NComplex = Complex<NFloat>;

// #[cfg(debug_assertions)]
// #[global_allocator]
// static A: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;
