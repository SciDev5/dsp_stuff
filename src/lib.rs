use num::Complex;

mod conv;
mod fft;

fn main() {
    println!("Hello, world!");
}

pub type NFloat = f64;
pub type NComplex = Complex<NFloat>;

#[cfg(debug_assertions)]
#[global_allocator]
static A: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;

#[test]
fn fft_alloc() {
    use num::complex::Complex32;
    use rustfft::FftPlanner;
    let mut s = Vec::from_iter(std::iter::repeat_n(Complex32::ONE, 1024));
    let mut scratch = Vec::from_iter(std::iter::repeat_n(Complex32::ZERO, 1024));
    let mut p = FftPlanner::<f32>::new();
    let a = p.plan_fft_forward(1024);
    let b = p.plan_fft_inverse(1024);
    assert_no_alloc::assert_no_alloc(|| {
        a.process_with_scratch(&mut s, &mut scratch);
        b.process_with_scratch(&mut s, &mut scratch);
    });
    dbg!(s);
}
