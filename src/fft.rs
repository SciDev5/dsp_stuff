use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

use crate::{NComplex, NFloat};

pub struct FFTPair {
    fwd: Arc<dyn Fft<NFloat>>,
    inv: Arc<dyn Fft<NFloat>>,
    scratch: Vec<NComplex>,
}
impl FFTPair {
    pub fn new_alloc(planner: &mut FftPlanner<NFloat>, size: usize) -> Self {
        let fwd = planner.plan_fft_forward(size);
        let inv = planner.plan_fft_inverse(size);
        Self {
            scratch: Vec::from_iter(std::iter::repeat_n(NComplex::ZERO, size)),
            fwd,
            inv,
        }
    }
    pub fn fwd(&mut self, buffer: &mut [NComplex]) {
        self.fwd.process_with_scratch(buffer, &mut self.scratch);
    }
    pub fn inv(&mut self, buffer: &mut [NComplex]) {
        self.inv_raw(buffer);
        let n = buffer.len() as NFloat;
        buffer.iter_mut().for_each(|v| *v /= n);
    }
    pub fn inv_raw(&mut self, buffer: &mut [NComplex]) {
        self.inv.process_with_scratch(buffer, &mut self.scratch);
    }
}
