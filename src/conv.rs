//! convolution

use std::{collections::VecDeque, sync::Arc};

use rustfft::FftPlanner;

use crate::{
    NComplex, NFloat,
    fft::FFTPair,
    wavetable::{Wavetable, WavetableHistory},
};

impl FFTPair {
    /// Evaluate the circular convolution of `a` and `b` using FFT.
    pub fn conv_circular<'a>(
        &mut self,
        a: &'a mut [NComplex],
        b: &mut [NComplex],
    ) -> ConvCircularResult<'a> {
        self.fwd(b);
        self.conv_circular_b_fft(a, b)
    }
    /// Evaluate the circular convolution of `a` and `IFFT(b_fft)` using FFT.
    pub fn conv_circular_b_fft<'a>(
        &mut self,
        a: &'a mut [NComplex],
        b: &[NComplex],
    ) -> ConvCircularResult<'a> {
        self.fwd(a);
        a.iter_mut().enumerate().for_each(|(i, a_i)| *a_i *= b[i]);
        self.inv(a);
        ConvCircularResult { full: a }
    }
}
pub struct ConvCircularResult<'a> {
    pub full: &'a [NComplex],
}
impl<'a> ConvCircularResult<'a> {
    /// Extracts only the part of the signal corresponding to a linear convolution of `a` and `b`,
    /// assuming `b` is nonzero on indices `0..nonzero_b_len`.
    #[inline]
    pub fn valid_only(self, nonzero_b_len: usize) -> &'a [NComplex] {
        &self.full[nonzero_b_len - 1..]
    }
}

/// Implementation of the overlap-scrap convolution (also known as overlap-save).
pub struct ConvOverlapScrap {
    i: usize,
    size: usize,
    filter_size: usize,
    fft: FFTPair,              // fft[size]
    filter_fft: Vec<NComplex>, // [size]
    /// `(&mut filter_fft, &mut fft) => void`
    pub regen_filter: Option<Box<dyn Fn(&mut [NComplex], &mut FFTPair) + Send>>,
    buf_in_save: Vec<NComplex>, // [scrap_size]
    buf_in: Vec<NComplex>,      // [size]
    buf_out: Vec<NComplex>,     // [size] (invalid: [scrap_size], valid: [block_size])
}
impl ConvOverlapScrap {
    pub fn new_alloc(planner: &mut FftPlanner<NFloat>, size: usize, filter_size: usize) -> Self {
        Self {
            i: size - 1,
            size,
            filter_size,
            fft: FFTPair::new_alloc(planner, size),
            filter_fft: Vec::from_iter(std::iter::repeat_n(NComplex::ZERO, size)),
            regen_filter: None,
            buf_in_save: Vec::from_iter(std::iter::repeat_n(NComplex::ZERO, filter_size - 1)),
            buf_in: Vec::from_iter(std::iter::repeat_n(NComplex::ZERO, size)),
            buf_out: Vec::from_iter(std::iter::repeat_n(NComplex::ZERO, size)),
        }
    }
    pub fn with_regen_filter(
        mut self,
        regen_filter: Box<dyn Fn(&mut [NComplex], &mut FFTPair) + Send>,
    ) -> Self {
        self.regen_filter = Some(regen_filter);
        self
    }
    #[inline]
    pub const fn latency(&self) -> usize {
        self.block_size() - 1
        // self.size - self.filter_size
    }
    #[inline]
    pub const fn block_size(&self) -> usize {
        self.size - self.scrap_size()
    }
    #[inline]
    const fn scrap_size(&self) -> usize {
        self.filter_size - 1
    }
    pub fn set_filter(&mut self, buf: &[NFloat]) {
        assert_eq!(buf.len(), self.filter_size);
        self.filter_fft
            .iter_mut()
            .enumerate()
            .for_each(|(i, f)| *f = NComplex::from(buf.get(i).copied().unwrap_or(0.0)));
        self.fft.fwd(&mut self.filter_fft);
    }
    pub fn set_filter_complex(&mut self, buf: &[NComplex]) {
        assert_eq!(buf.len(), self.filter_size);
        self.filter_fft
            .iter_mut()
            .enumerate()
            .for_each(|(i, f)| *f = buf.get(i).copied().unwrap_or(NComplex::ZERO));
        self.fft.fwd(&mut self.filter_fft);
    }
    fn process_buffered_block(&mut self) {
        let block_size = self.block_size();
        self.buf_in_save.copy_from_slice(&self.buf_in[block_size..]);
        if let Some(regen_filter) = &self.regen_filter {
            regen_filter(&mut self.filter_fft, &mut self.fft);
        }
        self.fft
            .conv_circular_b_fft(&mut self.buf_in, &self.filter_fft);
        std::mem::swap(&mut self.buf_in, &mut self.buf_out);
        self.buf_in[..(self.size - block_size)].copy_from_slice(&self.buf_in_save);
    }
    #[inline]
    pub fn process(&mut self, s: NFloat) -> NFloat {
        self.buf_in[self.i] = NComplex::new(s, 0.0);
        if self.i == self.size - 1 {
            self.process_buffered_block();
            self.i = self.scrap_size() - 1;
        }
        self.i += 1;
        return self.buf_out[self.i].re;
    }
}

pub struct ConvBlock {
    filter: Vec<NFloat>,
    buf: VecDeque<NFloat>,
}
impl ConvBlock {
    pub fn new_alloc(filter: Vec<NFloat>) -> Self {
        Self {
            buf: VecDeque::from_iter(std::iter::repeat_n(0.0, filter.len())),
            filter,
        }
    }
    pub fn process(&mut self, s: NFloat) -> NFloat {
        self.buf.pop_back();
        self.buf.push_front(s);
        self.buf
            .iter()
            .zip(self.filter.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

// if latency = [offset into impulse response] => zero effective latency
pub struct ConvHybridLong {
    conv_seed: ConvBlock,
    conv_stack: Vec<ConvOverlapScrap>,
}
impl ConvHybridLong {
    pub fn new_alloc(planner: &mut FftPlanner<NFloat>, mut filter: Vec<NFloat>) -> Self {
        let mut n = 5;
        let mut filter_rem = filter.split_off(1 << n);
        let conv_seed = ConvBlock::new_alloc(filter);
        filter = filter_rem;
        let mut conv_stack = Vec::new();
        loop {
            let size = 2 << n;
            if filter.len() < (1 << n) {
                filter.extend(std::iter::repeat_n(0.0, (1 << n) - filter.len()));
            }
            filter_rem = filter.split_off(1 << n);
            let mut conv = ConvOverlapScrap::new_alloc(planner, size, 1 << n);
            conv.set_filter(&filter);
            conv_stack.push(conv);

            if filter_rem.len() == 0 {
                break;
            }
            filter = filter_rem;
            n += 1;
        }
        Self {
            conv_seed,
            conv_stack,
        }
    }
    pub fn process(&mut self, s: NFloat) -> NFloat {
        self.conv_seed.process(s)
            + self
                .conv_stack
                .iter_mut()
                .map(|conv| conv.process(s))
                .sum::<NFloat>()
    }
}

pub struct ConvHybridDynamic<WT: Wavetable> {
    conv_seed: ConvBlock,
    conv_stack: Vec<ConvOverlapScrap>,
    filter_table: Arc<WavetableHistory<WT>>,
}
impl<WT: Wavetable + 'static> ConvHybridDynamic<WT> {
    pub fn new_alloc(
        planner: &mut FftPlanner<NFloat>,
        table: WT,
        gen_initial_input: impl Fn() -> WT::Inputs,
    ) -> Self {
        let size = table.size();
        if !size.is_power_of_two() {
            panic!("ConvHybridDynamic filter size must be power of two");
        }
        let n = 5;
        let filter_table = Arc::new(WavetableHistory::new_alloc(size, table, gen_initial_input));
        // let mut lrem = size - (1 << n);
        let conv_seed = ConvBlock::new_alloc(Vec::from_iter(std::iter::repeat_n(0.0, 1 << n)));
        let mut conv_stack = Vec::new();
        let mut filter_offset = 1 << n;
        for n_ in n..size.trailing_zeros() {
            let filter_section_size = 1 << n_;
            conv_stack.push({
                let loff = filter_offset;
                let filter_table = filter_table.clone();
                let lots_of_zeros =
                    Vec::from_iter(std::iter::repeat_n(NComplex::ZERO, filter_section_size));
                let s = ConvOverlapScrap::new_alloc(
                    planner,
                    filter_section_size << 1,
                    filter_section_size,
                )
                .with_regen_filter(Box::new(move |filt, fft| {
                    let filter_table = filter_table.as_ref();
                    filt[filter_section_size..].copy_from_slice(&lots_of_zeros);
                    for i in 0..filter_section_size {
                        filt[i] = NComplex::new(filter_table.sample(i, loff as usize + i), 0.0);
                    }
                    fft.fwd(filt);
                }));
                dbg!((filter_offset, filter_section_size, s.latency()));
                s
            });
            filter_offset += filter_section_size;
        }
        Self {
            conv_seed,
            conv_stack,
            filter_table,
        }
    }
    pub fn process(&mut self, s: NFloat, inputs: WT::Inputs) -> NFloat {
        {
            let filter_table = unsafe {
                (self.filter_table.as_ref() as *const WavetableHistory<WT>
                    as *mut WavetableHistory<WT>)
                    .as_mut()
                    .unwrap_unchecked()
            };
            filter_table.step(inputs);
            for i in 0..self.conv_seed.filter.len() {
                self.conv_seed.filter[i] = filter_table.sample(i, i);
            }
        }
        self.conv_seed.process(s)
            + self
                .conv_stack
                .iter_mut()
                .map(|conv| conv.process(s))
                .sum::<NFloat>()
    }
}

#[cfg(test)]
mod test {
    use assert_no_alloc::assert_no_alloc;
    use rayon::iter::{ParallelBridge, ParallelIterator};
    use rustfft::FftPlanner;

    use crate::{
        NComplex, NFloat,
        conv::{ConvBlock, ConvHybridDynamic, ConvHybridLong, ConvOverlapScrap},
        fft::FFTPair,
        wavetable::Wavetable,
    };

    #[test]
    fn test_conv_circular() {
        const N: usize = 1024;
        let mut fft = FFTPair::new_alloc(&mut FftPlanner::new(), N);
        for _ in 0..5 {
            let mut a = Vec::from_iter(
                std::iter::from_fn(|| {
                    Some(NComplex::new(
                        rand::random::<NFloat>() * 2.0 - 1.0,
                        rand::random::<NFloat>() * 2.0 - 1.0,
                    ))
                })
                .take(N),
            );
            let mut b = Vec::from_iter(
                std::iter::from_fn(|| {
                    Some(
                        NComplex::new(
                            rand::random::<NFloat>() * 2.0 - 1.0,
                            rand::random::<NFloat>() * 2.0 - 1.0,
                        ) / N as NFloat,
                    )
                })
                .take(N),
            );
            let c_ground_truth = {
                (0..N)
                    .map(|n| {
                        (0..N)
                            .par_bridge()
                            .map(|m| a[m] * b[(n + N - m) % N])
                            .sum::<NComplex>()
                    })
                    .collect::<Vec<_>>()
            };
            let c_fast =
                assert_no_alloc::assert_no_alloc(|| fft.conv_circular(&mut a, &mut b).full);
            let rms_gt = c_ground_truth
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<NFloat>()
                .sqrt();
            let rmse = c_ground_truth
                .iter()
                .zip(c_fast.iter())
                .map(|(x, y)| (x - y).norm_sqr())
                .sum::<NFloat>()
                .sqrt();
            dbg!(rmse / rms_gt);
            assert!(rmse / rms_gt < 1e-6);
        }
    }

    fn test_conv_generic<T>(
        signal_size: usize,
        filter_size: usize,
        conv_init: impl Fn(&Vec<NFloat>) -> T,
        conv_process: impl Fn(&mut T, NFloat) -> NFloat,
        latency: impl Fn(&T) -> usize,
    ) {
        for _ in 0..5 {
            let a = Vec::from_iter(
                std::iter::from_fn(|| Some(rand::random::<NFloat>() * 2.0 - 1.0)).take(signal_size),
            );
            let b = Vec::from_iter(
                std::iter::from_fn(|| {
                    Some((rand::random::<NFloat>() * 2.0 - 1.0) / filter_size as NFloat)
                })
                .take(filter_size),
            );
            let mut conv = conv_init(&b);
            let latency = latency(&conv);
            let c_ground_truth = {
                (0..signal_size + filter_size - 1)
                    .map(|n| {
                        (0..filter_size)
                            .map(|m| {
                                b[m] * if m > n || n.saturating_sub(m) >= a.len() {
                                    0.0
                                } else {
                                    a[n - m]
                                }
                            })
                            .sum::<NFloat>()
                    })
                    .collect::<Vec<_>>()
            };
            let c_fast = a
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0))
                .map(|smp| assert_no_alloc(|| conv_process(&mut conv, smp)))
                .skip(latency)
                .take(c_ground_truth.len())
                .collect::<Vec<_>>();
            let rms_gt = c_ground_truth.iter().map(|x| x * x).sum::<NFloat>().sqrt();
            let rms_f = c_fast.iter().map(|x| x * x).sum::<NFloat>().sqrt();
            let rmse = c_ground_truth
                .iter()
                .zip(c_fast.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<NFloat>()
                .sqrt();
            dbg!(rms_f);
            dbg!(rms_f / rms_gt);
            dbg!(rmse / rms_gt);
            assert!(rmse / rms_gt < 1e-8);
            assert!((rms_f / rms_gt - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_conv_overlap_scrap() {
        test_conv_generic(
            2345,
            60,
            |b| {
                let mut conv = ConvOverlapScrap::new_alloc(&mut FftPlanner::new(), 512, b.len());
                conv.set_filter(&b);
                conv
            },
            ConvOverlapScrap::process,
            ConvOverlapScrap::latency,
        );
    }

    #[test]
    fn test_conv_block() {
        test_conv_generic(
            2345,
            32,
            |b| ConvBlock::new_alloc(b.clone()),
            ConvBlock::process,
            |_| 0,
        );
    }

    #[test]
    fn test_conv_hybrid_long() {
        test_conv_generic(
            2345,
            500,
            |b| ConvHybridLong::new_alloc(&mut FftPlanner::new(), b.clone()),
            ConvHybridLong::process,
            |_| 0,
        );
    }

    struct WavetableConst {
        data: Vec<NFloat>,
    }
    impl Wavetable for WavetableConst {
        type Inputs = ();
        fn size(&self) -> usize {
            self.data.len()
        }
        fn sample_interp(&self, _inputs: &Self::Inputs, _pos: NFloat) -> NFloat {
            panic!("do not use")
        }
        fn sample(&self, _inputs: &Self::Inputs, pos: usize) -> NFloat {
            self.data[pos]
        }
    }
    #[test]
    fn test_conv_hybrid_dyn() {
        test_conv_generic(
            2345,
            1024,
            |b| {
                ConvHybridDynamic::new_alloc(
                    &mut FftPlanner::new(),
                    WavetableConst { data: b.clone() },
                    || (),
                )
            },
            |conv, s| conv.process(s, ()),
            |_| 0,
        );
    }
}
