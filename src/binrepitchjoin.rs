use iir_filters::{
    filter::{DirectForm2Transposed, Filter},
    filter_design::{FilterType, butter},
    sos::{Sos, zpk2sos},
};

use crate::NFloat;

// #[test]
fn s() {
    // use biquad::*;

    // // Cutoff and sampling frequencies
    // let f0 = 10.hz();
    // let fs = 1.khz();

    // // Create coefficients for the biquads
    // let coeffs =
    //     // Coefficients::<f32>::from_params(Type::LowPass, fs, f0, Q_BUTTERWORTH_F32).unwrap();
    //     Coefficients::<f32>::from_params(Type::BandPass, fs, f0, Q_BUTTERWORTH_F32).unwrap();

    // // Create two different biquads
    // let mut biquad1 = DirectForm1::<f32>::new(coeffs);
    // let mut biquad2 = DirectForm2Transposed::<f32>::new(coeffs);

    // biquad1.update_coefficients(new_coefficients);

    // let input_vec = Vec::from_iter((0..10000).p);
    // let mut output_vec1 = Vec::new();
    // let mut output_vec2 = Vec::new();

    // // Run for all the inputs
    // for elem in input_vec {
    //     output_vec1.push(biquad1.run(elem));
    //     output_vec2.push(biquad2.run(elem));
    // }

    ///////////////////////////////////////////

    use cpal::{traits::*, *};
    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();
    let config: StreamConfig = device.default_output_config().unwrap().into();
    // let fs = config.sample_rate.0.hz();
    // let f0 = 440.hz();
    // let f1 = 460.hz();

    // let coeffs_l =
    //     Coefficients::<f32>::from_params(Type::HighPass, fs, f0, Q_BUTTERWORTH_F32).unwrap();
    // let mut biquad2_l = DirectForm2Transposed::<f32>::new(coeffs_l);
    // let coeffs_h =
    //     Coefficients::<f32>::from_params(Type::LowPass, fs, f1, Q_BUTTERWORTH_F32).unwrap();
    // let mut biquad2_h = DirectForm2Transposed::<f32>::new(coeffs_h);
    // let coeffs_b =
    //     Coefficients::<f32>::from_params(Type::BandPass, fs, f1, Q_BUTTERWORTH_F32).unwrap();
    // let mut biquad2_b = DirectForm2Transposed::<f32>::new(coeffs_b);
    // let coeffs_b2 =
    //     Coefficients::<f32>::from_params(Type::BandPass, fs, f1, Q_BUTTERWORTH_F32).unwrap();
    // let mut biquad2_b2 = DirectForm2Transposed::<f32>::new(coeffs_b2);

    // let order = 2;
    // let cutoff_low = 440.0;
    // let cutoff_hi = 560.0;
    // let fs = config.sample_rate.0 as f64;

    // let zpk = butter(order, FilterType::BandPass(cutoff_low, cutoff_hi), fs).unwrap();
    // let sos = zpk2sos(&zpk, None).unwrap();

    // let mut dft2 = DirectForm2Transposed::new(&sos);

    let mut brj = BinRepitchJoin::new_alloc(
        BinRepitchJoinConfig {
            sample_rate: config.sample_rate.0 as f64,
            zero_freq: 440.0,
            log2_bin_width: 0.75 / 12.0,
            order_band_in: 2,
            order_band_out: 1,
            order_repitch_lowpass: 4,
        },
        // (-36..60).map(|i| i as f64 / 12.0).collect(),
        // (-48..60).map(|i| i as f64 / 12.0).collect(),
        (-12..36).map(|i| i as f64 / 12.0).collect(),
    );
    brj.set_repitch(|mapping| {
        for i in 0..mapping.len() {
            // mapping[i] = [0, 0, 2, 2, 4, 4, 7, 7, 7, 9, 9, 11][i % 12] + i / 12 * 12;
            mapping[i] = [0, 0, 2, 2, 3, 3, 7, 7, 7, 9, 9, 10][i % 12] + i / 12 * 12;
            // mapping[i] = [0, 0, 4, 4, 4, 4, 9, 9, 9, 9, 9, 0][i % 12] + i / 12 * 12;
        }
    });

    let _stream = device.build_output_stream(
        &config,
        {
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for i in 0..data.len() / 2 {
                    let mut v = rand::random_range(-1.0f64..1.0);
                    // v = biquad2_l.run(v);
                    // v = biquad2_h.run(v);
                    // v = biquad2_b.run(v);
                    // v = biquad2_b2.run(v);
                    // v = dft2.filter(v);
                    v = brj.process(v);
                    data[2 * i + 0] = v.clamp(-1.0, 1.0) as f32;
                    data[2 * i + 1] = v.clamp(-1.0, 1.0) as f32;
                }
            }
        },
        move |_err| {
            // react to errors here.
        },
        None, // None=blocking, Some(Duration)=timeout
    );

    std::thread::sleep(std::time::Duration::from_secs(2));
}

struct BinState {
    band_in: DirectForm2Transposed,
    band_out: DirectForm2Transposed,
    log2_band_center: f64,
    am_phase: (f64, f64),
    am_phase_step: (f64, f64),
    repitch_lowpass: DirectForm2Transposed,
}
impl BinState {
    fn gen_filter_sos(log2_band_center: f64, config: &BinRepitchJoinConfig) -> (Sos, Sos, Sos) {
        let (cutoff_low, cutoff_hi) = (
            (log2_band_center - config.log2_bin_width * 0.5).exp2() * config.zero_freq,
            (log2_band_center + config.log2_bin_width * 0.5).exp2() * config.zero_freq,
        );
        (
            zpk2sos(
                &butter(
                    config.order_band_in,
                    FilterType::BandPass(cutoff_low, cutoff_hi),
                    config.sample_rate,
                )
                .unwrap(),
                None,
            )
            .unwrap(),
            zpk2sos(
                &butter(
                    config.order_band_out,
                    FilterType::BandPass(cutoff_low, cutoff_hi),
                    config.sample_rate,
                )
                .unwrap(),
                None,
            )
            .unwrap(),
            zpk2sos(
                &butter(
                    config.order_repitch_lowpass,
                    FilterType::LowPass(cutoff_hi),
                    config.sample_rate,
                )
                .unwrap(),
                None,
            )
            .unwrap(),
        )
    }
    fn gen_am_frequencies(
        log2_band_center_from: f64,
        log2_band_center_to: f64,
        config: &BinRepitchJoinConfig,
    ) -> (f64, f64) {
        let log2_f_off = log2_band_center_to - log2_band_center_from;
        let freq_1 = (1.0 + log2_band_center_from + 0.5 * log2_f_off).exp2() * config.zero_freq;
        let freq_2 = (1.0 + log2_band_center_from + log2_f_off).exp2() * config.zero_freq;
        (freq_1 / config.sample_rate, freq_2 / config.sample_rate)
    }
    fn new(log2_band_center: f64, config: &BinRepitchJoinConfig) -> Self {
        let filter_sos = Self::gen_filter_sos(log2_band_center, config);
        Self {
            log2_band_center,
            am_phase: (0.0, 0.0),
            am_phase_step: Self::gen_am_frequencies(log2_band_center, log2_band_center, config),
            band_in: DirectForm2Transposed::new(&filter_sos.0),
            band_out: DirectForm2Transposed::new(&filter_sos.1),
            repitch_lowpass: DirectForm2Transposed::new(&filter_sos.2),
        }
    }
    fn update_config(&mut self, log2_band_center: f64, config: &BinRepitchJoinConfig) {
        // TODO: implement buttersworth filters myself so I can actually change the parameters.
        assert_no_alloc::permit_alloc(|| {
            let filter_sos = Self::gen_filter_sos(log2_band_center, config);
            self.log2_band_center = log2_band_center;
            self.band_in = DirectForm2Transposed::new(&filter_sos.0);
            self.band_out = DirectForm2Transposed::new(&filter_sos.1);
            self.repitch_lowpass = DirectForm2Transposed::new(&filter_sos.2);
        })
    }
    fn update_repitch_mapping(&mut self, log2_band_center_to: f64, config: &BinRepitchJoinConfig) {
        self.am_phase_step =
            Self::gen_am_frequencies(self.log2_band_center, log2_band_center_to, config);
    }
    fn sample_am(&mut self) -> (f64, f64) {
        self.am_phase.0 = (self.am_phase.0 + self.am_phase_step.0) % 1.0;
        self.am_phase.1 = (self.am_phase.1 + self.am_phase_step.1) % 1.0;
        (
            (self.am_phase.0 * std::f64::consts::TAU).sin(),
            (self.am_phase.1 * std::f64::consts::TAU).sin(),
        )
    }
}
pub struct BinRepitchJoin {
    pub config: BinRepitchJoinConfig,
    repitch_mappings: Vec<usize>,
    log2_bins: Vec<f64>,
    bin_states: Vec<BinState>,
    bin_scratch: Vec<f64>,
}
#[derive(Debug, Clone, Copy)]
pub struct BinRepitchJoinConfig {
    pub sample_rate: f64,
    pub zero_freq: f64,
    pub log2_bin_width: f64,
    pub order_band_in: u32,
    pub order_band_out: u32,
    pub order_repitch_lowpass: u32,
}
impl BinRepitchJoin {
    pub fn new_alloc(config: BinRepitchJoinConfig, log2_bins: Vec<f64>) -> Self {
        let n_bins = log2_bins.len();
        Self {
            config,
            repitch_mappings: (0..n_bins).collect(),
            bin_scratch: std::iter::repeat_n(0.0, n_bins).collect(),
            bin_states: log2_bins
                .iter()
                .map(|log2_band_center| BinState::new(*log2_band_center, &config))
                .collect(),
            log2_bins,
        }
    }

    pub fn process(&mut self, s: NFloat) -> NFloat {
        let s = s as f64;
        self.bin_scratch.iter_mut().for_each(|v| *v = 0.0);

        for (i, bin) in self.bin_states.iter_mut().enumerate() {
            let s_bin = bin.band_in.filter(s);
            let am = bin.sample_am();
            let s_bin_repitched = bin.repitch_lowpass.filter(s_bin * am.0) * am.1;
            self.bin_scratch[self.repitch_mappings[i]] += s_bin_repitched;
        }
        let mut out = 0.0;
        for (bin, scratch) in self.bin_states.iter_mut().zip(self.bin_scratch.iter()) {
            out += bin.band_out.filter(*scratch);
        }

        out as NFloat
    }

    pub fn set_config(&mut self, config: BinRepitchJoinConfig) {
        if config.log2_bin_width != self.config.log2_bin_width
            || config.order_band_in != self.config.order_band_in
            || config.order_band_out != self.config.order_band_out
            || config.order_repitch_lowpass != self.config.order_repitch_lowpass
            || config.sample_rate != self.config.sample_rate
            || config.zero_freq != self.config.zero_freq
        {
            for (log2_band_center, bin) in self.log2_bins.iter().zip(self.bin_states.iter_mut()) {
                bin.update_config(*log2_band_center, &config);
            }
        }

        if config.sample_rate != self.config.sample_rate
            || config.zero_freq != self.config.zero_freq
        {
            self.set_repitch(|_| {});
        }
        self.config = config;
    }
    pub fn set_bins(&mut self, f: impl FnOnce(&mut [f64])) {
        f(&mut self.log2_bins);

        for (log2_band_center, bin) in self.log2_bins.iter().zip(self.bin_states.iter_mut()) {
            bin.update_config(*log2_band_center, &self.config);
        }
        self.set_repitch(|_| {});
    }
    pub fn set_repitch(&mut self, f: impl FnOnce(&mut [usize])) {
        let n_bins = self.bin_scratch.len();

        f(&mut self.repitch_mappings);

        for (bin, scratch) in self.bin_states.iter().zip(self.bin_scratch.iter_mut()) {
            *scratch = bin.log2_band_center;
        }
        for (bin, mapping) in self
            .bin_states
            .iter_mut()
            .zip(self.repitch_mappings.iter_mut())
        {
            if *mapping >= n_bins {
                *mapping = n_bins - 1
            }
            bin.update_repitch_mapping(self.bin_scratch[*mapping], &self.config);
        }
    }
}
