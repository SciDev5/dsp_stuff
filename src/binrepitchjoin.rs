use crate::{
    NFloat,
    iir::{IIRFilterBA, IIRFilterBABand1, IIRFilterBABand2, IIRFilterBALowHigh3, IIRFilterType},
};

#[test]
fn brj_run() {
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

    // let mut iirfba = IIRFilterBA::<10>::new(
    //     config.sample_rate.0 as f64,
    //     crate::iir::IIRFilterType::BandPass {
    //         low_cutoff: 450.0,
    //         high_cutoff: 430.0,
    //     },
    //     3,
    // );

    // let mut brj = BinRepitchJoin::new(
    //     BinRepitchJoinConfig {
    //         sample_rate: config.sample_rate.0 as f64,
    //         zero_freq: 440.0,
    //         log2_bin_width: 0.75 / 12.0,
    //         order_band_in: 2,
    //         order_band_out: 1,
    //         order_repitch_lowpass: 4,
    //     },
    //     // (-36..60).map(|i| i as f64 / 12.0).collect(),
    //     // (-48..60).map(|i| i as f64 / 12.0).collect(),
    //     (-12..36).map(|i| i as f64 / 12.0).collect(),
    // );
    // brj.set_repitch(|mapping| {
    //     for i in 0..mapping.len() {
    //         // mapping[i] = [0, 0, 2, 2, 4, 4, 7, 7, 7, 9, 9, 11][i % 12] + i / 12 * 12;
    //         mapping[i] = [0, 0, 2, 2, 3, 3, 7, 7, 7, 9, 9, 10][i % 12] + i / 12 * 12;
    //         // mapping[i] = [0, 0, 4, 4, 4, 4, 9, 9, 9, 9, 9, 0][i % 12] + i / 12 * 12;
    //     }
    // });
    let mut brj = BinRepitchJoin::new(
        BinRepitchJoinConfig {
            sample_rate: config.sample_rate.0 as f64,
            zero_freq: 440.0,
            log2_bin_width: 0.75 / 12.0,
            order_band_in: 2,
            order_band_out: 1,
            order_repitch_lowpass: 4,
        },
        60,
        |i| {
            // let j = [0, 0, 2, 2, 4, 4, 7, 7, 7, 9, 9, 11][i % 12] + i / 12 * 12;
            let j = [0, 0, 2, 2, 3, 3, 7, 7, 7, 9, 9, 10][i % 12] + i / 12 * 12;
            // let j = [0, 0, 4, 4, 4, 4, 9, 9, 9, 9, 9, 0][i % 12] + i / 12 * 12;
            (
                i as f64 / 12.0 - 2.0,
                j as f64 / 12.0 - 2.0,
                Some(1.0 as NFloat),
            )
        },
    );

    let _stream = device.build_output_stream(
        &config,
        {
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for i in 0..data.len() / 2 {
                    let mut v: NFloat = rand::random_range(-1.0..1.0);
                    // v = biquad2_l.run(v);
                    // v = biquad2_h.run(v);
                    // v = biquad2_b.run(v);
                    // v = biquad2_b2.run(v);
                    // v = dft2.filter(v);
                    v = brj.process(v);
                    // v = iirfba.process(v);
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
    band_in: IIRFilterBABand2,  // order 2 bandpass
    band_out: IIRFilterBABand1, // order 1 bandpass
    log2_band_center: f64,
    log2_band_center_to: f64,
    am_phase: (NFloat, NFloat),
    am_phase_step: (NFloat, NFloat),
    gain: Option<NFloat>,
    repitch_lowpass: IIRFilterBALowHigh3, // order 3 lowpass
}
impl BinState {
    fn update_filters(
        &mut self,
        log2_band_center: f64,
        log2_band_center_to: f64,
        config: &BinRepitchJoinConfig,
    ) {
        let (low_cutoff, high_cutoff) = (
            (log2_band_center - config.log2_bin_width * 0.5).exp2() * config.zero_freq,
            (log2_band_center + config.log2_bin_width * 0.5).exp2() * config.zero_freq,
        );
        // config.order_band_in;
        self.band_in.set_filter(
            config.sample_rate,
            IIRFilterType::BandPass {
                low_cutoff,
                high_cutoff,
            },
            2,
        );
        let (low_cutoff, high_cutoff) = (
            (log2_band_center_to - config.log2_bin_width * 0.5).exp2() * config.zero_freq,
            (log2_band_center_to + config.log2_bin_width * 0.5).exp2() * config.zero_freq,
        );
        // config.order_band_out;
        // config.order_repitch_lowpass;
        self.band_out.set_filter(
            config.sample_rate,
            IIRFilterType::BandPass {
                low_cutoff,
                high_cutoff,
            },
            1,
        );
        self.repitch_lowpass.set_filter(
            config.sample_rate,
            IIRFilterType::LowPass {
                cutoff: high_cutoff,
            },
            3,
        );
    }
    fn gen_am_frequencies(
        log2_band_center_from: NFloat,
        log2_band_center_to: NFloat,
        config: &BinRepitchJoinConfig,
    ) -> (NFloat, NFloat) {
        let log2_f_off = log2_band_center_to - log2_band_center_from;
        let freq_1 =
            (1.0 + log2_band_center_from + 0.5 * log2_f_off).exp2() * config.zero_freq as NFloat;
        let freq_2 = (1.0 + log2_band_center_from + log2_f_off).exp2() * config.zero_freq as NFloat;
        (
            freq_1 / config.sample_rate as NFloat,
            freq_2 / config.sample_rate as NFloat,
        )
    }
    fn new(
        log2_band_center: f64,
        log2_band_center_to: f64,
        gain: Option<NFloat>,
        config: &BinRepitchJoinConfig,
    ) -> Self {
        let mut s = Self {
            log2_band_center,
            log2_band_center_to,
            am_phase: (0.0, 0.0),
            am_phase_step: Self::gen_am_frequencies(
                log2_band_center as NFloat,
                log2_band_center_to as NFloat,
                config,
            ),
            gain,
            band_in: IIRFilterBA::new(),
            band_out: IIRFilterBA::new(),
            repitch_lowpass: IIRFilterBA::new(),
        };
        s.update_filter_shape(config);
        s
    }
    fn update_filter_shape(&mut self, config: &BinRepitchJoinConfig) {
        if self.gain.is_some() {
            self.update_filters(self.log2_band_center, self.log2_band_center_to, config);
        }
    }
    fn retune(
        &mut self,
        log2_band_center: f64,
        log2_band_center_to: f64,
        gain: Option<NFloat>,
        config: &BinRepitchJoinConfig,
    ) {
        self.gain = gain;
        if gain.is_none() {
            return;
        }
        self.log2_band_center = log2_band_center;
        self.log2_band_center_to = log2_band_center_to;
        self.am_phase_step = Self::gen_am_frequencies(
            log2_band_center as NFloat,
            log2_band_center_to as NFloat,
            config,
        );
        self.update_filters(log2_band_center, log2_band_center_to, config);
    }
    fn sample_am(&mut self) -> (f64, f64) {
        self.am_phase.0 = (self.am_phase.0 + self.am_phase_step.0) % 1.0;
        self.am_phase.1 = (self.am_phase.1 + self.am_phase_step.1) % 1.0;
        (
            (self.am_phase.0 as f64 * std::f64::consts::TAU).sin(),
            (self.am_phase.1 as f64 * std::f64::consts::TAU).sin(),
        )
    }
}
pub struct BinRepitchJoin {
    pub config: BinRepitchJoinConfig,
    bin_states: Vec<BinState>,
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
    pub fn new(
        config: BinRepitchJoinConfig,
        n_bins: usize,
        f: impl Fn(usize) -> (f64, f64, Option<NFloat>),
    ) -> Self {
        Self {
            config,
            bin_states: (0..n_bins)
                .map(f)
                .map(|(log2_band_center, log2_band_center_to, gain)| {
                    BinState::new(log2_band_center, log2_band_center_to, gain, &config)
                })
                .collect(),
        }
    }

    pub fn process(&mut self, s: NFloat) -> NFloat {
        let s = s;

        let mut out = 0.0;
        for bin in &mut self.bin_states {
            if let Some(gain) = bin.gain {
                let s_bin = bin.band_in.process(s);
                let am = bin.sample_am();
                let s_bin_repitched =
                    bin.repitch_lowpass.process(s_bin * am.0 as NFloat) * am.1 as NFloat;
                out += bin.band_out.process(s_bin_repitched) * gain;
            }
        }

        out
    }

    pub fn set_config(&mut self, config: BinRepitchJoinConfig) {
        if config.log2_bin_width != self.config.log2_bin_width
            || config.order_band_in != self.config.order_band_in
            || config.order_band_out != self.config.order_band_out
            || config.order_repitch_lowpass != self.config.order_repitch_lowpass
            || config.sample_rate != self.config.sample_rate
            || config.zero_freq != self.config.zero_freq
        {
            for bin in &mut self.bin_states {
                bin.update_filter_shape(&config);
            }
        }
        self.config = config;
    }
    pub fn set_config_retune(
        &mut self,
        f: impl Fn(usize) -> (f64, f64, Option<NFloat>),
        config: BinRepitchJoinConfig,
    ) {
        self.config = config;
        for (i, bin) in self.bin_states.iter_mut().enumerate() {
            let (from, to, gain) = f(i);
            bin.retune(from, to, gain, &self.config);
        }
    }
}
