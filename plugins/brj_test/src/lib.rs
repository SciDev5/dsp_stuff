use dsp_stuff::binrepitchjoin::BinRepitchJoinConfig;
use dsp_stuff::{NFloat, binrepitchjoin::BinRepitchJoin};
use nih_plug::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A test tone generator that can either generate a sine wave based on the plugin's parameters or
/// based on the current MIDI input.
pub struct BRJTest {
    params: Arc<BRJTestParams>,
    brj: Option<[BinRepitchJoin; 2]>,
    pcache: (f32, f32, i32),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SavedSample {
    data: Vec<NFloat>,
}

#[derive(Params)]
struct BRJTestParams {
    #[id = "bw"]
    pub semitones_bin_width: FloatParam,
    #[id = "test_param_majory"]
    pub test_majory: FloatParam,
    #[id = "test_param_minory"]
    pub test_minory: FloatParam,
    #[id = "test_param_keyzero"]
    pub test_keyzero: IntParam,
}

impl Default for BRJTest {
    fn default() -> Self {
        Self {
            params: Arc::new(BRJTestParams::default()),
            brj: None,
            pcache: (f32::NAN, f32::NAN, -1),
        }
    }
}

impl Default for BRJTestParams {
    fn default() -> Self {
        Self {
            semitones_bin_width: FloatParam::new(
                "bin width",
                0.0,
                FloatRange::Linear {
                    min: 0.25,
                    max: 1.0,
                },
            )
            .with_unit("st"),
            test_majory: FloatParam::new("majory", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            test_minory: FloatParam::new("minory", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 }),
            test_keyzero: IntParam::new("key base", 0, IntRange::Linear { min: 0, max: 12 }),
        }
    }
}

impl Plugin for BRJTest {
    const NAME: &'static str = "BRJTest";
    const VENDOR: &'static str = "SciDev5";
    const URL: &'static str = "https://github.com/scidev5/dsp_stuff";
    const EMAIL: &'static str = "";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            // This is also the default and can be omitted here
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let config = BinRepitchJoinConfig {
            sample_rate: buffer_config.sample_rate as f64,
            zero_freq: 440.0,
            log2_bin_width: 0.75 / 12.0,
            order_band_in: 2,
            order_band_out: 1,
            order_repitch_lowpass: 4,
        };

        self.brj = Some([(); 2].map(|_| BinRepitchJoin::new(config, 72, |_| (0.0, 0.0, None))));

        true
    }

    // fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {}

    fn reset(&mut self) {}

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if let Some(brj) = self.brj.as_mut() {
            let semitones_bin_width = self.params.semitones_bin_width.value();
            let test_majory = self.params.test_majory.value();
            let test_minory = self.params.test_minory.value();
            let test_keyzero = self.params.test_keyzero.value();
            for i in 0..buffer.channels() {
                let new_config = BinRepitchJoinConfig {
                    log2_bin_width: semitones_bin_width as f64 / 12.0,
                    ..brj[i].config
                };
                if self.pcache != (test_majory, test_minory, test_keyzero) {
                    brj[i].set_config_retune(
                        |i| {
                            let j_maj = ([0, 0, 2, 2, 4, 4, 7, 7, 7, 9, 9, 11]
                                [(i.saturating_add_signed(test_keyzero as isize)) % 12]
                                - (i as i32 + test_keyzero) % 12)
                                as f64
                                / 12.0;
                            let j_min = ([0, 0, 2, 2, 3, 3, 7, 7, 7, 9, 9, 10]
                                [(i.saturating_add_signed(test_keyzero as isize)) % 12]
                                - (i as i32 + test_keyzero) % 12)
                                as f64
                                / 12.0;
                            let i = i as f64 / 12.0 - 2.0;
                            let j = i + j_maj * test_majory as f64 + j_min * test_minory as f64;
                            (i, j, Some(1.0))
                        },
                        new_config,
                    );
                } else {
                    brj[i].set_config(new_config);
                }
            }
            self.pcache = (test_majory, test_minory, test_keyzero);
            for channel_samples in buffer.iter_samples() {
                for (i, sample) in channel_samples.into_iter().enumerate() {
                    *sample = brj[i].process(*sample as NFloat) as f32;
                }
            }
        }

        ProcessStatus::KeepAlive
    }
}

impl ClapPlugin for BRJTest {
    const CLAP_ID: &'static str = "me.scidev5.brjtest";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Hybrid convolution, but the impulse response is a wavetable");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Reverb,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}
impl Vst3Plugin for BRJTest {
    const VST3_CLASS_ID: [u8; 16] = *b"SD5_BRJTest_____";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Reverb,
        Vst3SubCategory::Stereo,
        Vst3SubCategory::Mono,
    ];
}

nih_export_clap!(BRJTest);
nih_export_vst3!(BRJTest);
