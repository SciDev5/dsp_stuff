use dsp_stuff::NFloat;
use dsp_stuff::conv::ConvHybridDynamic;
use dsp_stuff::wavetable::DbgWavetable;
use nih_plug::{params::persist::PersistentField, prelude::*};
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A test tone generator that can either generate a sine wave based on the plugin's parameters or
/// based on the current MIDI input.
pub struct DynamiFIR {
    params: Arc<DynamiFIRParams>,
    conv: [ConvHybridDynamic<DbgWavetable>; 2],
    prev: [f32; 2],

    sample_rate: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SavedSample {
    data: Vec<NFloat>,
}

impl<'a> PersistentField<'a, SavedSample> for Arc<SavedSample> {
    fn set(&self, new_value: SavedSample) {
        let s = unsafe {
            (self.as_ref() as *const SavedSample as *mut SavedSample)
                .as_mut()
                .unwrap_unchecked()
        };
        s.data.copy_from_slice(&new_value.data);
    }
    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&SavedSample) -> R,
    {
        f(self)
    }
}

#[derive(Params)]
struct DynamiFIRParams {
    #[persist = "sample"]
    v: Arc<SavedSample>,

    #[id = "wt_pos"]
    pub wt_pos: FloatParam,
    #[id = "fb"]
    pub fb: FloatParam,
}

impl Default for DynamiFIR {
    fn default() -> Self {
        Self {
            params: Arc::new(DynamiFIRParams::default()),
            sample_rate: 1.0,
            conv: [
                ConvHybridDynamic::new_alloc(&mut FftPlanner::new(), DbgWavetable, || 0.0),
                ConvHybridDynamic::new_alloc(&mut FftPlanner::new(), DbgWavetable, || 0.0),
            ],
            prev: [0.0; 2],
        }
    }
}

impl Default for DynamiFIRParams {
    fn default() -> Self {
        Self {
            v: Arc::new(SavedSample {
                data: Vec::from([1.0, 2.0, 3.0]),
            }),
            wt_pos: FloatParam::new(
                "Wavetable Pos",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),
            fb: FloatParam::new(
                "Feedback",
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: 0.99,
                },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),
        }
    }
}

// struct DynamiFIREditor {}
// impl Editor for DynamiFIREditor {
//     fn spawn(
//         &self,
//         parent: ParentWindowHandle,
//         context: Arc<dyn GuiContext>,
//     ) -> Box<dyn std::any::Any + Send> {
//     }

//     fn size(&self) -> (u32, u32) {
//         todo!()
//     }

//     fn set_scale_factor(&self, factor: f32) -> bool {
//         todo!()
//     }

//     fn param_value_changed(&self, id: &str, normalized_value: f32) {
//         todo!()
//     }

//     fn param_modulation_changed(&self, id: &str, modulation_offset: f32) {
//         todo!()
//     }

//     fn param_values_changed(&self) {
//         todo!()
//     }
// }

impl Plugin for DynamiFIR {
    const NAME: &'static str = "DynamiFIR";
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
        self.sample_rate = buffer_config.sample_rate;

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
        for (_sample_id, channel_samples) in buffer.iter_samples().enumerate() {
            // Smoothing is optionally built into the parameters themselves
            let wt_pos = self.params.wt_pos.smoothed.next();
            let fb = self.params.fb.smoothed.next();

            for (i, sample) in channel_samples.into_iter().enumerate() {
                *sample = self.conv[i].process(
                    // (*sample + (fb * 0.5) * (self.prev[i] - *sample)) as NFloat,
                    (*sample + fb * self.prev[i]) as NFloat,
                    wt_pos as NFloat,
                ) as f32;
                // self.prev[i] = (*sample).tanh();
                self.prev[i] = (*sample).clamp(-1.0, 1.0);
            }
        }

        ProcessStatus::KeepAlive
    }
}

impl ClapPlugin for DynamiFIR {
    const CLAP_ID: &'static str = "me.scidev5.dynamifir";
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
impl Vst3Plugin for DynamiFIR {
    const VST3_CLASS_ID: [u8; 16] = *b"SD5_DynamiFIR___";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Reverb,
        Vst3SubCategory::Stereo,
        Vst3SubCategory::Mono,
    ];
}

nih_export_clap!(DynamiFIR);
nih_export_vst3!(DynamiFIR);
