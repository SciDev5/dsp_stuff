use crate::NFloat;

pub struct DbgWavetable;
impl Wavetable for DbgWavetable {
    type Inputs = NFloat;
    fn size(&self) -> usize {
        1024
    }
    fn sample(&self, inputs: &NFloat, pos: usize) -> NFloat {
        if pos == 0 {
            1.0
        } else {
            if pos == (*inputs * self.size() as NFloat) as usize {
                1.0
            } else {
                0.0
            }
        }
        // self.sample_interp(
        //     inputs,
        //     (pos as NFloat / (self.size() - 1) as NFloat).clamp(0.0, 1.0),
        // )
    }
    fn sample_interp(&self, inputs: &NFloat, pos: NFloat) -> NFloat {
        // let a = (pos * std::f64::consts::TAU as NFloat).sin();
        // let b = (pos - pos.floor()) * 2.0 - 1.0;
        ((pos * std::f64::consts::TAU as NFloat * (1.0) * 40.0).cos()
            + (pos
                * std::f64::consts::TAU as NFloat
                * (1.5 / 1.25 * 1.001 + *inputs * (1.25 - 1.5 / 1.25))
                * 40.0)
                .cos()
            + (pos * std::f64::consts::TAU as NFloat * (1.5) * 40.0).cos()
            + (pos * std::f64::consts::TAU as NFloat * (1.5 * 1.5) * 40.0).cos())
            * (pos * -2.0).exp()
            * 0.025
        // ((*inputs * 32.0 * (0.1 + pos)).cos().clamp(-0.05, 0.05)
        //     + (*inputs * 64.0 * (0.2 + pos)).cos().clamp(-0.05, 0.05)
        //     + (*inputs * 128.0 * (0.3 + pos)).cos().clamp(-0.05, 0.05)
        //     + (*inputs * 256.0 * (0.4 + pos)).cos().clamp(-0.05, 0.05)
        //     + (*inputs * 512.0 * (0.5 + pos)).cos().clamp(-0.05, 0.05))
        //     * (-3.0 * pos).exp()
        //     * 0.1
    }
}

pub trait Wavetable: Send + Sync {
    type Inputs: Send + Sync;
    fn size(&self) -> usize;
    fn sample(&self, inputs: &Self::Inputs, pos: usize) -> NFloat;
    fn sample_interp(&self, inputs: &Self::Inputs, pos: NFloat) -> NFloat;
}

pub struct WavetableHistory<WT: Wavetable> {
    input_history: Vec<WT::Inputs>,
    off: usize,
    table: WT,
}
impl<WT: Wavetable> WavetableHistory<WT> {
    pub fn new_alloc(
        history_len: usize,
        table: WT,
        gen_initial_input: impl Fn() -> WT::Inputs,
    ) -> Self {
        Self {
            input_history: Vec::from_iter(
                std::iter::repeat_with(gen_initial_input).take(history_len),
            ),
            off: 0,
            table,
        }
    }
    pub fn step(&mut self, inputs: WT::Inputs) {
        self.off = (self.off + 1) % self.history_len();
        self.input_history[self.off] = inputs;
    }
    #[inline]
    pub fn size(&self) -> usize {
        self.table.size()
    }
    #[inline]
    pub fn history_len(&self) -> usize {
        self.input_history.len()
    }
    pub fn sample(&self, latency: usize, pos: usize) -> NFloat {
        assert!(latency < self.history_len());
        self.table.sample(
            &self.input_history[(self.off + self.history_len() - latency) % self.history_len()],
            pos,
        )
    }
}
