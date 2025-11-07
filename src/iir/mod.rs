// adapted from https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_filter_design.py

use crate::NFloat;

pub enum IIRFilterType {
    LowPass { cutoff: f64 },
    HighPass { cutoff: f64 },
    BandPass { low_cutoff: f64, high_cutoff: f64 },
}

#[derive(Debug, Clone, Copy)]
enum ButterZPKError {
    CuttoffOutOfBounds,
}

fn butter_zpk<const N: usize>(
    fs: f64,
    ty: IIRFilterType,
    order: usize,
) -> Result<ZPK<N>, ButterZPKError> {
    let warp = |cutoff: f64| {
        let cutoff = cutoff * 2.0 / fs;
        if cutoff <= 0.0 || cutoff >= 1.0 {
            return Err(ButterZPKError::CuttoffOutOfBounds);
        }
        Ok(4.0 * (std::f64::consts::FRAC_PI_2 * cutoff).tan())
    };
    let zpk = butter_prototype::<N>(order);
    let zpk = match ty {
        IIRFilterType::LowPass { cutoff } => prototype_lowpass_zpk(zpk, warp(cutoff)?),
        IIRFilterType::HighPass { cutoff } => prototype_highpass_zpk(zpk, warp(cutoff)?),
        IIRFilterType::BandPass {
            low_cutoff,
            high_cutoff,
        } => {
            let low = warp(low_cutoff)?;
            let high = warp(high_cutoff)?;
            let warped_bandwidth = (high - low).abs();
            let warped_cutoff = (low * high).sqrt();
            prototype_bandpass_zpk(zpk, warped_cutoff, warped_bandwidth)
        }
    };
    let zpk = bilinear_zpk(zpk, 2.0);
    Ok(zpk)
}

/// no idea what this does but scipy has it /shrug
fn bilinear_zpk<const N: usize>((mut zeros, mut poles, mut gain): ZPK<N>, fs: f64) -> ZPK<N> {
    let degree = poles.len() - zeros.len();

    let fs2 = fs * 2.0;

    gain *= (zeros.iter().map(|z| fs2 - z).product::<num::Complex<f64>>()
        / poles.iter().map(|p| fs2 - p).product::<num::Complex<f64>>())
    .re;
    zeros.iter_mut().for_each(|z| *z = (fs2 + *z) / (fs2 - *z));
    poles.iter_mut().for_each(|p| *p = (fs2 + *p) / (fs2 - *p));
    zeros.extend((0..degree).map(|_| -num::Complex::ONE));

    (zeros, poles, gain)
}

pub struct IIRFilterBA<const N: usize> {
    bak: BAKF<N>,
    x: heapless::Deque<NFloat, N>,
    y: heapless::Deque<NFloat, N>,
}
impl<const N: usize> IIRFilterBA<N> {
    pub fn effective_order(&self) -> u32 {
        self.bak.0.len().max(self.bak.1.len()) as u32
    }
    pub fn new() -> Self {
        Self {
            bak: (heapless::Vec::new(), heapless::Vec::new(), 1.0), // H(z) = 1
            x: heapless::Deque::new(),
            y: heapless::Deque::new(),
        }
    }
    pub fn set_filter(&mut self, sample_rate: f64, ty: IIRFilterType, order: usize) {
        self.bak = bak_to_nfloat(zpk_to_bak(butter_zpk(sample_rate, ty, order).unwrap()));
    }
    pub fn with_filter(mut self, sample_rate: f64, ty: IIRFilterType, order: usize) -> Self {
        self.set_filter(sample_rate, ty, order);
        self
    }
    /// Filter process x is input, y is prior outputs. Most recent samples at front.
    pub fn process(&mut self, x_in: NFloat) -> NFloat {
        let x = &mut self.x;
        let y = &mut self.y;

        debug_assert_eq!(x.len(), y.len());
        // iter from front(most recent) to back
        let mut y_out = x_in;
        y_out += x
            .iter()
            .zip(self.bak.0.iter())
            // .zip(self.bak.1.iter())
            // .zip(std::iter::repeat_n(&1.0, 1).chain(self.bak.0.iter()))
            .map(|(x, a)| *x * a)
            .sum::<NFloat>();
        y_out *= self.bak.2;
        y_out -= y
            .iter()
            .zip(self.bak.1.iter())
            // .zip(self.bak.0.iter())
            // .zip(std::iter::repeat_n(&1.0, 1).chain(self.bak.1.iter()))
            .map(|(y, b)| *y * b)
            .sum::<NFloat>();

        unsafe {
            if x.len() == N {
                x.pop_back_unchecked();
                y.pop_back_unchecked();
            }
            x.push_front_unchecked(x_in);
            y.push_front_unchecked(y_out);
        }

        y_out
    }
}

#[test]
fn test_butter_zpk() {
    // let (z, p, k) = butter_zpk::<5>(44100.0, IIRFilterType::HighPass { cutoff: 500.0 }, 4).unwrap();
    // dbg!((z, p, k));
    let (z, p, k) = butter_zpk::<5>(
        44100.0,
        IIRFilterType::BandPass {
            low_cutoff: 817.0,
            high_cutoff: 999.0,
        },
        2,
    )
    .unwrap();
    // dbg!((z, p, k));
    dbg!(zpk_to_bak((z, p, k)));
    // let (z, p, k) = butter_prototype::<5>(1);
    // dbg!((&z, &p, &k));
    // let (z, p, k) = prototype_lowpass_zpk((z, p, k), 0.5);
    // dbg!((&z, &p, &k));
}
#[test]
fn test_impulse() {
    let mut thing =
        IIRFilterBA::<10>::new().with_filter(20.0, IIRFilterType::HighPass { cutoff: 4.0 }, 1);

    let x = Vec::from_iter(std::iter::repeat_n(0.0, 10).chain(std::iter::repeat_n(1.0, 30)));
    let y: Vec<_> = x.iter().map(|x| thing.process(*x)).collect();
    dbg!(y);
}

fn zpk_to_bak<const N: usize>((zeros, poles, gain): ZPK<N>) -> BAK<N> {
    let mut bak: BAK<N> = (
        poly_from_zeros(zeros).into_iter().map(|v| v.re).collect(),
        poly_from_zeros(poles).into_iter().map(|v| v.re).collect(),
        gain,
    );
    // reorder so highest order terms first
    bak.0.reverse();
    bak.1.reverse();
    bak
}
fn bak_to_nfloat<const N: usize>(bak: BAK<N>) -> BAKF<N> {
    (
        bak.0.into_iter().map(|b| b as NFloat).collect(),
        bak.1.into_iter().map(|a| a as NFloat).collect(),
        bak.2 as NFloat,
    )
}

/// returns [c_0,c_1,c_2,...] in $c_0 + c_1 x + c_2 x^2 + ...$ omitting the leading term, which would always be 1.
fn poly_from_zeros<const N: usize>(
    zeros: heapless::Vec<num::Complex<f64>, N>,
) -> heapless::Vec<num::Complex<f64>, N> {
    let mut out = heapless::Vec::<num::Complex<f64>, N>::new();
    for zero in zeros {
        let mut prev = num::Complex::ZERO;
        out.push(num::Complex::ONE).unwrap(); // no need to check, if it fit in `zeros` it'll fit in `out`.
        for c in &mut out {
            let now = *c;
            *c = prev - now * zero;
            prev = now;
        }
    }
    out
}

type ZPK<const N: usize> = (
    heapless::Vec<num::Complex<f64>, N>,
    heapless::Vec<num::Complex<f64>, N>,
    f64,
);
type BAK<const N: usize> = (heapless::Vec<f64, N>, heapless::Vec<f64, N>, f64);
type BAKF<const N: usize> = (heapless::Vec<NFloat, N>, heapless::Vec<NFloat, N>, NFloat);

/// Butterworth filter prototype.
///
/// z = [], p = [...; 2O-1], k = 1
fn butter_prototype<const N: usize>(order: usize) -> ZPK<N> {
    let mut poles = heapless::Vec::new();
    for i in ((1 - order as i32)..order as i32).step_by(2) {
        poles
            .push(
                -num::Complex {
                    re: 0.0,
                    im: i as f64 * std::f64::consts::FRAC_PI_2 / order as f64,
                }
                .exp(),
            )
            .expect("N must be at least {order * 2 - 1}");
    }

    (heapless::Vec::new(), poles, 1.0)
}

fn prototype_lowpass_zpk<const N: usize>(
    (mut zeros, mut poles, mut gain): ZPK<N>,
    warped_cutoff: f64,
) -> ZPK<N> {
    let degree = poles.len() - zeros.len();

    zeros.iter_mut().for_each(|z| *z *= warped_cutoff);
    poles.iter_mut().for_each(|p| *p *= warped_cutoff);
    gain *= warped_cutoff.powi(degree as i32);

    (zeros, poles, gain)
}

fn prototype_highpass_zpk<const N: usize>(
    (mut zeros, mut poles, mut gain): ZPK<N>,
    warped_cutoff: f64,
) -> ZPK<N> {
    let degree = poles.len() - zeros.len();

    gain *= (zeros.iter().map(|z| -z).product::<num::Complex<f64>>()
        / poles.iter().map(|p| -p).product::<num::Complex<f64>>())
    .re;
    zeros.iter_mut().for_each(|z| *z = warped_cutoff / *z);
    poles.iter_mut().for_each(|p| *p = warped_cutoff / *p);
    zeros.extend(std::iter::repeat_n(num::Complex::ZERO, degree));

    (zeros, poles, gain)
}
fn prototype_bandpass_zpk<const N: usize>(
    (mut zeros, mut poles, mut gain): ZPK<N>,
    warped_cutoff: f64,
    warped_bandwidth: f64,
) -> ZPK<N> {
    let degree = poles.len() - zeros.len();
    let (mut zeros_bp, mut poles_bp, _): ZPK<N> = (heapless::Vec::new(), heapless::Vec::new(), 0.0);

    zeros.iter_mut().for_each(|z| *z *= warped_bandwidth * 0.5);
    poles.iter_mut().for_each(|p| *p *= warped_bandwidth * 0.5);

    const N_TOO_SMALL_MSG: &str = "N must be at least {2 * max(zeros.len, poles.len)}";

    let warped_cutoff_sqr = warped_cutoff * warped_cutoff;
    for zero in zeros {
        let d = (zero * zero - warped_cutoff_sqr).sqrt();
        zeros_bp.push(zero + d).expect(N_TOO_SMALL_MSG);
        zeros_bp.push(zero - d).expect(N_TOO_SMALL_MSG);
    }
    for pole in poles {
        let d = (pole * pole - warped_cutoff_sqr).sqrt();
        poles_bp.push(pole + d).expect(N_TOO_SMALL_MSG);
        poles_bp.push(pole - d).expect(N_TOO_SMALL_MSG);
    }

    zeros_bp.extend(std::iter::repeat_n(num::Complex::ZERO, degree));

    gain *= warped_bandwidth.powi(degree as i32);

    (zeros_bp, poles_bp, gain)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_poly_from_zeros() {
        let s = poly_from_zeros(heapless::Vec::<_, 2>::from_iter(
            [1.0f64, 8.0].map(num::Complex::from).into_iter(),
        ));
        assert_eq!(s[0], num::Complex::from(8.0)); // x^0 term
        assert_eq!(s[1], num::Complex::from(-9.0)); // x^1 term
        let s = poly_from_zeros(heapless::Vec::<_, 2>::from_iter(
            [2.0f64, 6.0].map(num::Complex::from).into_iter(),
        ));
        assert_eq!(s[0], num::Complex::from(12.0)); // x^0 term
        assert_eq!(s[1], num::Complex::from(-8.0)); // x^1 term
    }
}
