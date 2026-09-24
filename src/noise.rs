//! Noise distribution chi for the LWE encodings.

use nalgebra as na;
use rand::Rng;

use crate::ring::Ring;

/// The discrete Gaussian is truncated at TAILCUT standard deviations,
/// so chi is B-bounded with B = ceil(TAILCUT * stddev) as required by the paper.
pub const TAILCUT: f64 = 6.0;

/// The noise distribution chi, which must be B-bounded with B = `bound()`.
/// The default is TUniform(0), i.e., {-1, 0, 1} with probabilities (1/4, 1/2, 1/4):
/// it is exactly bounded with B = 1, so it needs the smallest q for the same security,
/// see the README.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Noise {
    Zero,
    /// Discrete Gaussian with standard deviation `stddev`, truncated at TAILCUT * stddev.
    DiscreteGaussian {
        stddev: f64,
    },
    /// Uniform on [-2^log_bound, 2^log_bound], where the two end points have half the
    /// probability of the other points, i.e., `ND.TUniform(log_bound)` in the lattice estimator.
    TUniform {
        log_bound: u32,
    },
}

impl Default for Noise {
    fn default() -> Self {
        Noise::TUniform { log_bound: 0 }
    }
}

impl Noise {
    /// Bound B on the absolute value of a sample.
    pub fn bound(&self) -> u64 {
        match self {
            Noise::Zero => 0,
            Noise::DiscreteGaussian { stddev } => (TAILCUT * stddev).ceil() as u64,
            Noise::TUniform { log_bound } => 1 << log_bound,
        }
    }

    /// Standard deviation of the (untruncated) distribution.
    pub fn stddev(&self) -> f64 {
        match self {
            Noise::Zero => 0.0,
            Noise::DiscreteGaussian { stddev } => *stddev,
            Noise::TUniform { log_bound } => {
                ((2f64.powi(2 * *log_bound as i32 + 1) + 1.0) / 6.0).sqrt()
            }
        }
    }

    pub fn sampler(&self) -> NoiseSampler {
        match self {
            Noise::Zero => NoiseSampler::Zero,
            Noise::DiscreteGaussian { stddev } => {
                NoiseSampler::DiscreteGaussian(DiscreteGaussian::new(*stddev))
            }
            Noise::TUniform { log_bound } => {
                assert!(*log_bound < 62);
                NoiseSampler::TUniform {
                    log_bound: *log_bound,
                }
            }
        }
    }
}

/// NOTE: the samplers are not constant time.
pub enum NoiseSampler {
    Zero,
    DiscreteGaussian(DiscreteGaussian),
    TUniform { log_bound: u32 },
}

impl NoiseSampler {
    pub fn sample_i64<Rn: Rng + ?Sized>(&self, rng: &mut Rn) -> i64 {
        match self {
            NoiseSampler::Zero => 0,
            NoiseSampler::DiscreteGaussian(chi) => chi.sample_i64(rng),
            NoiseSampler::TUniform { log_bound } => {
                // u - 2^b + c for u uniform in [0, 2^{b+1}) and a uniform bit c
                // gives the end points with probability 2^{-(b+2)}
                // and the other points with probability 2^{-(b+1)}
                let u = rng.random_range(0..1i64 << (log_bound + 1));
                u - (1i64 << log_bound) + rng.random::<bool>() as i64
            }
        }
    }

    pub fn sample<R: Ring, Rn: Rng + ?Sized>(&self, rng: &mut Rn) -> R {
        let x = self.sample_i64(rng);
        let abs = R::from_u64(x.unsigned_abs());
        if x < 0 { -abs } else { abs }
    }

    /// mat += E where E is sampled entry-wise
    pub fn add_to<R: Ring, Rn: Rng + ?Sized>(&self, mat: &mut na::DMatrix<R>, rng: &mut Rn) {
        if matches!(self, NoiseSampler::Zero) {
            return;
        }
        mat.iter_mut().for_each(|x| *x += self.sample(rng));
    }
}

/// Discrete Gaussian over Z with standard deviation `stddev`,
/// truncated to [-B, B], sampled with a cumulative distribution table.
/// A standard deviation of 0 gives the zero distribution.
pub struct DiscreteGaussian {
    bound: u64,
    // cdt[i] = 2^64 * Pr[X <= i - bound]
    cdt: Vec<u64>,
}

impl DiscreteGaussian {
    pub fn new(stddev: f64) -> Self {
        assert!(stddev >= 0.0);
        let bound = Noise::DiscreteGaussian { stddev }.bound();
        if bound == 0 {
            return Self {
                bound,
                cdt: vec![u64::MAX],
            };
        }

        let weights: Vec<f64> = (-(bound as i64)..=bound as i64)
            .map(|x| (-((x * x) as f64) / (2.0 * stddev * stddev)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut acc = 0.0;
        let mut cdt: Vec<u64> = weights
            .iter()
            .map(|w| {
                acc += w;
                // the cast saturates
                (acc / total * 2f64.powi(64)) as u64
            })
            .collect();
        *cdt.last_mut().unwrap() = u64::MAX;
        Self { bound, cdt }
    }

    pub fn bound(&self) -> u64 {
        self.bound
    }

    pub fn is_zero(&self) -> bool {
        self.bound == 0
    }

    pub fn sample_i64<Rn: Rng + ?Sized>(&self, rng: &mut Rn) -> i64 {
        if self.is_zero() {
            return 0;
        }
        let u = rng.next_u64();
        let idx = self.cdt.partition_point(|c| *c < u);
        idx as i64 - self.bound as i64
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    use super::*;
    use crate::ring::centered;

    #[test]
    fn test_discrete_gaussian() {
        let mut rng = ChaCha12Rng::seed_from_u64(4);
        for stddev in [0.5, 1.0, 3.2, 10.0] {
            let chi = DiscreteGaussian::new(stddev);
            assert_eq!(chi.bound(), Noise::DiscreteGaussian { stddev }.bound());

            let samples: Vec<i64> = (0..100_000).map(|_| chi.sample_i64(&mut rng)).collect();
            assert!(samples.iter().all(|x| x.unsigned_abs() <= chi.bound()));
            let mean = samples.iter().sum::<i64>() as f64 / samples.len() as f64;
            let var = samples
                .iter()
                .map(|x| (*x as f64 - mean).powi(2))
                .sum::<f64>()
                / samples.len() as f64;
            assert!(mean.abs() < 0.1, "stddev {stddev}: mean {mean}");

            // for small stddev, the discrete Gaussian has a smaller variance than stddev^2
            let b = chi.bound() as i64;
            let w = |x: i64| (-((x * x) as f64) / (2.0 * stddev * stddev)).exp();
            let expected_var =
                (-b..=b).map(|x| (x * x) as f64 * w(x)).sum::<f64>() / (-b..=b).map(w).sum::<f64>();
            let expected = expected_var.sqrt();
            assert!(
                (var.sqrt() - expected).abs() < 0.02 * expected,
                "stddev {stddev}: measured {}",
                var.sqrt()
            );
        }
    }

    #[test]
    fn test_tuniform() {
        let mut rng = ChaCha12Rng::seed_from_u64(6);
        for log_bound in [0, 1, 3, 8] {
            let noise = Noise::TUniform { log_bound };
            let chi = noise.sampler();
            let bound = noise.bound() as i64;
            let n = 200_000;
            let samples: Vec<i64> = (0..n).map(|_| chi.sample_i64(&mut rng)).collect();
            assert!(samples.iter().all(|x| x.abs() <= bound));
            assert!(samples.contains(&bound) && samples.contains(&-bound));

            let mean = samples.iter().sum::<i64>() as f64 / n as f64;
            let std = (samples
                .iter()
                .map(|x| (*x as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64)
                .sqrt();
            assert!(
                mean.abs() < 0.02 * noise.stddev().max(1.0),
                "log_bound {log_bound}: mean {mean}"
            );
            assert!(
                (std - noise.stddev()).abs() < 0.02 * noise.stddev(),
                "log_bound {log_bound}: measured {std}, expected {}",
                noise.stddev()
            );

            // the end points have probability 2^{-(b+2)}
            let ends = samples.iter().filter(|x| x.abs() == bound).count() as f64 / n as f64;
            let expected = 2.0 * 2f64.powi(-(log_bound as i32 + 2));
            assert!(
                (ends - expected).abs() < 0.1 * expected,
                "log_bound {log_bound}: {ends}"
            );
        }
    }

    #[test]
    fn test_noise_ring() {
        let mut rng = ChaCha12Rng::seed_from_u64(5);
        for noise in [
            Noise::DiscreteGaussian { stddev: 3.2 },
            Noise::TUniform { log_bound: 2 },
        ] {
            let chi = noise.sampler();
            for _ in 0..1000 {
                let x: Wrapping<u16> = chi.sample(&mut rng);
                assert!(centered(x, 12).unsigned_abs() <= noise.bound() as u128);
            }
        }

        assert_eq!(Noise::default().bound(), 1);
        let zero = Noise::Zero.sampler();
        assert!((0..100).all(|_| zero.sample_i64(&mut rng) == 0));
        assert!(DiscreteGaussian::new(0.0).is_zero());
    }
}
