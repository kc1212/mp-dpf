//! Noise distribution chi for the LWE encodings.

use nalgebra as na;
use rand::Rng;

use crate::ring::Ring;

/// The discrete Gaussian is truncated at TAILCUT standard deviations,
/// so chi is B-bounded with B = ceil(TAILCUT * stddev) as required by the paper.
pub const TAILCUT: f64 = 6.0;

/// Bound B on the absolute value of a sample of chi.
pub fn noise_bound(stddev: f64) -> u64 {
    (TAILCUT * stddev).ceil() as u64
}

/// Discrete Gaussian over Z with standard deviation `stddev`,
/// truncated to [-B, B], sampled with a cumulative distribution table.
/// A standard deviation of 0 gives the zero distribution.
///
/// NOTE: the sampler is not constant time.
pub struct DiscreteGaussian {
    bound: u64,
    // cdt[i] = 2^64 * Pr[X <= i - bound]
    cdt: Vec<u64>,
}

impl DiscreteGaussian {
    pub fn new(stddev: f64) -> Self {
        assert!(stddev >= 0.0);
        let bound = noise_bound(stddev);
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

    pub fn sample<R: Ring, Rn: Rng + ?Sized>(&self, rng: &mut Rn) -> R {
        let x = self.sample_i64(rng);
        let abs = R::from_u64(x.unsigned_abs());
        if x < 0 { -abs } else { abs }
    }

    /// mat += E where E is sampled from chi entry-wise
    pub fn add_to<R: Ring, Rn: Rng + ?Sized>(&self, mat: &mut na::DMatrix<R>, rng: &mut Rn) {
        if self.is_zero() {
            return;
        }
        mat.iter_mut().for_each(|x| *x += self.sample(rng));
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
            assert_eq!(chi.bound(), noise_bound(stddev));

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
    fn test_discrete_gaussian_ring() {
        let mut rng = ChaCha12Rng::seed_from_u64(5);
        let chi = DiscreteGaussian::new(3.2);
        for _ in 0..1000 {
            let x: Wrapping<u16> = chi.sample(&mut rng);
            assert!(centered(x, 12).unsigned_abs() <= chi.bound() as u128);
        }

        let zero = DiscreteGaussian::new(0.0);
        assert!(zero.is_zero());
        assert!((0..100).all(|_| zero.sample_i64(&mut rng) == 0));
    }
}
