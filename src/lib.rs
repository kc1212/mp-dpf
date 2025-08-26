use std::num::Wrapping;

use nalgebra as na;
use rand::{CryptoRng, Rng, distr::Distribution};

pub mod niote;

pub struct UniformRandom;

impl Distribution<Wrapping<u64>> for UniformRandom {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Wrapping<u64> {
        Wrapping(rng.next_u64())
    }
}

pub struct UniformRandomBool;

impl Distribution<bool> for UniformRandomBool {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        // NOTE: this is not the best way to sample,
        // we should sample in batches
        rng.random_bool(0.5)
    }
}

// TODO write proper sampling
pub fn sample_discrete_gaussian<R: Rng + CryptoRng>(
    _rng: &mut R,
    _stddev: f64,
    nrows: usize,
    mcols: usize,
) -> na::DMatrix<Wrapping<u64>> {
    na::DMatrix::zeros(nrows, mcols)
}
