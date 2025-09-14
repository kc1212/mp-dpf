use std::{num::Wrapping, ops::Add};

use nalgebra as na;
use rand::{CryptoRng, Rng, distr::Distribution};

pub mod half_niote;
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

/// This is the result of a kronecker product
/// between an identity matrix and another matrix.
/// We do not perform the actual kronecker product.
pub(crate) struct LazyIdentityKronecker<'a> {
    id_dim: usize,
    view: na::DMatrixView<'a, Wrapping<u64>>,
}

impl<'a> LazyIdentityKronecker<'a> {
    pub(crate) fn new(id_dim: usize, view: na::DMatrixView<'a, Wrapping<u64>>) -> Self {
        Self { id_dim, view }
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        let (r, c) = self.view.shape();
        (r * self.id_dim, c * self.id_dim)
    }

    // pub (crate) fn iter(&self) -> impl Iterator<Item = Wrapping<u64>> {
    //     todo!()
    // }
}

impl<'a> Add<LazyIdentityKronecker<'a>> for na::DMatrixView<'_, Wrapping<u64>> {
    type Output = na::DMatrix<Wrapping<u64>>;

    fn add(self, rhs: LazyIdentityKronecker<'a>) -> Self::Output {
        todo!()
    }
}
