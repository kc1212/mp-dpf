use std::{
    fmt::Debug,
    num::Wrapping,
    ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bitvec::{order::Lsb0, view::BitView};
use itertools::izip;
use nalgebra as na;
use num_traits::Unsigned;
use rand::{CryptoRng, Rng, distr::Distribution};

pub mod full_niote;
pub mod half_niote;

pub trait Scalar:
    Unsigned
    + Copy
    + Clone
    + Debug
    + Add
    + Mul
    + Sub
    + AddAssign
    + MulAssign
    + SubAssign
    + Neg
    + 'static
{
    const BIT_LENGTH: usize;
    fn from_u64(x: u64) -> Self;
    fn as_u64(&self) -> u64;
    fn view_bits(&self) -> impl Iterator<Item = Self>;
}

impl Scalar for Wrapping<u64> {
    const BIT_LENGTH: usize = 64;

    fn from_u64(x: u64) -> Self {
        Self(x)
    }

    fn as_u64(&self) -> u64 {
        self.0
    }

    fn view_bits(&self) -> impl Iterator<Item = Self> {
        self.0
            .view_bits::<Lsb0>()
            .to_bitvec()
            .into_iter()
            .map(|x| if x { Wrapping(1) } else { Wrapping(0) })
    }
}

impl Scalar for Wrapping<u32> {
    const BIT_LENGTH: usize = 32;

    fn from_u64(x: u64) -> Self {
        Self(x as u32)
    }

    fn as_u64(&self) -> u64 {
        self.0 as u64
    }

    fn view_bits(&self) -> impl Iterator<Item = Self> {
        self.0
            .view_bits::<Lsb0>()
            .to_bitvec()
            .into_iter()
            .map(|x| if x { Wrapping(1) } else { Wrapping(0) })
    }
}

impl Scalar for Wrapping<u16> {
    const BIT_LENGTH: usize = 16;

    fn from_u64(x: u64) -> Self {
        Self(x as u16)
    }

    fn as_u64(&self) -> u64 {
        self.0 as u64
    }

    fn view_bits(&self) -> impl Iterator<Item = Self> {
        self.0
            .view_bits::<Lsb0>()
            .to_bitvec()
            .into_iter()
            .map(|x| if x { Wrapping(1) } else { Wrapping(0) })
    }
}

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
pub fn sample_discrete_gaussian<R: Rng + CryptoRng, S: Scalar>(
    _rng: &mut R,
    _stddev: f64,
    nrows: usize,
    mcols: usize,
) -> na::DMatrix<S> {
    na::DMatrix::zeros(nrows, mcols)
}

/// This is the result of a kronecker product
/// between an identity matrix and another matrix.
/// We do not perform the actual kronecker product.
pub(crate) struct LazyIdentityKronecker<'a, S: Scalar> {
    id_dim: usize,
    view: na::DMatrixView<'a, S>,
    // we need to keep a value here so that we can return its reference
    zero: S,
}

impl<'a, S: Scalar> LazyIdentityKronecker<'a, S> {
    pub(crate) fn new(id_dim: usize, view: na::DMatrixView<'a, S>) -> Self {
        Self {
            id_dim,
            view,
            zero: S::zero(),
        }
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        let (r, c) = self.view.shape();
        (r * self.id_dim, c * self.id_dim)
    }

    pub(crate) fn internal_shape(&self) -> (usize, usize) {
        self.view.shape()
    }

    pub(crate) fn internal(&'a self) -> na::DMatrixView<'a, S> {
        self.view.as_view()
    }
}

impl<'a, S: Scalar> Index<(usize, usize)> for LazyIdentityKronecker<'a, S> {
    type Output = S;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        let i_factor = i / self.view.shape().0;
        let j_factor = j / self.view.shape().1;
        if i_factor != j_factor {
            &self.zero
        } else {
            // index i, j of the concrete matrix
            let i = i % self.view.shape().0;
            let j = j % self.view.shape().1;
            self.view.get((i, j)).unwrap()
        }
    }
}

impl<'a, S: Scalar> Add<&LazyIdentityKronecker<'a, S>> for na::DMatrixView<'_, S> {
    type Output = na::DMatrix<S>;

    fn add(self, rhs: &LazyIdentityKronecker<'a, S>) -> Self::Output {
        let mut out = na::DMatrix::<S>::zeros(self.shape().0, self.shape().1);
        izip!(out.row_iter_mut(), self.row_iter())
            .enumerate()
            .for_each(|(i, (mut out_rows, a_rows))| {
                izip!(out_rows.iter_mut(), a_rows.iter())
                    .enumerate()
                    .for_each(|(j, (out, a_val))| {
                        // somehow we cannot use the operator
                        *out = a_val.add(rhs[(i, j)]);
                    })
            });
        out
    }
}

impl<'a, S: Scalar> Add<LazyIdentityKronecker<'a, S>> for na::DMatrixView<'_, S> {
    type Output = na::DMatrix<S>;
    fn add(self, rhs: LazyIdentityKronecker<'a, S>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, S: Scalar> Mul<&LazyIdentityKronecker<'a, S>> for na::DMatrixView<'_, S> {
    type Output = na::DMatrix<S>;

    fn mul(self, rhs: &LazyIdentityKronecker<'a, S>) -> Self::Output {
        // compute A * B
        let a_mat = self;
        let b_mat = rhs;

        assert_eq!(a_mat.shape().1, b_mat.shape().0);

        let mut out = na::DMatrix::<S>::zeros(a_mat.shape().0, b_mat.shape().1);

        // iterate over rows
        for i in 0..a_mat.shape().0 {
            // iterate over columns
            for j in 0..b_mat.shape().1 {
                // we assume the LazyIdentityKronecker has the shape
                // s_0   0   0   0   0
                // s_1   0   0   0   0
                //   0 s_0   0   0   0
                //   0 s_1   0   0   0
                //   ...
                //   0   0   0   0 s_0
                //   0   0   0   0 s_1
                let block_len = b_mat.internal_shape().0;
                assert_eq!(b_mat.internal_shape().1, 1);
                for k in 0..block_len {
                    out[(i, j)] += a_mat[(i, block_len * j + k)] * b_mat.internal()[(k, 0)];
                }
            }
        }
        out
    }
}

impl<'a, S: Scalar> Mul<LazyIdentityKronecker<'a, S>> for na::DMatrixView<'_, S> {
    type Output = na::DMatrix<S>;

    fn mul(self, rhs: LazyIdentityKronecker<'a, S>) -> Self::Output {
        self * &rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_kronecker_mul() {
        let id_dim = 10;
        let v: na::DVector<Wrapping<u64>> = na::DVector::new_random(5);

        let lazy_mat = LazyIdentityKronecker::new(id_dim, v.as_view());
        assert_eq!(lazy_mat.shape(), (v.shape().0 * id_dim, id_dim));

        let a_mat = na::DMatrix::new_random(3, lazy_mat.shape().0);
        let z = a_mat.as_view() * &lazy_mat;

        let id_mat = na::DMatrix::identity(id_dim, id_dim);
        let b_mat = id_mat.kronecker(&v);
        let z_expected = a_mat * &b_mat;

        for i in 0..b_mat.shape().0 {
            for j in 0..b_mat.shape().1 {
                assert_eq!(lazy_mat[(i, j)], b_mat[(i, j)]);
            }
        }

        assert_eq!(z, z_expected);
    }
}
