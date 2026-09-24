//! Arithmetic over Z_q with q = 2^log_q, generic over the storage word.
//!
//! Elements are stored as `Wrapping<T>` for an unsigned integer `T` with `T::BITS >= log_q`.
//! All ring operations wrap modulo 2^BITS, which is compatible with reduction modulo q
//! because q divides 2^BITS. Values therefore only need to be reduced (masked to the low
//! `log_q` bits) where the representative matters, i.e., for bit decomposition, rounding
//! and lifting.

use std::{num::Wrapping, ops::Neg};

use nalgebra::{self as na, ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, Scalar};
use num_traits::{One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

pub trait Ring:
    Scalar
    + Copy
    + Zero
    + One
    + ClosedAddAssign
    + ClosedSubAssign
    + ClosedMulAssign
    + Neg<Output = Self>
    + PartialOrd
    + Send
    + Sync
{
    const BITS: u32;

    /// 2^i, which is 0 when i >= BITS.
    fn pow2(i: u32) -> Self;

    /// self mod 2^k.
    fn low_bits(self, k: u32) -> Self;

    /// Logical right shift, which is 0 when k >= BITS.
    fn shr(self, k: u32) -> Self;

    fn bit(self, i: u32) -> bool;

    fn from_bool(b: bool) -> Self;

    /// Uniformly random element of Z_{2^BITS}.
    fn sample<Rn: Rng + ?Sized>(rng: &mut Rn) -> Self;
}

macro_rules! impl_ring {
    ($($t:ty),*) => {$(
        impl Ring for Wrapping<$t> {
            const BITS: u32 = <$t>::BITS;

            fn pow2(i: u32) -> Self {
                Wrapping((1 as $t).checked_shl(i).unwrap_or(0))
            }

            fn low_bits(self, k: u32) -> Self {
                if k >= <$t>::BITS {
                    self
                } else {
                    Wrapping(self.0 & (((1 as $t) << k) - 1))
                }
            }

            fn shr(self, k: u32) -> Self {
                Wrapping(self.0.checked_shr(k).unwrap_or(0))
            }

            fn bit(self, i: u32) -> bool {
                (self.0 >> i) & 1 == 1
            }

            fn from_bool(b: bool) -> Self {
                Wrapping(b as $t)
            }

            fn sample<Rn: Rng + ?Sized>(rng: &mut Rn) -> Self {
                Wrapping(rng.random())
            }
        }
    )*};
}

impl_ring!(u8, u16, u32, u64, u128);

/// The ciphertext modulus q = 2^log_q and plaintext modulus p = 2^log_p, with p | q.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moduli {
    pub log_q: u32,
    pub log_p: u32,
}

/// Which party is lifting, i.e., the bit b in Lift^b.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Party {
    Alice,
    Bob,
}

impl Moduli {
    pub fn new(log_q: u32, log_p: u32) -> Self {
        assert!(1 <= log_p && log_p <= log_q);
        Self { log_q, log_p }
    }

    pub fn check<R: Ring>(&self) {
        assert!(
            self.log_q <= R::BITS,
            "log_q = {} does not fit in a {}-bit word",
            self.log_q,
            R::BITS
        );
    }

    /// Delta = q/p
    pub fn delta<R: Ring>(&self) -> R {
        R::pow2(self.log_q - self.log_p)
    }

    /// Rounds x in Z_q to the nearest element of Z_p, i.e., round(x * p / q) mod p.
    pub fn round<R: Ring>(&self, x: R) -> R {
        let shift = self.log_q - self.log_p;
        let x = x.low_bits(self.log_q);
        if shift == 0 {
            return x;
        }
        (x + R::pow2(shift - 1)).low_bits(self.log_q).shr(shift)
    }

    /// Lift^b_K(x) = uncenter_q(center_p(center_p(x) + (2b - 1) center_p(mask)))
    /// for x, mask in Z_p. The output is in Z_q.
    pub fn lift<R: Ring>(&self, x: R, mask: R, party: Party) -> R {
        let z = match party {
            Party::Alice => x - mask,
            Party::Bob => x + mask,
        }
        .low_bits(self.log_p);
        // center_p maps into (-p/2, p/2]
        let centered = if z > R::pow2(self.log_p - 1) {
            z - R::pow2(self.log_p)
        } else {
            z
        };
        centered.low_bits(self.log_q)
    }

    pub fn round_vec<R: Ring>(&self, v: na::DVectorView<R>) -> na::DVector<R> {
        v.map(|x| self.round(x))
    }

    pub fn reduce_p_vec<R: Ring>(&self, v: na::DVectorView<R>) -> na::DVector<R> {
        v.map(|x| x.low_bits(self.log_p))
    }
}

/// PRF F_K used to re-randomize shares before lifting.
/// Both parties must call it with the same `ctx` for the same vector.
#[derive(Clone, Debug)]
pub struct Prf {
    key: [u8; 32],
}

impl Prf {
    pub fn new<Rn: Rng + ?Sized>(rng: &mut Rn) -> Self {
        Self { key: rng.random() }
    }

    fn masks<R: Ring>(&self, moduli: Moduli, ctx: u64, len: usize) -> Vec<R> {
        let mut rng = ChaCha12Rng::from_seed(self.key);
        rng.set_stream(ctx);
        (0..len)
            .map(|_| R::sample(&mut rng).low_bits(moduli.log_p))
            .collect()
    }

    /// Component-wise lifting of v in Z_p^len to Z_q^len.
    pub fn lift_vec<R: Ring>(
        &self,
        moduli: Moduli,
        ctx: u64,
        party: Party,
        v: na::DVectorView<R>,
    ) -> na::DVector<R> {
        let masks = self.masks::<R>(moduli, ctx, v.len());
        na::DVector::from_iterator(
            v.len(),
            v.iter()
                .zip(masks)
                .map(|(x, mask)| moduli.lift(*x, mask, party)),
        )
    }
}

/// Bit decomposition (LSB first) of every element of x mod 2^log_q,
/// the output has length x.len() * log_q.
pub fn bits<R: Ring>(x: na::DVectorView<R>, log_q: u32) -> na::DVector<R> {
    na::DVector::from_iterator(
        x.len() * log_q as usize,
        x.iter()
            .flat_map(|x| (0..log_q).map(move |i| R::from_bool(x.bit(i)))),
    )
}

/// The inverse of `bits`, i.e., G_a * xhat where G_a = I_a \otimes g^T.
pub fn compose<R: Ring>(xhat: na::DVectorView<R>, log_q: u32) -> na::DVector<R> {
    let log_q = log_q as usize;
    assert_eq!(xhat.len() % log_q, 0);
    na::DVector::from_iterator(
        xhat.len() / log_q,
        xhat.as_slice().chunks(log_q).map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(R::zero(), |acc, (i, b)| acc + R::pow2(i as u32) * *b)
        }),
    )
}

pub fn sample_vec<R: Ring, Rn: Rng + ?Sized>(len: usize, rng: &mut Rn) -> na::DVector<R> {
    na::DVector::from_fn(len, |_, _| R::sample(rng))
}

pub fn sample_mat<R: Ring, Rn: Rng + ?Sized>(
    nrows: usize,
    ncols: usize,
    rng: &mut Rn,
) -> na::DMatrix<R> {
    na::DMatrix::from_fn(nrows, ncols, |_, _| R::sample(rng))
}

pub fn sample_binary_mat<R: Ring, Rn: Rng + ?Sized>(
    nrows: usize,
    ncols: usize,
    rng: &mut Rn,
) -> na::DMatrix<R> {
    na::DMatrix::from_fn(nrows, ncols, |_, _| R::from_bool(rng.random()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> ChaCha12Rng {
        ChaCha12Rng::seed_from_u64(42)
    }

    fn check_bits_roundtrip<R: Ring>(log_q: u32) {
        let mut rng = rng();
        let x = sample_vec::<R, _>(10, &mut rng);
        let xhat = bits(x.as_view(), log_q);
        assert_eq!(xhat.len(), 10 * log_q as usize);
        assert!(xhat.iter().all(|b| *b == R::zero() || *b == R::one()));
        let x2 = compose(xhat.as_view(), log_q);
        assert_eq!(x2, x.map(|x| x.low_bits(log_q)));
    }

    #[test]
    fn test_bits_roundtrip() {
        check_bits_roundtrip::<Wrapping<u8>>(8);
        check_bits_roundtrip::<Wrapping<u16>>(12);
        check_bits_roundtrip::<Wrapping<u16>>(16);
        check_bits_roundtrip::<Wrapping<u32>>(20);
        check_bits_roundtrip::<Wrapping<u64>>(64);
        check_bits_roundtrip::<Wrapping<u128>>(100);
        check_bits_roundtrip::<Wrapping<u128>>(128);
    }

    /// v + w = Delta * z mod q (no noise) must locally round to z mod p
    fn check_local_rounding<R: Ring>(moduli: Moduli, trials: usize) {
        moduli.check::<R>();
        let mut rng = rng();
        for _ in 0..trials {
            let z = R::sample(&mut rng).low_bits(moduli.log_p);
            let v = R::sample(&mut rng);
            let w = moduli.delta::<R>() * z - v;
            let res = (moduli.round(v) + moduli.round(w)).low_bits(moduli.log_p);
            assert_eq!(res, z);
        }
    }

    #[test]
    fn test_local_rounding() {
        // p = q: rounding is the identity mod q
        check_local_rounding::<Wrapping<u16>>(Moduli::new(12, 12), 1000);
        check_local_rounding::<Wrapping<u32>>(Moduli::new(32, 32), 1000);
        // p < q: fails with probability ~ p/q per trial (a tie), so use a large gap
        check_local_rounding::<Wrapping<u64>>(Moduli::new(64, 32), 10000);
        check_local_rounding::<Wrapping<u64>>(Moduli::new(60, 20), 10000);
        check_local_rounding::<Wrapping<u128>>(Moduli::new(128, 64), 10000);
    }

    /// Z_p shares of a bit must lift to Z_q shares of the same bit
    fn check_lift<R: Ring>(moduli: Moduli, len: usize) {
        moduli.check::<R>();
        let mut rng = rng();
        let prf = Prf::new(&mut rng);
        let m = na::DVector::from_fn(len, |_, _| R::from_bool(rng.random()));
        let x = sample_vec::<R, _>(len, &mut rng).map(|x| x.low_bits(moduli.log_p));
        let y = (&m - &x).map(|x| x.low_bits(moduli.log_p));

        let x_lift = prf.lift_vec(moduli, 7, Party::Alice, x.as_view());
        let y_lift = prf.lift_vec(moduli, 7, Party::Bob, y.as_view());
        assert_eq!((x_lift + y_lift).map(|x| x.low_bits(moduli.log_q)), m);
    }

    #[test]
    fn test_lift() {
        // p = q: lifting is exact
        check_lift::<Wrapping<u16>>(Moduli::new(12, 12), 10000);
        check_lift::<Wrapping<u16>>(Moduli::new(16, 16), 10000);
        check_lift::<Wrapping<u64>>(Moduli::new(16, 16), 10000);
        // p < q: fails with probability ~ 1/p per coordinate
        check_lift::<Wrapping<u64>>(Moduli::new(64, 32), 10000);
        check_lift::<Wrapping<u64>>(Moduli::new(64, 40), 10000);
        check_lift::<Wrapping<u128>>(Moduli::new(128, 64), 10000);
        check_lift::<Wrapping<u128>>(Moduli::new(128, 128), 10000);
    }
}
