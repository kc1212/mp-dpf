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

    fn from_bool(b: bool) -> Self;

    /// x mod 2^BITS
    fn from_u64(x: u64) -> Self;

    /// The canonical representative in [0, 2^BITS).
    fn to_u128(self) -> u128;

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

            fn from_bool(b: bool) -> Self {
                Wrapping(b as $t)
            }

            fn from_u64(x: u64) -> Self {
                Wrapping(x as $t)
            }

            fn to_u128(self) -> u128 {
                self.0 as u128
            }

            fn sample<Rn: Rng + ?Sized>(rng: &mut Rn) -> Self {
                Wrapping(rng.random())
            }
        }
    )*};
}

impl_ring!(u8, u16, u32, u64, u128);

/// The ciphertext modulus q = 2^log_q, the plaintext modulus p = 2^log_p with p | q,
/// and the gadget base b = 2^log_b used for decomposing elements of Z_q into
/// digits_len() = ceil(log_q / log_b) digits in [0, b).
/// log_b = 1 is the binary decomposition used in the paper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moduli {
    pub log_q: u32,
    pub log_p: u32,
    pub log_b: u32,
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
        Self {
            log_q,
            log_p,
            log_b: 1,
        }
    }

    /// Use the gadget base b = 2^log_b.
    pub fn with_gadget_base(self, log_b: u32) -> Self {
        assert!(1 <= log_b && log_b <= self.log_q);
        Self { log_b, ..self }
    }

    /// Number of base-b digits of an element of Z_q.
    pub fn digits_len(&self) -> usize {
        self.log_q.div_ceil(self.log_b) as usize
    }

    /// b - 1, the largest digit.
    pub fn max_digit(&self) -> u128 {
        u128::MAX >> (128 - self.log_b)
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

/// The centered representative of x mod 2^log_q in (-2^{log_q - 1}, 2^{log_q - 1}].
pub fn centered<R: Ring>(x: R, log_q: u32) -> i128 {
    let x = x.low_bits(log_q);
    if x > R::pow2(log_q - 1) {
        -((-x).low_bits(log_q).to_u128() as i128)
    } else {
        x.to_u128() as i128
    }
}

/// Base-b decomposition (least significant digit first) of every element of x mod q,
/// the output has length x.len() * digits_len() and its entries are in [0, b).
pub fn decompose<R: Ring>(x: na::DVectorView<R>, moduli: Moduli) -> na::DVector<R> {
    let (log_q, log_b) = (moduli.log_q, moduli.log_b);
    let digits = moduli.digits_len() as u32;
    na::DVector::from_iterator(
        x.len() * digits as usize,
        x.iter().flat_map(move |x| {
            let x = x.low_bits(log_q);
            (0..digits).map(move |i| x.shr(i * log_b).low_bits(log_b))
        }),
    )
}

/// The inverse of `decompose`, i.e., G_a * xhat where G_a = I_a \otimes g^T
/// and g = (1, b, b^2, ...).
pub fn recompose<R: Ring>(xhat: na::DVectorView<R>, moduli: Moduli) -> na::DVector<R> {
    let digits = moduli.digits_len();
    assert_eq!(xhat.len() % digits, 0);
    na::DVector::from_iterator(
        xhat.len() / digits,
        xhat.as_slice().chunks(digits).map(|chunk| {
            chunk.iter().enumerate().fold(R::zero(), |acc, (i, d)| {
                acc + R::pow2(i as u32 * moduli.log_b) * *d
            })
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

    fn check_decompose_roundtrip<R: Ring>(log_q: u32, log_b: u32) {
        let moduli = Moduli::new(log_q, log_q).with_gadget_base(log_b);
        let mut rng = rng();
        let x = sample_vec::<R, _>(10, &mut rng);
        let xhat = decompose(x.as_view(), moduli);
        assert_eq!(xhat.len(), 10 * log_q.div_ceil(log_b) as usize);
        assert!(xhat.iter().all(|d| d.to_u128() <= moduli.max_digit()));
        let x2 = recompose(xhat.as_view(), moduli);
        assert_eq!(x2, x.map(|x| x.low_bits(log_q)));
    }

    #[test]
    fn test_decompose_roundtrip() {
        for (log_q, log_b) in [(8, 1), (8, 3), (8, 8)] {
            check_decompose_roundtrip::<Wrapping<u8>>(log_q, log_b);
        }
        for (log_q, log_b) in [(12, 1), (12, 5), (16, 4), (16, 16)] {
            check_decompose_roundtrip::<Wrapping<u16>>(log_q, log_b);
        }
        check_decompose_roundtrip::<Wrapping<u32>>(20, 7);
        check_decompose_roundtrip::<Wrapping<u64>>(64, 1);
        check_decompose_roundtrip::<Wrapping<u64>>(64, 32);
        check_decompose_roundtrip::<Wrapping<u128>>(100, 1);
        check_decompose_roundtrip::<Wrapping<u128>>(128, 32);
        check_decompose_roundtrip::<Wrapping<u128>>(128, 128);
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

    /// Z_p shares of a value in [0, max] must lift to Z_q shares of the same value,
    /// e.g., max = (b-1)^2 for a product of two digits
    fn check_lift<R: Ring>(moduli: Moduli, max: u64, len: usize) {
        moduli.check::<R>();
        let mut rng = rng();
        let prf = Prf::new(&mut rng);
        let m = na::DVector::from_fn(len, |_, _| R::from_u64(rng.random_range(0..=max)));
        let x = sample_vec::<R, _>(len, &mut rng).map(|x| x.low_bits(moduli.log_p));
        let y = (&m - &x).map(|x| x.low_bits(moduli.log_p));

        let x_lift = prf.lift_vec(moduli, 7, Party::Alice, x.as_view());
        let y_lift = prf.lift_vec(moduli, 7, Party::Bob, y.as_view());
        assert_eq!((x_lift + y_lift).map(|x| x.low_bits(moduli.log_q)), m);
    }

    #[test]
    fn test_lift() {
        // p = q: lifting is exact
        check_lift::<Wrapping<u16>>(Moduli::new(12, 12), 1, 10000);
        check_lift::<Wrapping<u16>>(Moduli::new(16, 16), 1, 10000);
        check_lift::<Wrapping<u64>>(Moduli::new(16, 16), 1, 10000);
        check_lift::<Wrapping<u64>>(Moduli::new(16, 16), 255 * 255, 10000);
        // p < q: fails with probability ~ (max + 1)/p per coordinate
        check_lift::<Wrapping<u64>>(Moduli::new(64, 32), 1, 10000);
        check_lift::<Wrapping<u64>>(Moduli::new(64, 40), 1, 10000);
        check_lift::<Wrapping<u64>>(Moduli::new(64, 48), 255 * 255, 10000);
        check_lift::<Wrapping<u128>>(Moduli::new(128, 64), 1, 10000);
        check_lift::<Wrapping<u128>>(Moduli::new(128, 96), u32::MAX as u64, 10000);
        check_lift::<Wrapping<u128>>(Moduli::new(128, 128), 1, 10000);
    }
}
