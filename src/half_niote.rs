//! Fully correct half-succinct NIOTE for Z_p^m \otimes Z_p^ell,
//! see Figure 12 of eprint 2026/1942 (adapted from AMR25, Construction 3.4).
//!
//! Alice's input is decomposed in base b = 2^log_b (see `Moduli`), the paper uses b = 2.

use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::{
    noise::Noise,
    ring::{Moduli, Ring, decompose, recompose, sample_binary_mat, sample_mat, sample_vec},
};

#[derive(Clone, Copy, Debug)]
pub struct HalfNIOTEparam {
    pub moduli: Moduli,
    // length of Alice's input (\bar{m} in the paper)
    pub m: usize,
    // length of Bob's input
    pub ell: usize,
    // lwe dim
    pub k: usize,
    // the noise distribution chi
    pub noise: Noise,
    // If set, Alice's input must already be decomposed, i.e., its entries are in [0, b),
    // and it is hashed directly instead of hashing its decomposition.
    // This is used for the inner levels of the succinct NIOTE,
    // where Alice's input is the decomposition of digests.
    pub decomposed_input: bool,
}

impl HalfNIOTEparam {
    pub fn log_q(&self) -> u32 {
        self.moduli.log_q
    }

    // length of the digest, n = k log q since A is binary (non-uniform LWE)
    pub fn n(&self) -> usize {
        self.k * self.log_q() as usize
    }

    // number of digits used to represent one element of Alice's input
    pub fn gadget_width(&self) -> usize {
        if self.decomposed_input {
            1
        } else {
            self.moduli.digits_len()
        }
    }

    // length of Alice's hashed (decomposed) vector
    pub fn m_hat(&self) -> usize {
        self.m * self.gadget_width()
    }

    // length of the output shares
    pub fn out_len(&self) -> usize {
        self.m * self.ell
    }

    /// Bound on ||v + w - Delta * (x \otimes y)||_inf for the raw shares (Lemma 5),
    /// i.e., (n + 1) * m_hat * (b - 1) * B where B bounds the entries of E' and \tilde{E}
    /// and the entries of x_hat are at most b - 1.
    /// It is a f64 since it overflows a u128 for large gadget bases.
    pub fn error_bound(&self) -> f64 {
        (self.n() as f64 + 1.0)
            * self.m_hat() as f64
            * self.moduli.max_digit() as f64
            * self.noise.bound() as f64
    }

    /// Bound on the probability that local rounding fails for one output coordinate,
    /// i.e., (||e||_inf + 1) * p / q (Lemma 5).
    pub fn coord_rounding_failure(&self) -> f64 {
        let e = self.error_bound();
        if self.moduli.log_p == self.moduli.log_q {
            // no rounding happens, so any noise is an error
            return if e == 0.0 { 0.0 } else { 1.0 };
        }
        let log_delta = (self.moduli.log_q - self.moduli.log_p) as i32;
        ((e + 1.0) * 2f64.powi(-log_delta)).min(1.0)
    }

    /// log2 of the union bound on the probability that local rounding fails
    /// for at least one output coordinate.
    pub fn log2_rounding_failure(&self) -> f64 {
        (self.out_len() as f64 * self.coord_rounding_failure()).log2()
    }
}

pub struct HalfNIOTEpk<R: Ring> {
    // n x m_hat, binary
    pub a_mat: na::DMatrix<R>,
    // n x (m * ell * k), uniform
    pub b_mat: na::DMatrix<R>,
}

pub fn setup<R: Ring>(param: HalfNIOTEparam, rng: &mut (impl Rng + CryptoRng)) -> HalfNIOTEpk<R> {
    param.moduli.check::<R>();
    let a_mat = sample_binary_mat(param.n(), param.m_hat(), rng);
    let b_mat = sample_mat(param.n(), param.out_len() * param.k, rng);
    HalfNIOTEpk { a_mat, b_mat }
}

/// returns (d = A * x_hat, psi = x_hat)
pub fn alice_encode<R: Ring>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<R>,
    x_vec: na::DVectorView<R>,
) -> (na::DVector<R>, na::DVector<R>) {
    assert_eq!(x_vec.len(), param.m);
    let x_hat = if param.decomposed_input {
        debug_assert!(
            x_vec
                .iter()
                .all(|x| x.to_u128() <= param.moduli.max_digit())
        );
        x_vec.clone_owned()
    } else {
        decompose(x_vec, param.moduli)
    };
    let d = &pk.a_mat * &x_hat;
    (d, x_hat)
}

/// T = B * (I_{m \ell} \otimes s)
fn compute_t<R: Ring>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<R>,
    s: na::DVectorView<R>,
) -> na::DMatrix<R> {
    let k = param.k;
    let mut t_mat = na::DMatrix::zeros(param.n(), param.out_len());
    for h in 0..param.out_len() {
        t_mat
            .column_mut(h)
            .gemv(R::one(), &pk.b_mat.columns(h * k, k), &s, R::zero());
    }
    t_mat
}

/// returns (C, phi = s)
pub fn bob_encode<R: Ring>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<R>,
    y_vec: na::DVectorView<R>,
    rng: &mut (impl Rng + CryptoRng),
) -> (na::DMatrix<R>, na::DVector<R>) {
    assert_eq!(y_vec.len(), param.ell);
    let s_vec = sample_vec(param.k, rng);

    let chi = param.noise.sampler();

    // T = B * (I_{m \ell} \otimes s) + E'
    let mut t_mat = compute_t(param, pk, s_vec.as_view());
    chi.add_to(&mut t_mat, rng);

    // C = A^T * T + \tilde{E} + (I_m \otimes g) * (I_m \otimes \Delta y^T)
    let mut c_mat = pk.a_mat.tr_mul(&t_mat);
    chi.add_to(&mut c_mat, rng);
    debug_assert_eq!(c_mat.shape(), (param.m_hat(), param.out_len()));

    let delta = param.moduli.delta::<R>();
    let width = param.gadget_width();
    for i in 0..param.m {
        for b in 0..width {
            // g_b = b^i, the gadget entry of the digit
            let g_b = R::pow2(b as u32 * param.moduli.log_b) * delta;
            for c in 0..param.ell {
                c_mat[(i * width + b, i * param.ell + c)] += g_b * y_vec[c];
            }
        }
    }

    (c_mat, s_vec)
}

/// returns v = C^T * x_hat in Z_q
pub fn alice_decode_raw<R: Ring>(
    c_mat: na::DMatrixView<R>,
    psi: na::DVectorView<R>,
) -> na::DVector<R> {
    c_mat.tr_mul(&psi)
}

/// returns w = -(I_{m \ell} \otimes s^T) * B^T * d in Z_q
pub fn bob_decode_raw<R: Ring>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<R>,
    d_vec: na::DVectorView<R>,
    phi: na::DVectorView<R>,
) -> na::DVector<R> {
    let k = param.k;
    let bt_d = pk.b_mat.tr_mul(&d_vec);
    na::DVector::from_fn(param.out_len(), |h, _| -bt_d.rows(h * k, k).dot(&phi))
}

/// Length of the input of `apply_p`, i.e., of decompose(d) \otimes decompose(s).
pub fn p_input_len(param: HalfNIOTEparam) -> usize {
    let digits = param.moduli.digits_len();
    param.n() * digits * param.k * digits
}

/// Bilinear encoder evaluation (Lemma 7): returns P * v_hat such that
/// P * (decompose(d) \otimes decompose(s)) = bob_decode_raw(d, s).
/// Since P is linear, it can be applied to additive shares of the tensor.
pub fn apply_p<R: Ring>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<R>,
    v_hat: na::DVectorView<R>,
) -> na::DVector<R> {
    let (n, k, digits) = (param.n(), param.k, param.moduli.digits_len());
    assert_eq!(v_hat.len(), p_input_len(param));

    // recompose the digits of s: w[c * k + a] = \sum_j b^j v_hat[c * k digits + a digits + j]
    let w = recompose(v_hat, param.moduli);
    // recompose the digits of d: u[i, a] = \sum_j b^j w[(i digits + j) * k + a]
    // so u is a share of d * s^T
    let mut u = na::DMatrix::<R>::zeros(n, k);
    for i in 0..n {
        for j in 0..digits {
            let g_j = R::pow2(j as u32 * param.moduli.log_b);
            for a in 0..k {
                u[(i, a)] += g_j * w[(i * digits + j) * k + a];
            }
        }
    }

    // out[h] = -\sum_{i, a} B[i, h k + a] * u[i, a]
    na::DVector::from_fn(param.out_len(), |h, _| {
        let mut acc = R::zero();
        for a in 0..k {
            acc += pk.b_mat.column(h * k + a).dot(&u.column(a));
        }
        -acc
    })
}

pub fn alice_decode<R: Ring>(
    param: HalfNIOTEparam,
    c_mat: na::DMatrixView<R>,
    psi: na::DVectorView<R>,
) -> na::DVector<R> {
    param
        .moduli
        .round_vec(alice_decode_raw(c_mat, psi).as_view())
}

pub fn bob_decode<R: Ring>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<R>,
    d_vec: na::DVectorView<R>,
    phi: na::DVectorView<R>,
) -> na::DVector<R> {
    param
        .moduli
        .round_vec(bob_decode_raw(param, pk, d_vec, phi).as_view())
}

/// v + w mod p, where v and w are the (rounded) outputs of the decoders
pub fn reconstruct<R: Ring>(
    param: HalfNIOTEparam,
    v: na::DVectorView<R>,
    w: na::DVectorView<R>,
) -> na::DVector<R> {
    param.moduli.reduce_p_vec((v + w).as_view())
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use proptest::{
        arbitrary::{StrategyFor, arbitrary_with},
        prelude::*,
        sample::size_range,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    use super::*;
    use crate::ring::centered;

    const NOISE: Noise = Noise::TUniform { log_bound: 2 };

    fn gen_vec_wrapping(m: usize) -> StrategyFor<Vec<Wrapping<u64>>> {
        arbitrary_with(size_range(m).lift())
    }

    fn gen_bool(m: usize) -> StrategyFor<Vec<bool>> {
        arbitrary_with(size_range(m).lift())
    }

    proptest! {
        #[test]
        fn test_protocol_mod_2(xs in gen_vec_wrapping(4), ys in gen_bool(2)) {
            let param = HalfNIOTEparam {
                moduli: Moduli::new(64, 1),
                m: xs.len(),
                ell: ys.len(),
                k: 2,
                noise: NOISE,
                decomposed_input: false,
            };
            let x_vec = na::DVector::from_vec(xs);
            let y_vec = na::DVector::from_vec(ys.into_iter().map(|y| Wrapping(y as u64)).collect());
            let res = run_protocol(param, x_vec.as_view(), y_vec.as_view());

            let expected = x_vec.kronecker(&y_vec).map(|x| Wrapping(x.0 % 2));
            assert_eq!(res, expected);
        }
    }

    fn run_protocol<R: Ring>(
        param: HalfNIOTEparam,
        x_vec: na::DVectorView<R>,
        y_vec: na::DVectorView<R>,
    ) -> na::DVector<R> {
        let mut rng = ChaCha12Rng::from_os_rng();
        let pk = setup(param, &mut rng);

        let (d, psi) = alice_encode(param, &pk, x_vec);
        assert_eq!(d.shape(), (param.n(), 1));
        let (c, phi) = bob_encode(param, &pk, y_vec, &mut rng);
        assert_eq!(c.shape(), (param.m_hat(), param.out_len()));
        assert_eq!(phi.shape(), (param.k, 1));

        // the raw shares are \Delta * (x \otimes y) + e with ||e||_inf <= error_bound
        let v_raw = alice_decode_raw(c.as_view(), psi.as_view());
        let w_raw = bob_decode_raw(param, &pk, d.as_view(), phi.as_view());
        let e = v_raw + w_raw - x_vec.kronecker(&y_vec) * param.moduli.delta::<R>();
        let e: Vec<u128> = e
            .iter()
            .map(|x| centered(*x, param.log_q()).unsigned_abs())
            .collect();
        assert!(e.iter().all(|x| *x as f64 <= param.error_bound()));
        if param.noise == Noise::Zero {
            assert!(e.iter().all(|x| *x == 0));
        } else {
            assert!(e.iter().any(|x| *x != 0));
        }

        let v = alice_decode(param, c.as_view(), psi.as_view());
        let w = bob_decode(param, &pk, d.as_view(), phi.as_view());
        assert_eq!(v.shape(), (param.out_len(), 1));
        assert_eq!(v.shape(), w.shape());
        reconstruct(param, v.as_view(), w.as_view())
    }

    /// returns whether the protocol computes x \otimes y mod p
    fn check_protocol<R: Ring>(param: HalfNIOTEparam) -> bool {
        let moduli = param.moduli;
        let mut rng = ChaCha12Rng::seed_from_u64(1);
        let x_vec = if param.decomposed_input {
            na::DVector::from_fn(param.m, |_, _| R::sample(&mut rng).low_bits(moduli.log_b))
        } else {
            sample_vec::<R, _>(param.m, &mut rng).map(|x| x.low_bits(moduli.log_p))
        };
        let y_vec = sample_vec::<R, _>(param.ell, &mut rng).map(|x| x.low_bits(moduli.log_p));

        let res = run_protocol(param, x_vec.as_view(), y_vec.as_view());
        let expected = x_vec.kronecker(&y_vec).map(|x| x.low_bits(moduli.log_p));
        res == expected
    }

    fn param(moduli: Moduli, m: usize, ell: usize, k: usize, noise: Noise) -> HalfNIOTEparam {
        HalfNIOTEparam {
            moduli,
            m,
            ell,
            k,
            noise,
            decomposed_input: false,
        }
    }

    #[test]
    fn test_protocol_no_noise() {
        // without noise, p = q works and the protocol is exact
        for decomposed_input in [false, true] {
            let d = |p: HalfNIOTEparam| HalfNIOTEparam {
                decomposed_input,
                ..p
            };
            for log_b in [1, 3, 4] {
                let moduli = Moduli::new(12, 12).with_gadget_base(log_b);
                assert!(check_protocol::<Wrapping<u16>>(d(param(
                    moduli,
                    5,
                    3,
                    2,
                    Noise::Zero
                ))));
            }
            let moduli = Moduli::new(16, 16);
            assert!(check_protocol::<Wrapping<u16>>(d(param(
                moduli,
                4,
                7,
                1,
                Noise::Zero
            ))));
            let moduli = Moduli::new(20, 20).with_gadget_base(7);
            assert!(check_protocol::<Wrapping<u32>>(d(param(
                moduli,
                3,
                2,
                3,
                Noise::Zero
            ))));
            let moduli = Moduli::new(64, 32).with_gadget_base(16);
            assert!(check_protocol::<Wrapping<u64>>(d(param(
                moduli,
                4,
                3,
                2,
                Noise::Zero
            ))));
        }
    }

    #[test]
    fn test_protocol_noise() {
        let configs = [
            (Moduli::new(64, 20), 4, 3, 1, NOISE),
            (Moduli::new(64, 16), 4, 3, 2, NOISE),
            (Moduli::new(64, 20), 2, 5, 1, NOISE),
            (
                Moduli::new(64, 20),
                4,
                3,
                1,
                Noise::TUniform { log_bound: 0 },
            ),
            (
                Moduli::new(64, 20),
                4,
                3,
                2,
                Noise::TUniform { log_bound: 3 },
            ),
            (Moduli::new(64, 16).with_gadget_base(8), 4, 3, 1, NOISE),
            (
                Moduli::new(64, 16).with_gadget_base(8),
                3,
                2,
                2,
                Noise::TUniform { log_bound: 2 },
            ),
            (
                Moduli::new(64, 12).with_gadget_base(16),
                4,
                3,
                1,
                Noise::TUniform { log_bound: 0 },
            ),
        ];
        for decomposed_input in [false, true] {
            for (moduli, m, ell, k, noise) in configs {
                let param = HalfNIOTEparam {
                    decomposed_input,
                    ..param(moduli, m, ell, k, noise)
                };
                // make sure the test is not flaky
                assert!(param.log2_rounding_failure() < -20.0, "{param:?}");
                assert!(check_protocol::<Wrapping<u64>>(param), "{param:?}");
            }
            for log_b in [1, 16] {
                let moduli = Moduli::new(128, 64).with_gadget_base(log_b);
                let param = HalfNIOTEparam {
                    decomposed_input,
                    ..param(moduli, 2, 3, 1, NOISE)
                };
                assert!(param.log2_rounding_failure() < -20.0, "{param:?}");
                assert!(check_protocol::<Wrapping<u128>>(param), "{param:?}");
            }
        }
    }

    #[test]
    fn test_protocol_noise_too_large() {
        // Delta = 4 is much smaller than the noise, so rounding must fail
        let moduli = Moduli::new(64, 62);
        assert!(!check_protocol::<Wrapping<u64>>(param(
            moduli, 4, 3, 1, NOISE
        )));
    }

    fn check_apply_p<R: Ring>(moduli: Moduli, m: usize, ell: usize, k: usize) {
        let param = param(moduli, m, ell, k, NOISE);
        let mut rng = ChaCha12Rng::seed_from_u64(2);
        let pk = setup::<R>(param, &mut rng);
        let d = sample_vec::<R, _>(param.n(), &mut rng);
        let s = sample_vec::<R, _>(param.k, &mut rng);

        let d_hat = decompose(d.as_view(), moduli);
        let s_hat = decompose(s.as_view(), moduli);
        let tensor = d_hat.kronecker(&s_hat);
        assert_eq!(tensor.len(), p_input_len(param));

        // P applied to shares of the tensor
        let share0 = sample_vec::<R, _>(tensor.len(), &mut rng);
        let share1 = &tensor - &share0;
        let out = apply_p(param, &pk, share0.as_view()) + apply_p(param, &pk, share1.as_view());

        let expected = bob_decode_raw(param, &pk, d.as_view(), s.as_view());
        assert_eq!(
            out.map(|x| x.low_bits(moduli.log_q)),
            expected.map(|x| x.low_bits(moduli.log_q))
        );
    }

    #[test]
    fn test_apply_p() {
        for log_b in [1, 5, 12] {
            check_apply_p::<Wrapping<u16>>(Moduli::new(12, 12).with_gadget_base(log_b), 3, 2, 2);
        }
        check_apply_p::<Wrapping<u32>>(Moduli::new(32, 16).with_gadget_base(8), 2, 3, 1);
        check_apply_p::<Wrapping<u64>>(Moduli::new(64, 32), 2, 2, 3);
        check_apply_p::<Wrapping<u64>>(Moduli::new(64, 32).with_gadget_base(20), 2, 2, 3);
    }
}
