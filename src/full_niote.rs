//! Succinct NIOTE for Z_p^m \otimes Z_p^ell with m = t^r n, obtained by bootstrapping
//! the half-succinct NIOTE along a t-ary tree, see Figures 13 and 14 of eprint 2026/1942.
//!
//! Levels are indexed as in the paper: j = 0 is the root and j = r - 1 are the leaves.
//! Level j has t^j blocks.
//! - The leaf level uses hOTE_ell for Z_p^{tn} \otimes Z_p^ell.
//! - The inner levels j = r - 2, ..., 1 and the root use hOTE_n for
//!   Z_p^{tn l_b} \otimes Z_p^{k l_b}, where Alice's input is the base-b decomposition
//!   of t child digests and Bob's input is the decomposition of phi_{j+1}.
//!   Here l_b = ceil(log q / log b) is the number of digits, the paper uses b = 2.
//!
//! NOTE: the root uses hOTE_n instead of a hasher-private OTE,
//! so the digest is not hasher-private.

use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::{
    half_niote::{self, HalfNIOTEparam, HalfNIOTEpk},
    noise::Noise,
    ring::{Moduli, Party, Prf, Ring, decompose},
};

#[derive(Clone, Copy, Debug)]
pub struct NIOTEparam {
    pub moduli: Moduli,
    // lwe dim
    pub k: usize,
    // arity of the tree
    pub t: usize,
    // depth of the tree
    pub r: usize,
    // length of Bob's input
    pub ell: usize,
    // the noise distribution chi
    pub noise: Noise,
}

impl NIOTEparam {
    pub fn log_q(&self) -> u32 {
        self.moduli.log_q
    }

    // length of the digest
    pub fn n(&self) -> usize {
        self.k * self.log_q() as usize
    }

    // length of Alice's input
    pub fn m(&self) -> usize {
        self.t.pow(self.r as u32) * self.n()
    }

    pub fn blocks(&self, j: usize) -> usize {
        self.t.pow(j as u32)
    }

    /// hOTE_ell: Z_p^{tn} \otimes Z_p^ell
    pub fn leaf_param(&self) -> HalfNIOTEparam {
        HalfNIOTEparam {
            moduli: self.moduli,
            m: self.t * self.n(),
            ell: self.ell,
            k: self.k,
            noise: self.noise,
            decomposed_input: false,
        }
    }

    /// hOTE_n: Z_p^{tn l_b} \otimes Z_p^{k l_b}
    pub fn inner_param(&self) -> HalfNIOTEparam {
        let digits = self.moduli.digits_len();
        HalfNIOTEparam {
            moduli: self.moduli,
            m: self.t * self.n() * digits,
            ell: self.k * digits,
            k: self.k,
            noise: self.noise,
            decomposed_input: true,
        }
    }

    /// Length of the shares of decompose(h) \otimes decompose(phi) for a single child digest h,
    /// which is the input of the bilinear map P (or P').
    fn child_tensor_len(&self) -> usize {
        half_niote::p_input_len(self.leaf_param())
    }

    /// The shares that are lifted are shares of products of two digits, i.e., of values
    /// in [0, (b-1)^2]. Lifting needs (b-1)^2 < p/2 unless p = q (Corollary 2).
    pub fn lifting_is_sound(&self) -> bool {
        let m = &self.moduli;
        m.log_p == m.log_q || 2 * m.log_b < m.log_p
    }

    /// log2 of an upper bound on the probability that v + w != x \otimes y (mod p) (Theorem 4),
    /// using a union bound over all local roundings (Lemma 5) and all liftings
    /// (Corollary 2 with B = (b-1)^2, i.e., ((b-1)^2 + 1)/p per coordinate).
    pub fn log2_failure_bound(&self) -> f64 {
        let leaf = self.leaf_param();
        let inner = self.inner_param();
        // number of output coordinates of hOTE_n over the root and the inner levels
        let inner_coords =
            (0..self.r - 1).map(|j| self.blocks(j)).sum::<usize>() as f64 * inner.out_len() as f64;
        let leaf_coords = (self.blocks(self.r - 1) * leaf.out_len()) as f64;

        let rounding = inner_coords * inner.coord_rounding_failure()
            + leaf_coords * leaf.coord_rounding_failure();
        // lifting is exact when p = q
        let lifting = if self.moduli.log_p == self.moduli.log_q {
            0.0
        } else {
            let max = self.moduli.max_digit() as f64;
            inner_coords * (max * max + 1.0) * 2f64.powi(-(self.moduli.log_p as i32))
        };
        (rounding + lifting).min(1.0).log2()
    }

    /// Total number of ring elements in the encoding E = (E_0, ..., E_{r-1}).
    /// It is a u128 since secure parameters overflow a u64.
    pub fn encoding_len(&self) -> u128 {
        let leaf = self.leaf_param();
        let inner = self.inner_param();
        let size = |p: HalfNIOTEparam| p.m_hat() as u128 * p.out_len() as u128;
        size(leaf) + (self.r as u128 - 1) * size(inner)
    }

    /// Total number of ring elements in the public key (A and B of both instances).
    pub fn pk_len(&self) -> u128 {
        let size = |p: HalfNIOTEparam| {
            p.n() as u128 * (p.m_hat() as u128 + p.out_len() as u128 * p.k as u128)
        };
        size(self.leaf_param()) + size(self.inner_param())
    }
}

pub struct NIOTEpk<R: Ring> {
    leaf: HalfNIOTEpk<R>,
    inner: HalfNIOTEpk<R>,
    prf: Prf,
}

pub fn setup<R: Ring>(param: NIOTEparam, rng: &mut (impl Rng + CryptoRng)) -> NIOTEpk<R> {
    assert!(param.r >= 2, "the tree must have at least two levels");
    assert!(param.lifting_is_sound(), "need (b-1)^2 < p/2 for lifting");
    NIOTEpk {
        leaf: half_niote::setup(param.leaf_param(), rng),
        inner: half_niote::setup(param.inner_param(), rng),
        prf: Prf::new(rng),
    }
}

fn concat<R: Ring>(parts: &[na::DVector<R>]) -> na::DVector<R> {
    let len = parts.iter().map(|p| p.len()).sum();
    na::DVector::from_iterator(len, parts.iter().flat_map(|p| p.iter().copied()))
}

/// PRF context for lifting the shares of block i at level j,
/// both parties must use the same context.
fn lift_ctx(j: usize, i: usize) -> u64 {
    ((j as u64) << 40) | i as u64
}

/// Alice's secret information, psi[j][i] is the state of block i at level j.
pub type AliceState<R> = Vec<Vec<na::DVector<R>>>;

/// returns (d, psi)
pub fn alice_encode<R: Ring>(
    param: NIOTEparam,
    pk: &NIOTEpk<R>,
    x_vec: na::DVectorView<R>,
) -> (na::DVector<R>, AliceState<R>) {
    assert_eq!(x_vec.len(), param.m());
    let mut psi = vec![vec![]; param.r];

    // leaf level: hash t^{r-1} blocks of length tn
    let leaf = param.leaf_param();
    let mut digests = Vec::with_capacity(param.blocks(param.r - 1));
    for i in 0..param.blocks(param.r - 1) {
        let x_i = x_vec.rows(i * leaf.m, leaf.m);
        let (h, psi_i) = half_niote::alice_encode(leaf, &pk.leaf, x_i);
        digests.push(h);
        psi[param.r - 1].push(psi_i);
    }

    // inner levels and root: hash t^j blocks of t bit-decomposed child digests
    let inner = param.inner_param();
    for j in (0..param.r - 1).rev() {
        let d_hat = decompose(concat(&digests).as_view(), param.moduli);
        debug_assert_eq!(d_hat.len(), param.blocks(j) * inner.m);
        digests = Vec::with_capacity(param.blocks(j));
        for i in 0..param.blocks(j) {
            let d_hat_i = d_hat.rows(i * inner.m, inner.m);
            let (h, psi_i) = half_niote::alice_encode(inner, &pk.inner, d_hat_i);
            digests.push(h);
            psi[j].push(psi_i);
        }
    }

    assert_eq!(digests.len(), 1);
    (digests.pop().unwrap(), psi)
}

/// returns ({E_j}, phi_0), where the encodings are indexed by level
pub fn bob_encode<R: Ring>(
    param: NIOTEparam,
    pk: &NIOTEpk<R>,
    y_vec: na::DVectorView<R>,
    rng: &mut (impl Rng + CryptoRng),
) -> (Vec<na::DMatrix<R>>, na::DVector<R>) {
    assert_eq!(y_vec.len(), param.ell);
    let mut e_out = Vec::with_capacity(param.r);

    let (e_leaf, mut phi) = half_niote::bob_encode(param.leaf_param(), &pk.leaf, y_vec, rng);
    e_out.push(e_leaf);

    for _ in (0..param.r - 1).rev() {
        let phi_hat = decompose(phi.as_view(), param.moduli);
        let (e_j, phi_j) =
            half_niote::bob_encode(param.inner_param(), &pk.inner, phi_hat.as_view(), rng);
        e_out.push(e_j);
        phi = phi_j;
    }

    // index by level, i.e., the root is at index 0
    e_out.reverse();
    (e_out, phi)
}

/// returns Alice's share v in Z_p^{m \ell}
pub fn alice_decode<R: Ring>(
    param: NIOTEparam,
    pk: &NIOTEpk<R>,
    e_mats: &[na::DMatrix<R>],
    psi: &AliceState<R>,
) -> na::DVector<R> {
    assert_eq!(e_mats.len(), param.r);
    let moduli = param.moduli;
    let inner = param.inner_param();
    let leaf = param.leaf_param();
    let chunk = param.child_tensor_len();

    // root: v_0 is a share of decompose(d_1) \otimes decompose(phi_1)
    let v_0 = moduli.round_vec(
        half_niote::alice_decode_raw(e_mats[0].as_view(), psi[0][0].as_view()).as_view(),
    );
    let mut v_prev = pk
        .prf
        .lift_vec(moduli, lift_ctx(0, 0), Party::Alice, v_0.as_view());

    // inner levels: v_j^i is a share of decompose(d_{j+1}^i) \otimes decompose(phi_{j+1})
    for j in 1..param.r - 1 {
        assert_eq!(v_prev.len(), param.blocks(j) * chunk);
        let v_j: Vec<_> = (0..param.blocks(j))
            .map(|i| {
                let raw = half_niote::alice_decode_raw(e_mats[j].as_view(), psi[j][i].as_view())
                    + half_niote::apply_p(inner, &pk.inner, v_prev.rows(i * chunk, chunk));
                let rounded = moduli.round_vec(raw.as_view());
                pk.prf
                    .lift_vec(moduli, lift_ctx(j, i), Party::Alice, rounded.as_view())
            })
            .collect();
        v_prev = concat(&v_j);
    }

    // leaves: v_{r-1}^i is a share of x_i \otimes y
    let j = param.r - 1;
    assert_eq!(v_prev.len(), param.blocks(j) * chunk);
    let v_leaf: Vec<_> = (0..param.blocks(j))
        .map(|i| {
            let raw = half_niote::alice_decode_raw(e_mats[j].as_view(), psi[j][i].as_view())
                + half_niote::apply_p(leaf, &pk.leaf, v_prev.rows(i * chunk, chunk));
            moduli.round_vec(raw.as_view())
        })
        .collect();
    let v = concat(&v_leaf);
    debug_assert_eq!(v.len(), param.m() * param.ell);
    v
}

/// returns Bob's share w in Z_p^{m \ell}
pub fn bob_decode<R: Ring>(
    param: NIOTEparam,
    pk: &NIOTEpk<R>,
    d_vec: na::DVectorView<R>,
    phi: na::DVectorView<R>,
) -> na::DVector<R> {
    let moduli = param.moduli;
    let inner = param.inner_param();
    let leaf = param.leaf_param();
    let chunk = param.child_tensor_len();

    // root: w_0 is a share of decompose(d_1) \otimes decompose(phi_1)
    let w_0 = half_niote::bob_decode(inner, &pk.inner, d_vec, phi);
    let mut w_prev = pk
        .prf
        .lift_vec(moduli, lift_ctx(0, 0), Party::Bob, w_0.as_view());

    // inner levels
    for j in 1..param.r - 1 {
        assert_eq!(w_prev.len(), param.blocks(j) * chunk);
        let w_j: Vec<_> = (0..param.blocks(j))
            .map(|i| {
                let raw = half_niote::apply_p(inner, &pk.inner, w_prev.rows(i * chunk, chunk));
                let rounded = moduli.round_vec(raw.as_view());
                pk.prf
                    .lift_vec(moduli, lift_ctx(j, i), Party::Bob, rounded.as_view())
            })
            .collect();
        w_prev = concat(&w_j);
    }

    // leaves
    let j = param.r - 1;
    assert_eq!(w_prev.len(), param.blocks(j) * chunk);
    let w_leaf: Vec<_> = (0..param.blocks(j))
        .map(|i| {
            let raw = half_niote::apply_p(leaf, &pk.leaf, w_prev.rows(i * chunk, chunk));
            moduli.round_vec(raw.as_view())
        })
        .collect();
    concat(&w_leaf)
}

pub fn reconstruct<R: Ring>(
    param: NIOTEparam,
    v: na::DVectorView<R>,
    w: na::DVectorView<R>,
) -> na::DVector<R> {
    param.moduli.reduce_p_vec((v + w).as_view())
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    use super::*;
    use crate::ring::{centered, sample_vec};

    fn run_protocol<R: Ring>(param: NIOTEparam) {
        if param.noise == Noise::Zero {
            assert_eq!(param.log2_failure_bound(), f64::NEG_INFINITY);
        } else {
            // make sure the test is not flaky
            assert!(param.log2_failure_bound() < -20.0, "{param:?} is flaky");
        }

        let mut rng = ChaCha12Rng::seed_from_u64(3);
        let moduli = param.moduli;
        let x_vec = sample_vec::<R, _>(param.m(), &mut rng).map(|x| x.low_bits(moduli.log_p));
        let y_vec = sample_vec::<R, _>(param.ell, &mut rng).map(|x| x.low_bits(moduli.log_p));

        let pk = setup(param, &mut rng);

        let (d_vec, psi) = alice_encode(param, &pk, x_vec.as_view());
        assert_eq!(d_vec.len(), param.n());
        assert_eq!(psi.len(), param.r);
        for (j, psi_j) in psi.iter().enumerate() {
            assert_eq!(psi_j.len(), param.blocks(j));
        }

        let (e_mats, phi) = bob_encode(param, &pk, y_vec.as_view(), &mut rng);
        assert_eq!(e_mats.len(), param.r);
        assert_eq!(phi.len(), param.k);

        let v_vec = alice_decode(param, &pk, &e_mats, &psi);
        let w_vec = bob_decode(param, &pk, d_vec.as_view(), phi.as_view());
        assert_eq!(v_vec.len(), param.m() * param.ell);
        assert_eq!(w_vec.len(), param.m() * param.ell);

        let z = reconstruct(param, v_vec.as_view(), w_vec.as_view());
        let expected = x_vec.kronecker(&y_vec).map(|x| x.low_bits(moduli.log_p));
        assert_eq!(z, expected);
    }

    fn param(moduli: Moduli, k: usize, t: usize, r: usize, ell: usize) -> NIOTEparam {
        NIOTEparam {
            moduli,
            k,
            t,
            r,
            ell,
            noise: Noise::Zero,
        }
    }

    // NOTE: the tests use p = q and no noise, the protocol is then exact and
    // rounding/lifting are the identity. With noise or p < q, local rounding and lifting
    // fail with probability about (||e|| + 1)/Delta and ((b-1)^2 + 1)/p per coordinate
    // (see `log2_failure_bound`), which requires a large log_q and makes the encodings
    // and the public key too large for a unit test.

    #[test]
    fn test_protocol_u8_deep() {
        run_protocol::<Wrapping<u8>>(param(Moduli::new(8, 8), 1, 2, 4, 3));
    }

    #[test]
    fn test_protocol_u16() {
        run_protocol::<Wrapping<u16>>(param(Moduli::new(16, 16), 1, 2, 2, 3));
    }

    #[test]
    fn test_protocol_u16_masked() {
        // log_q smaller than the word size
        run_protocol::<Wrapping<u16>>(param(Moduli::new(12, 12), 1, 2, 3, 5));
    }

    #[test]
    fn test_protocol_u32_t3() {
        run_protocol::<Wrapping<u32>>(param(Moduli::new(16, 16), 1, 3, 2, 2));
    }

    #[test]
    fn test_protocol_u64_k2() {
        run_protocol::<Wrapping<u64>>(param(Moduli::new(10, 10), 2, 2, 2, 4));
    }

    #[test]
    fn test_protocol_gadget_base() {
        // log_b does not divide log_q, so the last digit is smaller
        run_protocol::<Wrapping<u8>>(param(Moduli::new(8, 8).with_gadget_base(3), 1, 2, 4, 3));
        run_protocol::<Wrapping<u16>>(param(Moduli::new(12, 12).with_gadget_base(5), 2, 2, 3, 2));
        run_protocol::<Wrapping<u16>>(param(Moduli::new(16, 16).with_gadget_base(4), 1, 3, 2, 3));
        // the digits are the whole element
        run_protocol::<Wrapping<u16>>(param(Moduli::new(16, 16).with_gadget_base(16), 1, 2, 3, 2));
    }

    #[test]
    fn test_protocol_gadget_base_large_q() {
        // a larger gadget base makes log q = 64 and 128 feasible
        run_protocol::<Wrapping<u64>>(param(Moduli::new(64, 64).with_gadget_base(16), 1, 2, 3, 2));
        run_protocol::<Wrapping<u128>>(param(
            Moduli::new(128, 128).with_gadget_base(32),
            1,
            2,
            2,
            1,
        ));
    }

    /// The full protocol with noise, p < q, rounding and lifting at every level.
    /// This is the smallest such configuration: it needs k = 1 (no security) and about
    /// 1.5 GB, see the README. It only runs in release builds since it is slow otherwise.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "slow without optimisations, use --release")]
    fn test_protocol_noise_end_to_end() {
        let param = NIOTEparam {
            moduli: Moduli::new(127, 60).with_gadget_base(12),
            k: 1,
            t: 2,
            r: 2,
            ell: 1,
            noise: Noise::default(),
        };
        run_protocol::<Wrapping<u128>>(param);
    }

    /// One decoding step of Figure 14 with noise, i.e., the induction step of Lemma 9:
    /// given lifted shares of decompose(h) \otimes decompose(s) from the parent level, the parties
    /// locally compute round(De_A^raw + P * v_hat) and round(P * w_hat), which must be
    /// shares of x \otimes y. A whole tree with noise is too large to test, see
    /// `examples/params.rs`.
    fn check_level_with_noise<R: Ring>(
        moduli: Moduli,
        m: usize,
        ell: usize,
        noise: Noise,
        decomposed_input: bool,
    ) {
        let param = HalfNIOTEparam {
            moduli,
            m,
            ell,
            k: 1,
            noise,
            decomposed_input,
        };
        let tensor_len = half_niote::p_input_len(param);
        let max = moduli.max_digit() as f64;
        let lift_failure =
            tensor_len as f64 * (max * max + 1.0) * 2f64.powi(-(moduli.log_p as i32));
        assert!(
            param.log2_rounding_failure() < -20.0 && lift_failure.log2() < -20.0,
            "{param:?} is flaky"
        );

        let mut rng = ChaCha12Rng::seed_from_u64(6);
        let x_vec = if decomposed_input {
            na::DVector::from_fn(m, |_, _| R::sample(&mut rng).low_bits(moduli.log_b))
        } else {
            sample_vec::<R, _>(m, &mut rng).map(|x| x.low_bits(moduli.log_p))
        };
        let y_vec = sample_vec::<R, _>(ell, &mut rng).map(|x| x.low_bits(moduli.log_p));

        let pk = half_niote::setup::<R>(param, &mut rng);
        let prf = Prf::new(&mut rng);
        let (h, psi) = half_niote::alice_encode(param, &pk, x_vec.as_view());
        let (c_mat, s) = half_niote::bob_encode(param, &pk, y_vec.as_view(), &mut rng);

        // the parent level outputs Z_p shares of decompose(h) \otimes decompose(s), which are lifted
        let tensor = decompose(h.as_view(), moduli).kronecker(&decompose(s.as_view(), moduli));
        assert_eq!(tensor.len(), tensor_len);
        let t0 = sample_vec::<R, _>(tensor_len, &mut rng).map(|x| x.low_bits(moduli.log_p));
        let t1 = (&tensor - &t0).map(|x| x.low_bits(moduli.log_p));
        let v_hat = prf.lift_vec(moduli, 1, Party::Alice, t0.as_view());
        let w_hat = prf.lift_vec(moduli, 1, Party::Bob, t1.as_view());
        assert_eq!((&v_hat + &w_hat).map(|x| x.low_bits(moduli.log_q)), tensor);

        let v_raw = half_niote::alice_decode_raw(c_mat.as_view(), psi.as_view())
            + half_niote::apply_p(param, &pk, v_hat.as_view());
        let w_raw = half_niote::apply_p(param, &pk, w_hat.as_view());

        // the noise is the same as for the half-succinct NIOTE
        let e = &v_raw + &w_raw - x_vec.kronecker(&y_vec) * moduli.delta::<R>();
        assert!(e.iter().any(|x| *x != R::zero()));
        assert!(
            e.iter()
                .all(|x| centered(*x, moduli.log_q).unsigned_abs() as f64 <= param.error_bound())
        );

        let v = moduli.round_vec(v_raw.as_view());
        let w = moduli.round_vec(w_raw.as_view());
        let z = moduli.reduce_p_vec((v + w).as_view());
        assert_eq!(z, x_vec.kronecker(&y_vec).map(|x| x.low_bits(moduli.log_p)));
    }

    #[test]
    fn test_level_with_noise() {
        let tuniform0 = Noise::TUniform { log_bound: 0 };
        let tuniform2 = Noise::TUniform { log_bound: 2 };
        for decomposed_input in [false, true] {
            let moduli = Moduli::new(100, 50);
            check_level_with_noise::<Wrapping<u128>>(moduli, 2, 3, tuniform2, decomposed_input);
            let moduli = Moduli::new(90, 45);
            check_level_with_noise::<Wrapping<u128>>(moduli, 3, 2, tuniform0, decomposed_input);
            let moduli = Moduli::new(100, 53).with_gadget_base(8);
            check_level_with_noise::<Wrapping<u128>>(moduli, 2, 3, tuniform2, decomposed_input);
            let moduli = Moduli::new(128, 72).with_gadget_base(16);
            check_level_with_noise::<Wrapping<u128>>(moduli, 3, 2, tuniform0, decomposed_input);
        }
    }
}
