use std::num::Wrapping;

use bitvec::{order::Lsb0, view::BitView};
use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::sample_discrete_gaussian;

pub struct HalfNIOTEpk {
    pub a_mat: na::DMatrix<Wrapping<u64>>,
    pub b_mat: na::DMatrix<Wrapping<u64>>,
}

#[derive(Clone, Copy, Debug)]
pub struct HalfNIOTEparam {
    pub log_q: u64, // fixed to 64
    pub m: usize,
    pub ell: usize,
    pub k: usize, // lwe dim
    pub stddev: f64,
}

impl HalfNIOTEparam {
    pub fn n(&self) -> usize {
        self.k * self.log_q as usize
    }

    // length of Alice's input
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn log_q(&self) -> u64 {
        self.log_q
    }

    // length of Bob's input
    pub fn ell(&self) -> usize {
        self.ell
    }

    // length of the LWE secret
    pub fn k(&self) -> usize {
        self.k
    }
}

pub fn setup<R: Rng + CryptoRng>(param: HalfNIOTEparam, _rng: &mut R) -> HalfNIOTEpk {
    // let a_mat = na::DMatrix::<bool>::from_distribution_generic(
    //     nalgebra::Dyn(param.n),
    //     nalgebra::Dyn(param.m * param.log_q),
    //     &UniformRandomBool,
    //     rng,
    // );
    let a_mat = na::DMatrix::<bool>::new_random(param.n(), param.m * param.log_q as usize);
    let a_mat = a_mat.map(|x| if x { Wrapping(1u64) } else { Wrapping(0u64) });

    let b_mat = na::DMatrix::<bool>::new_random(param.n(), param.m * param.ell * param.n());
    let b_mat = b_mat.map(|x| if x { Wrapping(1u64) } else { Wrapping(0u64) });

    HalfNIOTEpk { a_mat, b_mat }
}

fn bits(x_vec: na::DVectorView<Wrapping<u64>>) -> na::DVector<Wrapping<u64>> {
    let input_len = x_vec.len();
    // bits function
    let x_vec_hat = x_vec
        .iter()
        .flat_map(|x| {
            x.0.view_bits::<Lsb0>()
                .to_bitvec()
                .into_iter()
                .map(|x| Wrapping(x as u64))
        })
        .collect::<Vec<Wrapping<u64>>>();
    assert_eq!(x_vec_hat.len(), 64 * input_len);
    na::DVector::from_vec(x_vec_hat)
}

/// returns (d, psi = x_hat)
pub fn alice_encode(
    pk: &HalfNIOTEpk,
    x_vec: na::DVectorView<Wrapping<u64>>,
) -> (na::DVector<Wrapping<u64>>, na::DVector<Wrapping<u64>>) {
    let a_mat: na::DMatrixView<Wrapping<u64>> = pk.a_mat.as_view();
    let x_vec_hat = bits(x_vec);
    debug_assert_eq!(x_vec_hat.len(), x_vec.len() * 64);
    let d = a_mat * &x_vec_hat;
    (d, x_vec_hat)
}

// this can be a constant/lazy static
pub(crate) fn gadget_vec(log_q: u64) -> na::DVector<Wrapping<u64>> {
    let mut v = Vec::with_capacity(log_q as usize);
    for i in 0..log_q {
        v.push(Wrapping(1u64 << i));
    }
    na::DVector::from_vec(v)
}

fn compute_z(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk,
    s: na::DVectorView<Wrapping<u64>>,
    g_vec: na::DVectorView<Wrapping<u64>>,
) -> na::DMatrix<Wrapping<u64>> {
    let id_ml = na::DMatrix::identity(param.m * param.ell, param.m * param.ell);
    &pk.b_mat * id_ml.kronecker(&s.kronecker(&g_vec))
}

fn compute_z_prime(
    param: HalfNIOTEparam,
    y_delta_vec: na::DVectorView<Wrapping<u64>>,
    g_vec: na::DVectorView<Wrapping<u64>>,
) -> na::DMatrix<Wrapping<u64>> {
    let id_m = na::DMatrix::identity(param.m, param.m);
    id_m.kronecker(&y_delta_vec.transpose().kronecker(&g_vec))
}

// returns (c, phi=s)
pub fn bob_encode(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk,
    y_vec: na::DVectorView<Wrapping<u64>>,
    rng: &mut (impl Rng + CryptoRng),
) -> (na::DMatrix<Wrapping<u64>>, na::DVector<Wrapping<u64>>) {
    // B * (I_{m \ell} \otimes s \otimes g^T_q)
    let s_vec = na::DVector::<Wrapping<u64>>::new_random(param.k);
    debug_assert_eq!(s_vec.shape(), (param.k, 1)); // This means it is a column matrix
    let g_vec = gadget_vec(param.log_q);
    debug_assert_eq!(
        pk.b_mat.shape(),
        (param.n(), param.m * param.ell * param.n())
    );
    let z = compute_z(param, pk, s_vec.as_view(), g_vec.as_view());
    debug_assert_eq!(z.shape(), (param.n(), param.m * param.ell));

    // I_m \otimes (y * q/2)^T \otimes g^T_q
    let delta = Wrapping(1u64 << (param.log_q - 1)); // q/2
    let y_delta_vec = y_vec * delta;
    let z_prime = compute_z_prime(param, y_delta_vec.as_view(), g_vec.as_view());
    debug_assert_eq!(
        z_prime.shape(),
        (param.m * param.log_q as usize, param.m * param.ell)
    );

    // errors
    let e_prime = sample_discrete_gaussian(rng, param.stddev, param.n(), param.m * param.ell);
    let e_tilde = sample_discrete_gaussian(
        rng,
        param.stddev,
        param.m * param.log_q as usize,
        param.m * param.ell,
    );

    // put everything together
    let a_mat_t = pk.a_mat.transpose();
    let c_mat = &a_mat_t * z + a_mat_t * e_prime + e_tilde + z_prime;
    debug_assert_eq!(
        c_mat.shape(),
        (param.m * param.log_q as usize, param.m * param.ell)
    );

    // send c_mat to alice
    (c_mat, s_vec)
}

// returns v = C^T * x_hat
pub fn alice_decode(
    c_mat: na::DMatrixView<Wrapping<u64>>,
    psi: na::DVectorView<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    let x_vec_hat = psi;
    c_mat.transpose() * x_vec_hat
}

// returns w = - (I_{m \ell} \otimes s^T \otimes g_q) * B^T * d
pub fn bob_decode(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk,
    d_vec: na::DVectorView<Wrapping<u64>>,
    phi: na::DVectorView<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    let id_ml = na::DMatrix::identity(param.m * param.ell, param.m * param.ell);
    let s = phi;
    let g_vec_t = gadget_vec(param.log_q).transpose();

    // d =
    -(id_ml.kronecker(&s.transpose().kronecker(&g_vec_t)) * pk.b_mat.transpose() * d_vec)
}

pub fn reconstruct(
    param: HalfNIOTEparam,
    v: na::DVectorView<Wrapping<u64>>,
    w: na::DVectorView<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    (v + w).map(|x| {
        // we need to divide by q/2 and round
        // this is basically the same as shift right by 63
        Wrapping(x.0 >> (param.log_q - 1))
    })
}

#[cfg(test)]
mod tests {
    use proptest::{
        arbitrary::{StrategyFor, arbitrary_with},
        prelude::*,
        sample::size_range,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    use super::*;

    fn gen_vec_wrapping(m: usize) -> StrategyFor<Vec<Wrapping<u64>>> {
        arbitrary_with(size_range(m).lift())
    }

    fn gen_bool(m: usize) -> StrategyFor<Vec<bool>> {
        arbitrary_with(size_range(m).lift())
    }

    proptest! {
        #[test]
        fn test_protocol(xs in gen_vec_wrapping(4), ys in gen_bool(2)) {
            run_protocol(xs, ys);
        }
    }

    fn run_protocol(xs: Vec<Wrapping<u64>>, ys: Vec<bool>) {
        let param = HalfNIOTEparam {
            log_q: 64,
            m: xs.len(),
            ell: ys.len(),
            k: 2,
            stddev: 3.2,
        };

        let mut rng = ChaCha12Rng::from_os_rng();

        let pk = setup(param, &mut rng);

        let x_vec = na::DVector::from_vec(xs);
        let (d, psi) = alice_encode(&pk, x_vec.as_view());

        let y_vec = na::DVector::from_vec(ys.into_iter().map(|y| Wrapping(y as u64)).collect());
        let (c, phi) = bob_encode(param, &pk, y_vec.as_view(), &mut rng);
        assert_eq!(phi.shape(), (param.k, 1));

        let v = alice_decode(c.as_view(), psi.as_view());
        let w = bob_decode(param, &pk, d.as_view(), phi.as_view());
        assert_eq!(v.shape(), (param.m * param.ell, 1));
        assert_eq!(v.shape(), w.shape());

        let res = reconstruct(param, v.as_view(), w.as_view());
        assert_eq!(v.shape(), res.shape());

        // round(v + m)_p = x \otimes y mod p
        let expected = x_vec.kronecker(&y_vec).map(|x| Wrapping(x.0 % 2));
        assert_eq!(res, expected);
    }
}
