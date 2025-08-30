use std::num::Wrapping;

use bitvec::{order::Lsb0, view::BitView};
use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::sample_discrete_gaussian;

pub struct NIOTEpk {
    pub a_mat: na::DMatrix<Wrapping<u64>>,
    pub b_mat: na::DMatrix<Wrapping<u64>>,
}

#[derive(Clone, Copy, Debug)]
pub struct HalfNIOTEparam {
    log_q: u64, // fixed to 64
    m: usize,
    ell: usize,
    k: usize, // lwe dim
    stddev: f64,
}

impl HalfNIOTEparam {
    pub fn n(&self) -> usize {
        self.k * self.log_q as usize
    }
}

pub fn alice_setup<R: Rng + CryptoRng>(
    param: HalfNIOTEparam,
    _rng: &mut R,
) -> na::DMatrix<Wrapping<u64>> {
    // let a_matrix = na::DMatrix::<bool>::from_distribution_generic(
    //     nalgebra::Dyn(param.n),
    //     nalgebra::Dyn(param.m * param.log_q),
    //     &UniformRandomBool,
    //     rng,
    // );
    let a_matrix = na::DMatrix::<bool>::new_random(param.n(), param.m * param.log_q as usize);
    a_matrix.map(|x| if x { Wrapping(1u64) } else { Wrapping(0u64) })
}

pub fn bob_setup<R: Rng + CryptoRng>(
    param: HalfNIOTEparam,
    _rng: &mut R,
) -> na::DMatrix<Wrapping<u64>> {
    let b_matrix =
        na::DMatrix::<bool>::new_random(param.n(), param.m * param.ell * param.n() as usize);
    b_matrix.map(|x| if x { Wrapping(1u64) } else { Wrapping(0u64) })
}

fn bits(x_vec: &na::DVector<Wrapping<u64>>) -> na::DVector<Wrapping<u64>> {
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
    a_mat: &na::DMatrix<Wrapping<u64>>,
    x_vec: &na::DVector<Wrapping<u64>>,
) -> (na::DVector<Wrapping<u64>>, na::DVector<Wrapping<u64>>) {
    let x_vec_hat = bits(x_vec);
    debug_assert_eq!(x_vec_hat.len(), x_vec.len() * 64);
    let d = a_mat * &x_vec_hat;
    (d, x_vec_hat)
}

// this can be a constant/lazy static
fn gadget_vec(logq: u64) -> na::DVector<Wrapping<u64>> {
    let mut v = Vec::with_capacity(logq as usize);
    for i in 0..logq {
        v.push(Wrapping(1u64 << i));
    }
    na::DVector::from_vec(v)
}

fn gadget_mat(k: usize, logq: usize) -> na::DMatrix<Wrapping<u64>> {
    let g_vec = gadget_vec(logq as u64);
    let id_g = na::DMatrix::identity(k, k);
    id_g.kronecker(&g_vec)
}

// returns (c, phi=s)
pub fn bob_encode(
    param: &HalfNIOTEparam,
    pk: &NIOTEpk,
    y_vec: &na::DVector<Wrapping<u64>>,
    rng: &mut (impl Rng + CryptoRng),
) -> (na::DMatrix<Wrapping<u64>>, na::DVector<Wrapping<u64>>) {
    // B * (I_{m \ell} \otimes s \otimes g^T_q)
    let s = na::DVector::<Wrapping<u64>>::new_random(param.k);
    let id_ml = na::DMatrix::identity(param.m * param.ell, param.m * param.ell);
    let g_vec = gadget_vec(param.log_q);
    debug_assert_eq!(
        pk.b_mat.shape(),
        (param.n(), param.m * param.ell * param.n())
    );
    let z = &pk.b_mat * id_ml.kronecker(&s.kronecker(&g_vec));
    debug_assert_eq!(z.shape(), (param.n(), param.m * param.ell));

    // I_m \otimes (y * q/2)^T \otimes g^T_q
    let id_m = na::DMatrix::identity(param.m, param.m);
    let delta = Wrapping(1u64 << (param.log_q - 1)); // q/2
    let y_delta_vec = y_vec * delta;
    let z_prime = id_m.kronecker(&y_delta_vec.transpose().kronecker(&g_vec));
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
    (c_mat, s)
}

// returns v = C^T * x_hat
pub fn alice_decode(
    c_mat: &na::DMatrix<Wrapping<u64>>,
    psi: &na::DVector<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    let x_vec_hat = psi;
    c_mat.transpose() * x_vec_hat
}

// returns w = - (I_{m \ell} \otimes s^T \otimes g_q) * B^T * d
pub fn bob_decode(
    param: &HalfNIOTEparam,
    pk: &NIOTEpk,
    d_vec: &na::DVector<Wrapping<u64>>,
    phi: &na::DVector<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    let id_ml = na::DMatrix::identity(param.m * param.ell, param.m * param.ell);
    let s = phi;
    let g_vec_t = gadget_vec(param.log_q).transpose();

    // d =
    -(id_ml.kronecker(&s.transpose().kronecker(&g_vec_t)) * pk.b_mat.transpose() * d_vec)
}

pub fn reconstruct(
    param: &HalfNIOTEparam,
    v: &na::DVector<Wrapping<u64>>,
    w: &na::DVector<Wrapping<u64>>,
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

        let a_mat = alice_setup(param, &mut rng);
        let b_mat = bob_setup(param, &mut rng);
        let pk = NIOTEpk { a_mat, b_mat };

        let x_vec = na::DVector::from_vec(xs);
        let (d, psi) = alice_encode(&pk.a_mat, &x_vec);

        let y_vec = na::DVector::from_vec(ys.into_iter().map(|y| Wrapping(y as u64)).collect());
        let (c, phi) = bob_encode(&param, &pk, &y_vec, &mut rng);

        let v = alice_decode(&c, &psi);
        let w = bob_decode(&param, &pk, &d, &phi);
        assert_eq!(v.shape(), (param.m * param.ell, 1));
        assert_eq!(v.shape(), w.shape());

        let res = reconstruct(&param, &v, &w);
        assert_eq!(v.shape(), res.shape());

        // round(v + m)_p = x \otimes y mod p
        let expected = x_vec.kronecker(&y_vec).map(|x| Wrapping(x.0 % 2));
        assert_eq!(res, expected);
    }
}
