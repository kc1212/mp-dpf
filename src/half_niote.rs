use std::fmt::Debug;

use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::{LazyIdentityKronecker, Scalar, sample_discrete_gaussian};

pub struct HalfNIOTEpk<S: Scalar> {
    pub a_mat: na::DMatrix<S>,
    pub b_mat: na::DMatrix<S>,
}

#[derive(Clone, Copy, Debug)]
pub struct HalfNIOTEparam {
    pub log_q: u64,
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

pub fn setup<R: Rng + CryptoRng, S: Scalar>(param: HalfNIOTEparam, _rng: &mut R) -> HalfNIOTEpk<S> {
    // let a_mat = na::DMatrix::<bool>::from_distribution_generic(
    //     nalgebra::Dyn(param.n),
    //     nalgebra::Dyn(param.m * param.log_q),
    //     &UniformRandomBool,
    //     rng,
    // );
    let a_mat = na::DMatrix::<bool>::new_random(param.n(), param.m * param.log_q as usize);
    let a_mat = a_mat.map(|x| if x { S::one() } else { S::zero() });

    let b_mat = na::DMatrix::<bool>::new_random(param.n(), param.m * param.ell * param.n());
    let b_mat = b_mat.map(|x| if x { S::one() } else { S::zero() });

    HalfNIOTEpk { a_mat, b_mat }
}

fn bits<S: Scalar>(x_vec: na::DVectorView<S>) -> na::DVector<S> {
    let input_len = x_vec.len();
    // bits function
    let x_vec_hat = x_vec.iter().flat_map(|x| x.view_bits()).collect::<Vec<S>>();
    assert_eq!(x_vec_hat.len(), S::BIT_LENGTH * input_len);
    na::DVector::from_vec(x_vec_hat)
}

/// returns (d, psi = x_hat)
pub fn alice_encode<S: Scalar>(
    pk: &HalfNIOTEpk<S>,
    x_vec: na::DVectorView<S>,
) -> (na::DVector<S>, na::DVector<S>) {
    let a_mat: na::DMatrixView<S> = pk.a_mat.as_view();
    let x_vec_hat = bits(x_vec);
    debug_assert_eq!(x_vec_hat.len(), x_vec.len() * S::BIT_LENGTH);
    let d = a_mat * &x_vec_hat;
    (d, x_vec_hat)
}

// this can be a constant/lazy static
pub(crate) fn gadget_vec<S: Scalar>() -> na::DVector<S> {
    let log_q = S::BIT_LENGTH;
    let mut v = Vec::with_capacity(log_q as usize);
    for i in 0..log_q {
        v.push(S::from_u64(1u64 << i));
    }
    na::DVector::from_vec(v)
}

fn compute_z<S: Scalar>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<S>,
    s: na::DVectorView<S>,
    g_vec: na::DVectorView<S>,
) -> na::DMatrix<S> {
    // let id_ml = na::DMatrix::identity(param.m * param.ell, param.m * param.ell);
    // &pk.b_mat * id_ml.kronecker(&s.kronecker(&g_vec))
    // NOTE: compared to the above, we have about −96.639% improvement
    let s_mul_g = s.kronecker(&g_vec);
    pk.b_mat.as_view() * LazyIdentityKronecker::new(param.m * param.ell, s_mul_g.as_view())
}

// fn compute_z_prime(
//     param: HalfNIOTEparam,
//     y_delta_vec: na::DVectorView<Wrapping<u64>>,
//     g_vec: na::DVectorView<Wrapping<u64>>,
// ) -> na::DMatrix<Wrapping<u64>> {
//     let id_m = na::DMatrix::identity(param.m, param.m);
//     id_m.kronecker(&y_delta_vec.transpose().kronecker(&g_vec))
// }

// returns (c, phi=s)
pub fn bob_encode<S: Scalar>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<S>,
    y_vec: na::DVectorView<S>,
    rng: &mut (impl Rng + CryptoRng),
) -> (na::DMatrix<S>, na::DVector<S>)
where
    rand::distr::StandardUniform: rand::distr::Distribution<S>,
{
    // B * (I_{m \ell} \otimes s \otimes g^T_q)
    let s_vec = na::DVector::<S>::new_random(param.k);
    debug_assert_eq!(s_vec.shape(), (param.k, 1)); // This means it is a column matrix
    let g_vec = gadget_vec::<S>();
    debug_assert_eq!(
        pk.b_mat.shape(),
        (param.n(), param.m * param.ell * param.n())
    );
    let z = compute_z(param, pk, s_vec.as_view(), g_vec.as_view());
    debug_assert_eq!(z.shape(), (param.n(), param.m * param.ell));

    // I_m \otimes (y * q/2)^T \otimes g^T_q
    let delta = S::from_u64(1u64 << (param.log_q - 1)); // q/2
    let y_delta_vec = y_vec * delta;

    // let z_prime = compute_z_prime(param, y_delta_vec.as_view(), g_vec.as_view());
    let y_kro_g = y_delta_vec.transpose().kronecker(&g_vec);
    let z_prime = LazyIdentityKronecker::new(param.m, y_kro_g.as_view());

    debug_assert_eq!(
        z_prime.shape(),
        (param.m * param.log_q as usize, param.m * param.ell)
    );

    // errors
    let e_prime =
        sample_discrete_gaussian::<_, S>(rng, param.stddev, param.n(), param.m * param.ell);
    let e_tilde = sample_discrete_gaussian::<_, S>(
        rng,
        param.stddev,
        param.m * param.log_q as usize,
        param.m * param.ell,
    );

    // put everything together
    let a_mat_t = pk.a_mat.transpose();
    let c_mat = &a_mat_t * z + a_mat_t * e_prime + (e_tilde.as_view() + z_prime);
    debug_assert_eq!(
        c_mat.shape(),
        (param.m * param.log_q as usize, param.m * param.ell)
    );

    // send c_mat to alice
    (c_mat, s_vec)
}

// returns v = C^T * x_hat
pub fn alice_decode<S: Scalar>(
    c_mat: na::DMatrixView<S>,
    psi: na::DVectorView<S>,
) -> na::DVector<S> {
    let x_vec_hat = psi;
    c_mat.transpose() * x_vec_hat
}

// returns w = - (I_{m \ell} \otimes s^T \otimes g_q) * B^T * d
pub fn bob_decode<S: Scalar>(
    param: HalfNIOTEparam,
    pk: &HalfNIOTEpk<S>,
    d_vec: na::DVectorView<S>,
    phi: na::DVectorView<S>,
) -> na::DVector<S> {
    let id_ml = na::DMatrix::identity(param.m * param.ell, param.m * param.ell);
    let s = phi;
    let g_vec_t = gadget_vec::<S>().transpose();

    // output d, note that this is not negated, we negate during reconstruction
    id_ml.kronecker(&s.transpose().kronecker(&g_vec_t)) * pk.b_mat.transpose() * d_vec
}

pub fn reconstruct<S: Scalar>(
    param: HalfNIOTEparam,
    v: na::DVectorView<S>,
    w: na::DVectorView<S>,
) -> na::DVector<S> {
    (v - w).map(|x| {
        // we need to divide by q/2 and round
        // this is basically the same as shift right by 63
        //
        // we don't have shl or shr implemented for Scalar
        // so manually convert to and from u64 to shift
        S::from_u64(x.as_u64() >> (param.log_q - 1))
    })
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

    fn gen_vec_wrapping(m: usize) -> StrategyFor<Vec<Wrapping<u64>>> {
        arbitrary_with(size_range(m).lift())
    }

    fn gen_bool(m: usize) -> StrategyFor<Vec<bool>> {
        arbitrary_with(size_range(m).lift())
    }

    proptest! {
        #[test]
        fn test_protocol(xs in gen_vec_wrapping(4), ys in gen_bool(2)) {
            run_protocol::<Wrapping<u64>>(xs, ys);
        }
    }

    fn run_protocol<S: Scalar>(xs: Vec<S>, ys: Vec<bool>)
    where
        rand::distr::StandardUniform: rand::distr::Distribution<S>,
    {
        let param = HalfNIOTEparam {
            log_q: S::BIT_LENGTH as u64,
            m: xs.len(),
            ell: ys.len(),
            k: 2,
            stddev: 3.2,
        };
        println!("[test] half niote with param: {param:?}");

        let mut rng = ChaCha12Rng::from_os_rng();

        let pk = setup(param, &mut rng);

        let x_vec = na::DVector::from_vec(xs);
        let (d, psi) = alice_encode::<S>(&pk, x_vec.as_view());

        let y_vec = na::DVector::from_vec(ys.into_iter().map(|y| S::from_u64(y as u64)).collect());
        let (c, phi) = bob_encode(param, &pk, y_vec.as_view(), &mut rng);
        assert_eq!(phi.shape(), (param.k, 1));

        // note that because the A matrix is not n by m anymore, but n by m log q,
        // that's why the dimension of C needs to be multiplied by log q
        assert_eq!(
            c.shape(),
            (param.m * param.log_q as usize, param.m * param.ell)
        );

        let v = alice_decode(c.as_view(), psi.as_view());
        let w = bob_decode(param, &pk, d.as_view(), phi.as_view());
        assert_eq!(v.shape(), (param.m * param.ell, 1));
        assert_eq!(v.shape(), w.shape());

        let actual = reconstruct(param, v.as_view(), w.as_view());
        assert_eq!(v.shape(), actual.shape());

        // round(v + m)_p = x \otimes y mod p
        let expected = x_vec.kronecker(&y_vec).map(|x| S::from_u64(x.as_u64() % 2));
        assert_eq!(actual, expected);
    }
}
