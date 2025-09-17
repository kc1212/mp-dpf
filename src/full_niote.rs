use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::{
    Scalar,
    half_niote::{self, HalfNIOTEparam, HalfNIOTEpk, gadget_vec},
};

#[derive(Clone, Copy, Debug)]
pub struct NIOTEparam {
    pub half: HalfNIOTEparam,
    pub t: usize,
    pub r: usize,
}

impl NIOTEparam {
    pub fn is_consistent(&self) -> bool {
        self.half.m() == self.half.n() * self.t.pow(self.r as u32)
    }

    pub fn n(&self) -> usize {
        self.half.n()
    }

    pub fn log_q(&self) -> u64 {
        self.half.log_q()
    }

    pub fn m(&self) -> usize {
        self.half.m()
    }

    pub fn ell(&self) -> usize {
        self.half.ell()
    }

    pub fn k(&self) -> usize {
        self.half.k()
    }

    pub fn t_pow_r_minus_i_1(&self, i: usize) -> usize {
        self.t.pow((self.r - i - 1) as u32)
    }
}

pub struct NIOTEpk<S: Scalar> {
    half: HalfNIOTEpk<S>,
    p_mat_prods: Vec<na::DMatrix<S>>,
}

pub fn setup<R: Rng + CryptoRng, S: Scalar>(param: NIOTEparam, rng: &mut R) -> NIOTEpk<S> {
    let pk = half_niote::setup(param.half, rng);

    // TODO how to come up with p_mat?
    let p_mat: na::DMatrix<S> =
        na::DMatrix::identity(param.n() * param.n(), param.t * param.n() * param.n());
    /*
    // nt^r
    let nt_to_r = param.t.pow(param.r as u32) * param.n();
    let ident_tnr: na::DMatrix<S> = na::DMatrix::identity(nt_to_r, nt_to_r);
    let g = gadget_vec::<S>();
    let p_mat: na::DMatrix<S> = &pk.a_mat * (ident_tnr.kronecker(&g));
    assert_eq!(pk.a_mat.shape(), (param.n(), nt_to_r * S::BIT_LENGTH));
    */

    // let ident_logq: na::DMatrix<S> = na::DMatrix::identity(S::BIT_LENGTH, S::BIT_LENGTH);
    // let p_mat: na::DMatrix<S> = pk.a_mat.kronecker(&ident_logq);
    let p_mat_prods = scan_p_mat_prods(param, p_mat.as_view());

    #[cfg(test)]
    {
        for (i, p) in p_mat_prods.iter().enumerate() {
            println!("p dims: {i}, {:?}", p.shape())
        }
    }

    NIOTEpk {
        half: pk,
        p_mat_prods,
    }
}

// output (d, psi)
#[allow(clippy::type_complexity)]
pub fn alice_encode<S: Scalar>(
    param: NIOTEparam,
    pk: &NIOTEpk<S>,
    x_vec: na::DVectorView<S>,
) -> (na::DVector<S>, Vec<Vec<na::DVector<S>>>) {
    let mut x_i = x_vec.clone_owned();
    let mut psi_out = vec![];
    for i in 0..param.r {
        let step = param.n() * param.t;
        let j_len = param.t_pow_r_minus_i_1(i);
        // x_{i+1} should have length t^{r - i - 1} \cdot n
        let mut x_ip1 = na::DVector::zeros(j_len * param.n());
        let mut psi_i = Vec::with_capacity(j_len);

        for j in 0..j_len {
            // x_i is a column vector, consider x_ij as the vertical conccatenation of the column vector
            let x_i_view = x_i.rows(j * step, step);
            debug_assert_eq!(x_i_view.shape(), (step, 1));

            let (d_ij, psi_ij) = half_niote::alice_encode(&pk.half, x_i_view);
            psi_i.push(psi_ij);

            // TODO reduce data movement by using mul_to in `half_niote::alice_encode` and write the result to mut view
            let mut x_ip1_view = x_ip1.rows_mut(j * param.n(), param.n());
            x_ip1_view.copy_from(&d_ij);
        }

        x_i = x_ip1;
        psi_out.push(psi_i);
    }

    (x_i, psi_out)
}

// output { E_i }, phi
pub fn bob_encode<S: Scalar>(
    param: NIOTEparam,
    pk: &NIOTEpk<S>,
    y_vec: na::DVectorView<S>,
    rng: &mut (impl Rng + CryptoRng),
) -> (Vec<na::DMatrix<S>>, na::DVector<S>)
where
    rand::distr::StandardUniform: rand::distr::Distribution<S>,
{
    let mut e_out = Vec::with_capacity(param.r);
    // the first iteration is different from the rest in terms of dimension, so we do it separately
    let (e_mat_0, phi_1) = half_niote::bob_encode(param.half, &pk.half, y_vec, rng);

    // note that usually for hOTE the output dimension is (m log q, m l)
    // but here we've instantiated hOTE with size m = nt
    debug_assert_eq!(
        e_mat_0.shape(),
        (
            param.t * param.n() * param.log_q() as usize,
            param.t * param.n() * param.ell()
        )
    );
    e_out.push(e_mat_0);

    if param.r == 1 {
        return (e_out, phi_1);
    }

    let g = gadget_vec::<S>();
    let mut y_i = phi_1.kronecker(&g);

    for i in 1..param.r {
        debug_assert_eq!(y_i.shape(), (param.log_q() as usize * param.k(), 1));
        let (e_mat_i, phi_ip1) = half_niote::bob_encode(param.half, &pk.half, y_i.as_view(), rng);
        debug_assert_eq!(
            e_mat_i.shape(),
            (
                param.t * param.n() * param.log_q() as usize,
                param.t * param.n() * param.ell()
            )
        );

        e_out.push(e_mat_i);

        if i == param.r - 1 {
            return (e_out, phi_ip1);
        }

        y_i = phi_ip1.kronecker(&g);
    }

    unreachable!()
}

// compute
// \{  \prod_i=1^k P_i  \} for k \in [r]
// where P_i = I_{t^{r-i-1}} \otimes P
fn scan_p_mat_prods<S: Scalar>(
    param: NIOTEparam,
    p_mat: na::DMatrixView<S>,
) -> Vec<na::DMatrix<S>> {
    // P: n^2 \times tn^2
    assert_eq!(
        p_mat.shape(),
        (param.n() * param.n(), param.n() * param.n() * param.t)
    );

    // P_i will have dimension (n*t^{r - 1 - i}, n*t^{r - i})
    // below we remove the factor n for brevity, so the above becomes (t^{r - 1 - i}, t^{r - i})
    // P_0               : (t^{r - 1}, t^r)
    // P_1 * P_0         : (t^{r - 2}, t^{r - 1}) * (t^{r - 1}, t^{r})
    //                   : (t^{r - 2}, t^{r})
    // P_2 * P_1 * P_0   : (t^{r - 3}, t^{r - 2}), (t^{r - 2}, t^{r})
    //                   : (t^{r - 3}, t^{r})
    // P_{r-1} *...* P_0 : (t^0, t^r)
    let mut p_mats = (0..param.r).map(|i| {
        let t_pow = param.t_pow_r_minus_i_1(i);
        // let t_pow = param.t.pow((param.r - i) as u32);
        let id = na::DMatrix::identity(t_pow, t_pow);
        id.kronecker(&p_mat)
    });

    // fill the accummulator with the first element
    let mut acc: na::DMatrix<S> = p_mats.next().unwrap();
    let mut steps = Vec::with_capacity(param.r);
    steps.push(acc.clone());

    // fill the rest
    for p_mat_i in p_mats {
        // dimension of acc will reduce on every iteration
        acc = p_mat_i * acc;
        steps.push(acc.clone());
    }
    steps
}

// output v_r
pub fn alice_decode<S: Scalar>(
    param: NIOTEparam,
    pk: &NIOTEpk<S>,
    e_mats: &[na::DMatrix<S>],
    psi: &[Vec<na::DVector<S>>],
) -> na::DVector<S> {
    assert_eq!(e_mats.len(), param.r);
    let mut acc = na::DVector::zeros(param.n() * param.n() * param.t.pow(param.r as u32));

    // we start at the bottom level (the leaves of the hash tree) at i = 0
    // the top level is at i = r-1
    for i in 0..param.r {
        let t_pow = param.t_pow_r_minus_i_1(i);

        // suppose e_mats[i] has dimension (tn \log q) x (tn \ell)
        // internally a mat-vec multiplication happens on transpose of e_mats[i] in hOTE.alice_decode
        // so the resulting vector has length tn \ell
        // also there are t^{r - i - 1} of those vectors
        let chunk_size = e_mats[i].shape().1; // this is tn \ell
        let v_i_len = chunk_size * t_pow;
        let mut v_i = na::DVector::zeros(v_i_len);

        for j in 0..t_pow {
            let v_ij = half_niote::alice_decode(e_mats[i].as_view(), psi[i][j].as_view());
            debug_assert_eq!(v_ij.shape(), (chunk_size, 1));
            v_i.rows_mut(chunk_size * j, chunk_size).copy_from(&v_ij);
        }

        if i == 0 {
            acc += v_i;
        } else {
            let p_mat_prod = &pk.p_mat_prods[i - 1];
            acc += (v_i.transpose() * p_mat_prod).transpose();
        }
    }

    // debug_assert_eq!(acc.shape(), (param.m() * param.ell(), 1));
    acc
}

pub fn bob_decode<S: Scalar>(
    param: NIOTEparam,
    pk: &NIOTEpk<S>,
    d_vec: na::DVectorView<S>,
    phi: na::DVectorView<S>,
) -> na::DVector<S> {
    let w_r_minus_1 = half_niote::bob_decode(param.half, &pk.half, d_vec, phi);
    if param.r == 1 {
        w_r_minus_1
    } else {
        (w_r_minus_1.transpose() * &pk.p_mat_prods[param.r - 2]).transpose()
    }
}

pub fn reconstruct<S: Scalar>(
    param: NIOTEparam,
    v: na::DVectorView<S>,
    w: na::DVectorView<S>,
) -> na::DVector<S> {
    half_niote::reconstruct(param.half, v, w)
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    use super::*;

    fn run_protocol<S: Scalar>(k: usize, t: usize, r: usize, xs: Vec<S>, ys: Vec<bool>)
    where
        rand::distr::StandardUniform: rand::distr::Distribution<S>,
    {
        // if we only want one instance of hOTE, then we need to ensure that \ell = k \log q
        let ell = ys.len();
        let log_q = S::BIT_LENGTH as u64;

        // m = t^r n
        let n = xs.len() / t.pow(r as u32);
        assert_eq!(n, ell);
        let half = HalfNIOTEparam {
            log_q,
            m: n * t,
            ell,
            k,
            stddev: 0.,
        };
        let param = NIOTEparam { half, t, r };
        println!("[test] full niote with param: {param:?}");

        let mut rng = ChaCha12Rng::from_os_rng();
        let pk = setup(param, &mut rng);

        let x_vec = na::DVector::from_vec(xs);
        let (d_vec, psi) = alice_encode(param, &pk, x_vec.as_view());

        let y_vec = na::DVector::from_vec(ys.into_iter().map(|y| S::from_u64(y as u64)).collect());
        let (e_mats, phi) = bob_encode(param, &pk, y_vec.as_view(), &mut rng);

        let v_vec = alice_decode(param, &pk, &e_mats, &psi);
        let w_vec = bob_decode(param, &pk, d_vec.as_view(), phi.as_view());

        let actual = reconstruct(param, v_vec.as_view(), w_vec.as_view());
        let expected = x_vec.kronecker(&y_vec).map(|x| S::from_u64(x.as_u64() % 2));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_protocol_depth1() {
        type MyScalar = Wrapping<u16>;
        let t = 2;
        let r = 1;
        let k = 1; // normally k is big
        let n = k * MyScalar::BIT_LENGTH;
        run_protocol::<MyScalar>(
            k,
            t,
            r,
            vec![Wrapping(1); t.pow(r as u32) * n],
            vec![true; n],
        );
    }

    #[test]
    fn test_protocol_depth2() {
        type MyScalar = Wrapping<u16>;
        let t = 2;
        let r = 2;
        let k = 1; // normally k is big
        let n = k * MyScalar::BIT_LENGTH;
        run_protocol::<MyScalar>(
            k,
            t,
            r,
            vec![Wrapping(1); t.pow(r as u32) * n],
            vec![true; n],
        );
    }
}
