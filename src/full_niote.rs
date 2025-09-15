use std::num::Wrapping;

use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::half_niote::{self, HalfNIOTEparam, HalfNIOTEpk, gadget_vec};

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

pub struct NIOTEpk {
    half: HalfNIOTEpk,
    p_mat_prods: Vec<na::DMatrix<Wrapping<u64>>>,
}

pub fn setup<R: Rng + CryptoRng>(param: NIOTEparam, rng: &mut R) -> NIOTEpk {
    let pk = half_niote::setup(param.half, rng);

    let ident_tn = na::DMatrix::identity(param.t * param.n(), param.t * param.n());
    let g = gadget_vec(param.log_q());
    let p_mat = &pk.a_mat * (ident_tn.kronecker(&g));
    let p_mat_prods = scan_p_mat_prods(param, p_mat.as_view());

    NIOTEpk {
        half: pk,
        p_mat_prods,
    }
}

// output (d, psi)
#[allow(clippy::type_complexity)]
pub fn alice_encode(
    param: NIOTEparam,
    pk: &NIOTEpk,
    x_vec: na::DVectorView<Wrapping<u64>>,
) -> (
    na::DVector<Wrapping<u64>>,
    Vec<Vec<na::DVector<Wrapping<u64>>>>,
) {
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
pub fn bob_encode(
    param: NIOTEparam,
    pk: &NIOTEpk,
    y_vec: na::DVectorView<Wrapping<u64>>,
    rng: &mut (impl Rng + CryptoRng),
) -> (Vec<na::DMatrix<Wrapping<u64>>>, na::DVector<Wrapping<u64>>) {
    let mut e_out = Vec::with_capacity(param.r);
    // the first iteration is different from the rest in terms of dimension, so we do it separately
    let (e_mat_0, phi_1) = half_niote::bob_encode(param.half, &pk.half, y_vec, rng);
    debug_assert_eq!(
        e_mat_0.shape(),
        (param.t * param.n(), param.t * param.n() * param.ell())
    );
    e_out.push(e_mat_0);

    if param.r == 1 {
        return (e_out, phi_1);
    }

    let g = gadget_vec(param.log_q());
    let mut y_i = phi_1.kronecker(&g);

    // TODO maybe there's a more elegant way of writing the logic
    // (1..param.r).fold(y_i, |acc, _| {
    //         let (e_mat_i, phi_ip1) = half_niote::bob_encode(param.half, pk, y_i.as_view(), rng);
    //         phi_ip1.kronecker(&g)
    // });
    for i in 1..param.r {
        debug_assert_eq!(y_i.shape(), (param.log_q() as usize * param.k(), 1));
        let (e_mat_i, phi_ip1) = half_niote::bob_encode(param.half, &pk.half, y_i.as_view(), rng);
        debug_assert_eq!(
            e_mat_i.shape(),
            (param.t * param.n(), (param.t * param.n()).pow(2))
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
fn scan_p_mat_prods(
    param: NIOTEparam,
    p_mat: na::DMatrixView<Wrapping<u64>>,
) -> Vec<na::DMatrix<Wrapping<u64>>> {
    // P: n \times tn
    assert_eq!(p_mat.shape(), (param.n(), param.t * param.n()));

    // P_i will have dimension (n*t^{r - 1 - i}, n*t^{r - i})
    // below we remove the factor n for brevity, so the above becomes (t^{r - 1 - i}, t^{r - i})
    // P_1 * P_0         : (t^{r - 2}, t^{r - 1}) * (t^{r - 1}, n*t^{r})
    //                   : (t^{r - 2}, t^{r})
    // P_2 * P_1 * P_0   : (t^{r - 3}, t^{r - 2}), (t^{r - 2}, t^{r})
    //                   : (t^{r - 3}, t^{r})
    // P_{r-1} *...* P_0 : (t^0, t^r)
    let mut p_mats = (0..param.r).map(|i| {
        let t_pow = param.t_pow_r_minus_i_1(i);
        let id = na::DMatrix::identity(t_pow, t_pow);
        id.kronecker(&p_mat)
    });

    // fill the accummulator with the first element
    let mut acc: na::DMatrix<Wrapping<u64>> = p_mats.next().unwrap();
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
pub fn alice_decode(
    param: NIOTEparam,
    pk: &NIOTEpk,
    e_mats: &[na::DMatrix<Wrapping<u64>>],
    psi: &[Vec<na::DVector<Wrapping<u64>>>],
) -> na::DVector<Wrapping<u64>> {
    assert_eq!(e_mats.len(), param.r);
    let mut acc = na::DVector::zeros(param.m() * param.ell());

    // we write the algorithm a bit differently than what is the paper because
    // doing a lot of tensor products with identity matrix is not efficient
    for i in 0..param.r {
        let t_pow = param.t_pow_r_minus_i_1(i);

        // note: the length of e_mats[0] is different from the rest
        // suppose e_mats[i] has dimension (tn) x (tn \ell_i)
        // internally a mat-vec multiplication happens on transpose of e_mats[i]
        // so the resulting vector has length tn \ell_i
        // also there are t^{r - i - 1} of those vectors
        let chunk_size = e_mats[i].shape().1;
        let v_i_len = chunk_size * t_pow;
        let mut v_i = na::DVector::zeros(v_i_len);

        for j in 0..t_pow {
            let v_ij = half_niote::alice_decode(e_mats[i].as_view(), psi[i][j].as_view());
            debug_assert_eq!(v_ij.shape(), (chunk_size, 1));
            v_i.rows_mut(chunk_size * j, chunk_size).copy_from(&v_ij);
        }

        let p_mat_prod_i = &pk.p_mat_prods[i];
        acc += p_mat_prod_i * v_i;
    }

    debug_assert_eq!(acc.shape(), (param.m() * param.ell(), 1));
    acc
}

pub fn bob_decode(
    param: NIOTEparam,
    pk: &NIOTEpk,
    d_vec: na::DVectorView<Wrapping<u64>>,
    phi: na::DVectorView<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    let w_r_minus_1 = half_niote::bob_decode(param.half, &pk.half, d_vec, phi);
    &pk.p_mat_prods[param.r - 1] * w_r_minus_1
}

pub fn reconstruct(
    param: NIOTEparam,
    v: na::DVectorView<Wrapping<u64>>,
    w: na::DVectorView<Wrapping<u64>>,
) -> na::DVector<Wrapping<u64>> {
    half_niote::reconstruct(param.half, v, w)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    use super::*;

    fn run_protocol(k: usize, t: usize, r: usize, xs: Vec<Wrapping<u64>>, ys: Vec<bool>) {
        // if we only want one instance of hOTE, then we need to ensure that \ell = k \log q
        let ell = ys.len();
        let log_q = 64;
        assert_eq!(ell, k * log_q as usize);

        // m = t^r n
        let m = xs.len();
        let half = HalfNIOTEparam {
            log_q,
            m,
            ell,
            k,
            stddev: 3.2,
        };
        let param = NIOTEparam { half, t, r };

        let mut rng = ChaCha12Rng::from_os_rng();
        let pk = setup(param, &mut rng);

        let x_vec = na::DVector::from_vec(xs);
        let (d_vec, psi) = alice_encode(param, &pk, x_vec.as_view());

        let y_vec = na::DVector::from_vec(ys.into_iter().map(|y| Wrapping(y as u64)).collect());
        let (e_mats, phi) = bob_encode(param, &pk, y_vec.as_view(), &mut rng);

        let v_vec = alice_decode(param, &pk, &e_mats, &psi);
        let w_vec = bob_decode(param, &pk, d_vec.as_view(), phi.as_view());

        let _z = reconstruct(param, v_vec.as_view(), w_vec.as_view());
    }

    #[test]
    fn test_protocol() {
        let t = 2;
        let r = 1;
        let k = 1; // normally k is big
        let n = k * 64;
        run_protocol(
            k,
            t,
            r,
            vec![Wrapping(1u64); t.pow(r as u32) * n],
            vec![true; n],
        );
    }
}
