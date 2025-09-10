use std::num::Wrapping;

use nalgebra as na;
use rand::{CryptoRng, Rng};

use crate::half_niote::{self, HalfNIOTEparam, NIOTEpk, gadget_vec};

#[derive(Clone, Copy, Debug)]
pub struct NIOTEparam {
    half: HalfNIOTEparam,
    t: usize,
    r: usize,
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
}

pub fn setup<R: Rng + CryptoRng>(
    param: NIOTEparam,
    rng: &mut R,
) -> (NIOTEpk, na::DMatrix<Wrapping<u64>>) {
    let pk = half_niote::setup(param.half, rng);

    let ident_tn = na::DMatrix::identity(param.t * param.n(), param.t * param.n());
    let g = gadget_vec(param.log_q());
    let p_mat = &pk.a_mat * (ident_tn.kronecker(&g));

    (pk, p_mat)
}

pub fn alice_encode(
    param: NIOTEparam,
    a_mat: &na::DMatrix<Wrapping<u64>>,
    x_vec: &na::DVector<Wrapping<u64>>,
) -> (
    na::DVector<Wrapping<u64>>,
    Vec<Vec<na::DVector<Wrapping<u64>>>>,
) {
    let mut x_i = x_vec.clone();
    let mut psi_out = vec![];
    for i in 0..param.r {
        let step = param.n() * param.t;
        let j_len = param.t.pow((param.r - i - 1) as u32);
        let mut x_ip1 = na::DVector::zeros(j_len);
        let mut psi_i = Vec::with_capacity(j_len);

        for j in 0..j_len {
            let x_i_view = x_i.rows(j * step, step);
            let (d_ij, psi_ij) = half_niote::alice_encode(a_mat.as_view(), x_i_view);
            psi_i.push(psi_ij);

            // TODO reduce data movement by using mul_to in `half_niote::alice_encode` and write the result to mut view
            let mut x_ip1_view = x_ip1.rows_mut(j * step, step);
            x_ip1_view.copy_from(&d_ij);
        }

        x_i = x_ip1;
        psi_out.push(psi_i);
    }

    (x_i, psi_out)
}
