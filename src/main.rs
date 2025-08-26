use std::num::Wrapping;

use nalgebra as na;

fn main() {
    let a_matrix = na::DMatrix::<Wrapping<u64>>::new_random(2, 2);
    let b_matrix = na::dmatrix![Wrapping(1u64), Wrapping(0); Wrapping(0), Wrapping(1)];
    let a_prime = a_matrix.transpose();
    let b_vec = na::dvector![Wrapping(5), Wrapping(6)];
    let ress = b_matrix.kronecker(&a_matrix);
    let zz = -ress;
    let _res = a_matrix * b_vec;
    println!("{} {}", zz, a_prime);
}
