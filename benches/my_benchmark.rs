use criterion::{Criterion, criterion_group, criterion_main};
use mp_dpf::{half_niote, half_niote::HalfNIOTEparam, noise::Noise, ring::Moduli};
use nalgebra as na;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::{hint::black_box, num::Wrapping};

fn half_niote_bob_encode(c: &mut Criterion) {
    let xs = [Wrapping(12u64); 4];
    let ys = vec![Wrapping(1u64); 64];
    let param = HalfNIOTEparam {
        moduli: Moduli::new(64, 1),
        m: xs.len(),
        ell: ys.len(),
        k: 2,
        noise: Noise::default(),
        decomposed_input: false,
    };

    let mut rng = ChaCha12Rng::from_os_rng();

    let pk = half_niote::setup::<Wrapping<u64>>(param, &mut rng);

    // let x_vec = na::DVector::from_vec(xs);
    // let (d, psi) = half_niote::alice_encode(param, &pk, x_vec.as_view());

    let y_vec = na::DVector::from_vec(ys);
    let y_vec_view = y_vec.as_view();

    c.bench_function("half niote bob encode", |b| {
        b.iter(|| {
            black_box(half_niote::bob_encode(param, &pk, y_vec_view, &mut rng));
        })
    });
}

criterion_group!(benches, half_niote_bob_encode);
criterion_main!(benches);
