//! Checks the 128-bit secure parameter sets from the README with the library's own
//! correctness bound (`NIOTEparam::log2_failure_bound`) and prints their sizes.
//!
//! The parameters were found with `scripts/param_search.py` (lattice estimator, 128-bit
//! security against primal uSVP under the default cost model, uniform secret) and a
//! correctness failure probability of at most 2^-40, for t = 2, r = 2, ell = 1 and
//! TUniform(0) noise.
//!
//! cargo run --release --example params

use mp_dpf::{full_niote::NIOTEparam, noise::Noise, ring::Moduli};

// (log_b, k, log_q, log_p)
const SECURE: [(u32, usize, u32, u32); 8] = [
    (1, 9089, 236, 93),
    (4, 9128, 237, 96),
    (8, 9434, 245, 102),
    (16, 10227, 266, 117),
    (24, 11097, 289, 133),
    (32, 12014, 313, 148),
    (40, 12906, 337, 164),
    (48, 13821, 361, 180),
];

fn human(bytes: f64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB"];
    let mut b = bytes;
    for u in units {
        if b < 1024.0 || u == "ZiB" {
            return format!("{b:.1} {u}");
        }
        b /= 1024.0;
    }
    unreachable!()
}

fn main() {
    println!(
        "{:>5} {:>6} {:>5} {:>5} | {:>9} | {:>12} {:>12} {:>10}",
        "log_b", "k", "log_q", "log_p", "log2 fail", "encoding", "B", "digest"
    );
    for (log_b, k, log_q, log_p) in SECURE {
        let param = NIOTEparam {
            moduli: Moduli::new(log_q, log_p).with_gadget_base(log_b),
            k,
            t: 2,
            r: 2,
            ell: 1,
            noise: Noise::default(),
        };
        assert!(param.lifting_is_sound());
        let fail = param.log2_failure_bound();
        assert!(fail <= -40.0, "{param:?}: 2^{fail}");

        let elem_bytes = log_q as f64 / 8.0;
        // the public key is dominated by B, which can be expanded from a seed
        let leaf = param.leaf_param();
        let inner = param.inner_param();
        let b_elems = (param.n() * k) as f64 * (leaf.out_len() + inner.out_len()) as f64;
        assert!(b_elems <= param.pk_len() as f64);
        println!(
            "{log_b:>5} {k:>6} {log_q:>5} {log_p:>5} | {fail:>9.1} | {:>12} {:>12} {:>10}",
            human(param.encoding_len() as f64 * elem_bytes),
            human(b_elems * elem_bytes),
            human(param.n() as f64 * elem_bytes),
        );
    }
}
