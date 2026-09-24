//! Prints the smallest log q (and the best log p for it) such that the succinct NIOTE
//! is correct except with probability 2^-40, together with the encoding size.
//!
//! cargo run --release --example params

use mp_dpf::{full_niote::NIOTEparam, ring::Moduli};

const TARGET_LOG2_FAILURE: f64 = -40.0;
const STDDEV: f64 = 3.2;

fn best_log_p(log_q: u32, k: usize, t: usize, r: usize, ell: usize) -> Option<(u32, f64)> {
    (1..log_q)
        .map(|log_p| {
            let param = NIOTEparam {
                moduli: Moduli::new(log_q, log_p),
                k,
                t,
                r,
                ell,
                stddev: STDDEV,
            };
            (log_p, param.log2_failure_bound())
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

fn main() {
    println!("stddev = {STDDEV}, target failure <= 2^{TARGET_LOG2_FAILURE}");
    println!(
        "{:>3} {:>3} {:>3} {:>4} | {:>6} {:>6} {:>10} | {:>14}",
        "k", "t", "r", "ell", "log_q", "log_p", "log2 fail", "encoding (GiB)"
    );
    for k in [1, 2, 4] {
        for t in [2, 4] {
            for r in [2, 3] {
                let ell = 1;
                let found = (8..=256).find_map(|log_q| {
                    best_log_p(log_q, k, t, r, ell)
                        .filter(|(_, fail)| *fail <= TARGET_LOG2_FAILURE)
                        .map(|(log_p, fail)| (log_q, log_p, fail))
                });
                let Some((log_q, log_p, fail)) = found else {
                    println!("{k:>3} {t:>3} {r:>3} {ell:>4} | none up to log_q = 256");
                    continue;
                };
                let param = NIOTEparam {
                    moduli: Moduli::new(log_q, log_p),
                    k,
                    t,
                    r,
                    ell,
                    stddev: STDDEV,
                };
                let word_bytes = log_q.div_ceil(8).next_power_of_two() as f64;
                let gib = param.encoding_len() as f64 * word_bytes / (1u64 << 30) as f64;
                println!(
                    "{k:>3} {t:>3} {r:>3} {ell:>4} | {log_q:>6} {log_p:>6} {fail:>10.1} | {gib:>14.1}"
                );
            }
        }
    }
}
