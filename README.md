# mp-dpf

Implementation of the succinct non-interactive oblivious tensor evaluation (NIOTE) from
Appendix A (Figures 12–14) of [eprint 2026/1942](https://eprint.iacr.org/2026/1942), which
builds on [AMR25](https://eprint.iacr.org/2025/336) (Constructions 3.4 and 3.6).

- `src/half_niote.rs`: the half-succinct NIOTE (Figure 12).
- `src/full_niote.rs`: the succinct NIOTE, which bootstraps the half-succinct NIOTE along a
  t-ary tree of depth r (Figures 13 and 14). The root uses the half-succinct NIOTE instead
  of a hasher-private OTE, so the digest is not hasher-private.
- `src/ring.rs`: arithmetic over Z_q with q = 2^log_q, generic over the storage word
  (`Wrapping<u8>` up to `Wrapping<u128>`), gadget decomposition, rounding and lifting.
- `src/noise.rs`: the noise distribution: t-uniform (the default) or a truncated
  discrete Gaussian.

Two extensions of the paper are parameters:

- **Gadget base** `b = 2^log_b` (`Moduli::with_gadget_base`): digests and secrets are
  decomposed into `⌈log_q / log_b⌉` digits in `[0, b)` instead of bits. `log_b = 1` is
  exactly the paper. The paper suggests this optimisation (Section 3.1, "On the Efficiency
  of NIOTE"). Its constants in Lemma 5, Corollary 2 and Theorem 4 change: the noise bound
  gains a factor `b - 1`, and the lifted values are at most `(b - 1)^2` instead of 1. The
  security argument is unchanged, since `A` stays binary and `n = k·log_q`.
- **Noise** `Noise::TUniform { log_bound }` (the default, with `log_bound = 0`) or
  `Noise::DiscreteGaussian { stddev }` (truncated at 6σ).

## Tests

```bash
cargo test --release
```

The full NIOTE is tested end to end without noise with `p = q`, where the protocol is
exact, for several word sizes and gadget bases. Noise is tested in the half-succinct
NIOTE, in one decoding step of Figure 14 (`full_niote::tests::test_level_with_noise`), and
end to end in `full_niote::tests::test_protocol_noise_end_to_end`. The last one runs the
full protocol with noise, `p < q`, and rounding and lifting at every level. It uses the
smallest such configuration that fits in memory (`k = 1`, `log_q = 127`, `log_b = 12`,
failure probability about 2^-20), which checks correctness only and has no security. It
takes about 9 s and 1.5 GB, and it is skipped in debug builds, where it is slow.

## Parameters

`scripts/param_search.py` searches for parameters that are both secure and correct, and
minimise the encoding size. It iterates two constraints until they agree:

- **Correctness:** the smallest `q` (with the best `p`) such that the failure probability
  is at most 2^-40. This is a union bound over all local roundings (Lemma 5) and all
  liftings (Corollary 2) with base-b constants. It is the same bound as
  `NIOTEparam::log2_failure_bound`.
- **Security:** the smallest LWE dimension `k` with 128 bits against primal uSVP, under the
  [lattice estimator](https://github.com/malb/lattice-estimator)'s default cost model
  (the same criterion as `parameters.sage` in the paper repository). The secret is uniform
  mod q, as in Figure 12.

```bash
PYTHONPATH=/path/to/lattice-estimator sage -python scripts/param_search.py 0,1,2,4 1,4,8,16,24,32,40,48
cargo run --release --example params   # re-checks the table below with the Rust bound
```

The search is a fixed point because correctness makes `q` depend on `k` (the noise bound
grows with `n = k·log_q`), while security makes `k` depend on `q`.

Results for the smallest tree (t = 2, r = 2, ℓ = 1) and TUniform(0) noise. `m = t^r·n` is
the length of Alice's input. Sizes assume `log_q` bits per element.

| log_b | k | log q | log p | log₂ fail | uSVP | n = k·log q | m | encoding | B matrix | digest |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (paper) | 9,089 | 236 | 93 | −40.1 | 128.1 | 2.1M | 8.6M | 55,000 ZiB | 1.1M ZiB | 60 MiB |
| 4 | 9,128 | 237 | 96 | −40.1 | 128.1 | 2.2M | 8.7M | 926 ZiB | 70,000 ZiB | 61 MiB |
| 8 | 9,434 | 245 | 102 | −40.1 | 128.1 | 2.3M | 9.2M | 156 ZiB | 24,000 ZiB | 68 MiB |
| 16 | 10,227 | 266 | 117 | −40.2 | 128.0 | 2.7M | 10.9M | 42 ZiB | 13,000 ZiB | 86 MiB |
| 24 | 11,097 | 289 | 133 | −40.0 | 128.1 | 3.2M | 12.8M | 31 ZiB | 13,000 ZiB | 110 MiB |
| **32** | **12,014** | **313** | **148** | **−40.4** | **128.0** | **3.8M** | **15.0M** | **22.5 ZiB** | **13,500 ZiB** | **140 MiB** |
| 40 | 12,906 | 337 | 164 | −40.3 | 128.0 | 4.3M | 17.4M | 25 ZiB | 18,000 ZiB | 175 MiB |
| 48 | 13,821 | 361 | 180 | −40.2 | 128.1 | 5.0M | 20.0M | 27 ZiB | 23,000 ZiB | 215 MiB |

### Findings

- **The unpacked construction is not practical at 128-bit security.** Even the best
  parameters need an encoding of about 22.5 ZiB (1 ZiB = 2^70 bytes). The inner encoding
  has `(t·n·ℓ_b)²·k·ℓ_b` elements with `n = k·log_q` in the millions, where ℓ_b is the
  number of digits. A ring/module-LWE instantiation, which the paper leaves to future work,
  is needed.
- **A larger gadget base helps up to `log_b ≈ 32`**: about 2,400× smaller than `log_b = 1`.
  Beyond that, `q` grows by about 3 bits per bit of `log_b` (β for the noise, 2β for
  lifting), which outweighs the smaller number of digits. `p` also grows by about 2·log_b
  bits.
- **The noise width barely matters.** TUniform(0) through TUniform(4) differ by less than
  1% in size. A wider noise lowers `k` slightly but raises `q` by about the same amount.
  TUniform(0) is marginally best and is the default. It has an exact bound (B = 1), whereas
  a Gaussian has to be truncated at many σ.
- **B must be generated from a seed.** At realistic `k`, the public-key matrix `B` is
  about `k/(t·ℓ_b)` times larger than the encoding (13,500 ZiB at `log_b = 32`). It is
  uniformly random, so it can be expanded from a short seed (not implemented yet).
- **`q` is larger than 128 bits** for every secure parameter set, so a wider word type
  than `u128` is needed.

### Caveats

- Security is primal uSVP only, as in `parameters.sage`. In spot checks, BDD was about
  0.4 bits below uSVP, so rows at exactly 128.0 are marginal. Arora–GB and BKW are
  excluded, as in `parameters.sage`. Arora–GB may be relevant for such narrow bounded
  noise: an early estimate gave it about 116 bits at k = 8192 with TUniform(1). It was not
  checked at the parameters above.
- The paper's root is a hasher-private OTE, which is modelled here as the half-succinct
  NIOTE, as in the code.
- The paper's Lemmas 5 and 9 and Theorem 4 state the constants for `log_b = 1` only.

## Benchmark

### Linux

Make sure flamegraph is installed: `cargo install flamegraph`.

### macOS

Make sure samply is installed: `cargo install samply`.

```bash
samply record target/release/deps/my_benchmark-b988e4327dc7f9b6 'half niote bob encode' --bench --profile-time 60
```

You can find the executable name when running the benchmark normally.
