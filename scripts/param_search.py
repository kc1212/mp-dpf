"""Search for secure and correct parameters of the succinct NIOTE (eprint 2026/1942,
Figures 12-14) with a base-b = 2^log_b gadget, minimising the encoding size.

For each noise TUniform(e) and gadget base log_b, it iterates to a fixed point:
- correctness: the smallest q (and best p) such that the failure probability is at most
  2^FAIL, using Lemma 5 / Corollary 2 / Theorem 4 with base-b constants
  (the same bound as `NIOTEparam::log2_failure_bound` in the Rust code),
- security: the smallest LWE dimension k with LAMBDA bits against primal uSVP under the
  lattice estimator's default cost model (as in parameters.sage), with a uniform secret.

Usage (https://github.com/malb/lattice-estimator checked out in $ESTIMATOR):
    PYTHONPATH=$ESTIMATOR sage -python scripts/param_search.py 0,1,2 1,8,16,32
where the arguments are the comma-separated e's and log_b's. Prints one JSON line per row.
"""

import json
import math
import sys

from estimator import LWE, ND
from sage.all import log

LAMBDA = 128  # security target: log2 of the cost of primal uSVP
FAIL = -40  # correctness target: log2 of the failure probability
T, R, ELL = 2, 2, 1  # tree arity, tree depth, length of Bob's input

_kcache = {}


def security(k, log_q, e):
    p = LWE.Parameters(n=k, q=2**log_q, Xs=ND.UniformMod(2**log_q), Xe=ND.TUniform(e))
    return float(log(LWE.primal_usvp(p)["rop"], 2))


def min_k(log_q, e):
    """smallest k with security >= LAMBDA for q = 2^log_q and chi = TUniform(e)"""
    key = (log_q, e)
    if key not in _kcache:
        # exponential search for an upper bound, then binary search
        lo, hi = 256, 512
        while security(hi, log_q, e) < LAMBDA:
            lo, hi = hi, 2 * hi
        while lo < hi:
            mid = (lo + hi) // 2
            if security(mid, log_q, e) < LAMBDA:
                lo = mid + 1
            else:
                hi = mid
        _kcache[key] = lo
    return _kcache[key]


def sizes(log_q, log_b, k):
    digits = math.ceil(log_q / log_b)
    n = k * log_q  # digest length, A is binary
    m_hat = T * n * digits  # Alice's decomposed input per block (leaf and inner)
    inner_out = m_hat * k * digits
    leaf_out = T * n * ELL
    return n, m_hat, inner_out, leaf_out


def log2_fail(log_q, log_p, log_b, k, bound):
    n, m_hat, inner_out, leaf_out = sizes(log_q, log_b, k)
    max_digit = 2**log_b - 1
    err = (n + 1) * m_hat * max_digit * bound  # Lemma 5 with digits in [0, b)
    coords_inner = sum(T**j for j in range(R - 1)) * inner_out  # root and inner levels
    coords_leaf = T ** (R - 1) * leaf_out
    rounding = (coords_inner + coords_leaf) * (err + 1) * 2.0 ** (log_p - log_q)
    lifting = coords_inner * (max_digit**2 + 1) * 2.0 ** (-log_p)  # Corollary 2
    return math.log2(min(1.0, rounding + lifting))


def best_log_p(log_q, log_b, k, bound):
    return min(
        (log2_fail(log_q, log_p, log_b, k, bound), log_p)
        for log_p in range(max(2 * log_b + 1, log_q // 4), log_q)
    )


def min_log_q(log_b, k, bound):
    lo, hi = 16, 4000
    while lo < hi:
        mid = (lo + hi) // 2
        if best_log_p(mid, log_b, k, bound)[0] <= FAIL:
            hi = mid
        else:
            lo = mid + 1
    fail, log_p = best_log_p(lo, log_b, k, bound)
    return lo, log_p, fail


def main():
    for e in [int(a) for a in sys.argv[1].split(",")]:
        bound = 2**e
        for log_b in [int(a) for a in sys.argv[2].split(",")]:
            # fixed point: q depends on k (correctness) and k on q (security)
            k = 256
            while True:
                log_q, _, _ = min_log_q(log_b, k, bound)
                k2 = min_k(log_q, e)
                if k2 <= k:
                    break
                k = k2
            log_q, log_p, fail = min_log_q(log_b, k, bound)
            n, m_hat, inner_out, leaf_out = sizes(log_q, log_b, k)
            row = dict(
                e=e,
                log_b=log_b,
                k=k,
                log_q=log_q,
                log_p=log_p,
                log2_fail=round(fail, 1),
                security=round(security(k, log_q, e), 1),
                n=n,
                encoding_bytes=log_q * ((R - 1) * m_hat * inner_out + m_hat * leaf_out) / 8,
                b_matrix_bytes=log_q * n * k * (inner_out + leaf_out) / 8,
                digest_bytes=n * log_q / 8,
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
