import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.special import gamma
from scipy.stats import norm, rankdata
from scipy.optimize import minimize


def lbe(
    pvals,
    a=None,
    lb=0.05,
    ci_level=0.95,
    qvalues=True,
    fdr_level=0.05,
    n_significant=None,
):
    """
    Location-Based Estimator (LBE) for False Discovery Rate (FDR) estimation.
    Supports both 1D dense arrays/lists and SciPy sparse matrices (where non-zero
    stored entries represent active tested hypotheses).
    """
    sparse_input = issparse(pvals)
    if sparse_input:
        coo = pvals.tocoo()
        pvals_arr = np.asarray(coo.data, dtype=np.float64)
    else:
        pvals_arr = np.asarray(pvals, dtype=np.float64).ravel()

    if len(pvals_arr) == 0:
        raise ValueError("ERROR: pvals input is empty.")

    if np.min(pvals_arr) < 0 or np.max(pvals_arr) > 1:
        raise ValueError("ERROR: p-values not in valid range.")

    m = len(pvals_arr)
    fdr = None

    if a is not None and a < 1:
        a = None
        sdbound = np.sqrt(1 / (3 * m))
        pi0 = min(1.0, float(np.mean(pvals_arr) * 2))
        icpi0 = [0, pi0 - norm.ppf((1 - ci_level), 0, sdbound)]
    else:
        if a is None:
            a = lbe_a(m, lb)
        sdbound = np.sqrt((1 / (gamma(a + 1)) ** 2) * ((gamma(2 * a + 1) - (gamma(a + 1)) ** 2) / m))
        # Clip p-values slightly below 1.0 to avoid log(0) -> inf
        safe_pvals = np.clip(pvals_arr, 0.0, 1.0 - np.finfo(np.float64).eps)
        pi0 = min(1.0, float(np.mean((-np.log(1.0 - safe_pvals)) ** a) / gamma(a + 1)))
        icpi0 = [0, min(1.0, pi0 - norm.ppf((1 - ci_level), 0, sdbound))]

    if sdbound > 0.5:
        print(
            """WARNING: l = {0}. A smaller value is
                recommended for a (or l).""".format(
                sdbound
            )
        )

    if qvalues:
        sort_pval = np.sort(pvals_arr)
        # Use int32 instead of int16 to prevent overflow on large sparse matrices (>32,767 elements)
        rank_pval = (rankdata(pvals_arr, method="ordinal") - 1).astype(np.int32)

        # Vectorized cumulative minimum from right to left (replaces Python for-loop)
        denominators = np.arange(1, m + 1, dtype=np.float64)
        raw_q = (pi0 * m * sort_pval) / denominators
        qval = np.minimum.accumulate(raw_q[::-1])[::-1]

        if n_significant is not None:
            fdr_level = float(qval[n_significant])
            fdr = fdr_level
        else:
            sig_mask_sorted = qval <= fdr_level
            n_significant = int(np.sum(sig_mask_sorted))
            if n_significant > 0:
                fdr = float(max(np.max(qval[sig_mask_sorted]), 0.0))
            else:
                print("No data matches the specified FDR threshold. Setting FDR to 0.")
                fdr = 0.0

        mapped_qvals = qval[rank_pval]
        significant = mapped_qvals <= fdr_level
        sig_count = int(np.sum(significant))

        # Reconstruct sparse matrices if the input was sparse
        if sparse_input:
            out_qvals = csr_matrix((mapped_qvals, (coo.row, coo.col)), shape=pvals.shape)
            out_pvals = pvals.tocsr()
            out_sig = csr_matrix((significant, (coo.row, coo.col)), shape=pvals.shape, dtype=bool)
        else:
            out_qvals = mapped_qvals
            out_pvals = pvals_arr
            out_sig = significant

        r = [
            fdr,
            pi0,
            icpi0,
            ci_level,
            a,
            sdbound,
            out_qvals,
            out_pvals,
            out_sig,
            sig_count,
        ]
    else:
        out_pvals = pvals.tocsr() if sparse_input else pvals_arr
        r = [None, pi0, icpi0, ci_level, a, sdbound, None, out_pvals, None, None]

    return r


def lbe_a(m, l):
    res = minimize(asearch, [1.0], bounds=[(0.3, 25.0)], args=(m, l), method="L-BFGS-B")
    val = float(np.asarray(res.x).ravel()[0])
    return max(1.0, val)


def asearch(a, m, l):
    a_val = float(np.asarray(a).ravel()[0])
    return float(
        np.abs(np.sqrt(1 / (gamma(a_val + 1)) ** 2 * ((gamma(2 * a_val + 1) - (gamma(a_val + 1)) ** 2) / m)) - l)
    )
