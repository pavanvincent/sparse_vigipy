import random
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.sparse import issparse, csr_matrix
from scipy.stats import norm
from scipy.special import digamma, polygamma
from ..sparse_utils.sparse_Container import Container


# The trigamma function is mathematically the first derivative of digamma (first-order polygamma).
def trigamma(x):
    return polygamma(1, x)


def bcpnn(
    container: Container,
    min_events: int = 4,
    MC: bool = False,
    num_MC: int = 10000,
) -> Container:
    """
    Bayesian Confidence Propagation Neural Network (BCPNN) using sparse contingency structures:
    - Marginal totals computed at the report level (matching signal_methods.py & fast_profiles.py)
    - Analytical mode (MC=False) strictly identical to bcpnn_stats in signal_methods.py
    - Optional Monte Carlo mode (MC=True) via parallel Numba kernel

    Arguments:
        container: A Container object produced by Container.from_raw_tables() or convert()
        min_events (int): Minimum number of AE reports to be considered a candidate signal
        MC (bool): Use Monte Carlo simulations via Numba parallel kernel
        num_MC (int): Number of MC simulations to run per cell

    Returns:
        Container: Object containing all_signals, signals, num_signals, and sparse IC matrices
    """
    input_params = {
        "min_events": min_events,
        "MC": MC,
        "num_MC": num_MC,
    }

    has_sparse_cont = hasattr(container, "contingency") and issparse(container.contingency)

    if has_sparse_cont:
        cont = container.contingency.tocoo()
        N = float(container.N) if hasattr(container, "N") else float(cont.sum())

        # Filter non-zero entries directly in sparse format
        mask = cont.data >= min_events if min_events > 1 else cont.data > 0
        valid_rows = cont.row[mask]
        valid_cols = cont.col[mask]
        n11 = np.asarray(cont.data[mask], dtype=np.float64)

        # Compute marginal sums at the REPORT level (matching signal_methods.py & fast_profiles.py)
        if hasattr(container, "product_features") and hasattr(container, "event_outcomes"):
            row_sums = np.asarray(container.get_sparse_features().sum(axis=0), dtype=np.float64).ravel()
            col_sums = np.asarray(container.get_sparse_outcomes().sum(axis=0), dtype=np.float64).ravel()
        else:
            row_sums = np.asarray(container.contingency.sum(axis=1), dtype=np.float64).ravel()
            col_sums = np.asarray(container.contingency.sum(axis=0), dtype=np.float64).ravel()

        n1j = row_sums[valid_rows]
        ni1 = col_sums[valid_cols]

        # Retrieve product and adverse event names from index or fallback to DataFrame
        if hasattr(container, "product_index") and hasattr(container, "ae_index"):
            products = np.asarray(container.product_index)[valid_rows]
            aes = np.asarray(container.ae_index)[valid_cols]
        else:
            DATA = container.data
            if min_events > 1:
                DATA = DATA.loc[DATA["events"] >= min_events]
            products = DATA["product_name"].to_numpy()
            aes = DATA["ae_name"].to_numpy()
    else:
        # Fallback if only container.data is available
        DATA = container.data
        N = float(container.N)
        if min_events > 1:
            DATA = DATA.loc[DATA["events"] >= min_events]

        n11 = DATA["events"].to_numpy(dtype=np.float64)
        n1j = DATA["product_aes"].to_numpy(dtype=np.float64)
        ni1 = DATA["count_across_brands"].to_numpy(dtype=np.float64)
        products = DATA["product_name"].to_numpy()
        aes = DATA["ae_name"].to_numpy()

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    if not MC:
        # Analytical BCPNN (strictly identical to bcpnn_stats in signal_methods.py)
        p1 = 1.0 + n1j
        p2 = 1.0 + N - n1j
        q1 = 1.0 + ni1
        q2 = 1.0 + N - ni1
        r1 = 1.0 + n11
        r2b = N - n11 - 1.0 + (2.0 + N) ** 2 / (q1 * p1)

        IC = np.asarray(
            (
                digamma(r1)
                - digamma(r1 + r2b)
                - (digamma(p1) - digamma(p1 + p2) + digamma(q1) - digamma(q1 + q2))
            )
            / np.log(2.0),
            dtype=np.float64,
        )

        IC_variance = np.asarray(
            (
                (trigamma(r1) - trigamma(r1 + r2b))
                + (trigamma(p1) - trigamma(p1 + p2))
                + (trigamma(q1) - trigamma(q1 + q2))
            )
            / (np.log(2.0) ** 2),
            dtype=np.float64,
        )

        IC_std = np.sqrt(IC_variance)
        lower_bound = norm.ppf(0.025, IC, IC_std)
        upper_bound = norm.ppf(0.975, IC, IC_std)
        posterior_prob = norm.cdf(0.0, IC, IC_std)
    else:
        # Vectorized Dirichlet priors for Monte Carlo mode
        q1j = (n1j + 0.5) / (N + 1.0)
        qi1 = (ni1 + 0.5) / (N + 1.0)
        qi0 = (N - ni1 + 0.5) / (N + 1.0)
        q0j = (N - n1j + 0.5) / (N + 1.0)

        a_ = 0.5 / (q1j * qi1)

        g11 = (q1j * qi1 * a_) + n11
        g10 = (q1j * qi0 * a_) + n10
        g01 = (q0j * qi1 * a_) + n01
        g00 = (q0j * qi0 * a_) + n00

        posterior_prob, IC, lower_bound, upper_bound = _bcpnn_numba_kernel(
            g11, g10, g01, g00, num_MC
        )

    # Sort all arrays by descending lower_bound BEFORE cumulative FDR/Se/Sp calculation
    sort_idx = np.argsort(-lower_bound)

    products = products[sort_idx]
    aes = aes[sort_idx]
    n11 = n11[sort_idx]
    n10 = n10[sort_idx]
    n01 = n01[sort_idx]
    n00 = n00[sort_idx]
    IC = IC[sort_idx]
    lower_bound = lower_bound[sort_idx]
    upper_bound = upper_bound[sort_idx]
    p_h0 = np.asarray(posterior_prob[sort_idx], dtype=np.float64)

    # Compute FDR, FNR, Se and Sp
    p_h1 = 1.0 - p_h0
    total_true_signals = np.sum(p_h1)
    total_true_negatives = np.sum(p_h0)
    num_cells = len(p_h0)

    true_positives_cum = np.cumsum(p_h1)
    false_positives_cum = np.cumsum(p_h0)
    post_range = np.arange(1, num_cells + 1, dtype=np.float64)

    FDR = false_positives_cum / post_range
    Se = true_positives_cum / (total_true_signals + 1e-7)
    FNR = (total_true_signals - true_positives_cum) / (total_true_signals + 1e-7)
    Sp = (total_true_negatives - false_positives_cum) / (total_true_negatives + 1e-7)

    # Signal criterion: IC025 (lower_bound) > 0, equivalent to 2**lower_bound > 1
    signal_mask = lower_bound > 0.0
    num_signals = int(np.sum(signal_mask))

    # Build result Container
    RC = Container(params=True)
    RC.param["input_params"] = input_params

    if has_sparse_cont:
        sorted_rows = valid_rows[sort_idx]
        sorted_cols = valid_cols[sort_idx]
        RC.sparse_ic = csr_matrix((IC, (sorted_rows, sorted_cols)), shape=container.contingency.shape)
        RC.sparse_lb = csr_matrix((lower_bound, (sorted_rows, sorted_cols)), shape=container.contingency.shape)
        RC.sparse_signals = csr_matrix(
            (signal_mask, (sorted_rows, sorted_cols)), shape=container.contingency.shape, dtype=bool
        )

    RC.all_signals = pd.DataFrame(
        {
            "Product": products,
            "Adverse Event": aes,
            "2**IC": 2.0 ** IC,
            "lower_bound": 2.0 ** lower_bound,
            "upper_bound": 2.0 ** upper_bound,
            "p_{H0}": p_h0,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "FDR": FDR,
            "FNR": FNR,
            "Se": Se,
            "Sp": Sp,
        }
    )

    RC.signals = RC.all_signals.loc[RC.all_signals["lower_bound"] > 1.0].copy()
    RC.num_signals = num_signals
    return RC


# -----------------------------------
# Parallelized Numba kernel for MC loop
# -----------------------------------
@njit(parallel=True)
def _bcpnn_numba_kernel(g11, g10, g01, g00, num_MC):
    num_cell = len(g11)

    posterior_prob = np.empty(num_cell, dtype=np.float64)
    IC = np.empty(num_cell, dtype=np.float64)
    lower_bound = np.empty(num_cell, dtype=np.float64)
    upper_bound = np.empty(num_cell, dtype=np.float64)

    idx_25 = min(int(num_MC * 0.025), num_MC - 1)
    idx_50 = min(int(num_MC * 0.500), num_MC - 1)
    idx_975 = min(int(num_MC * 0.975), num_MC - 1)

    for m in prange(num_cell):
        ic_monte = np.empty(num_MC, dtype=np.float64)

        alpha11 = g11[m]
        alpha10 = g10[m]
        alpha01 = g01[m]
        alpha00 = g00[m]

        for i in range(num_MC):
            gamma11 = random.gammavariate(alpha11, 1.0)
            gamma10 = random.gammavariate(alpha10, 1.0)
            gamma01 = random.gammavariate(alpha01, 1.0)
            gamma00 = random.gammavariate(alpha00, 1.0)

            total = gamma11 + gamma10 + gamma01 + gamma00

            if total > 0.0:
                p11 = gamma11 / total
                p1_ = (gamma11 + gamma10) / total
                p_1 = (gamma11 + gamma01) / total

                if p1_ > 0.0 and p_1 > 0.0 and p11 > 0.0:
                    ic_monte[i] = np.log2(p11 / (p1_ * p_1))
                else:
                    ic_monte[i] = -np.inf
            else:
                ic_monte[i] = -np.inf

        ic_monte.sort()

        under_zero_count = np.searchsorted(ic_monte, 0.0, side="left")
        posterior_prob[m] = under_zero_count / num_MC

        lower_bound[m] = ic_monte[idx_25]
        IC[m] = ic_monte[idx_50]
        upper_bound[m] = ic_monte[idx_975]

    return posterior_prob, IC, lower_bound, upper_bound
