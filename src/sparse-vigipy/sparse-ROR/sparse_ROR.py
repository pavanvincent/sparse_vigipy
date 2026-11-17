import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from scipy.stats import norm
from ..sparse-utils.sparse_Container import Container


def ror(
    container,
    min_events=4,
):
    """
    Calculate the Reporting Odds Ratio (ROR) using sparse contingency structures:
    - Confidence Interval [LB, UB] at 95% estimated using Woolf's method
    - Signal Detection Rate (SDR) if log(LB) > 0
    - p-value computed using a one-sided Wald test

    Arguments:
        container: A Container object produced by convert() from sparse_data_prep.py
        min_events: Minimum number of AE reports to be considered a candidate signal

    Returns:
        Container: Object containing all_signals, signals, num_signals, and sparse ROR matrices
    """
    has_sparse_cont = hasattr(container, "contingency") and issparse(container.contingency)

    if has_sparse_cont:
        cont = container.contingency.tocoo()
        N = float(container.N) if hasattr(container, "N") else float(cont.sum())

        # Filter non-zero entries directly in sparse format
        mask = cont.data >= min_events if min_events > 1 else cont.data > 0
        valid_rows = cont.row[mask]
        valid_cols = cont.col[mask]
        n11 = np.asarray(cont.data[mask], dtype=np.float64)

        # Compute marginal sums directly on the sparse matrix
        row_sums = np.asarray(container.contingency.sum(axis=1)).ravel()
        col_sums = np.asarray(container.contingency.sum(axis=0)).ravel()

        n1j = row_sums[valid_rows]
        ni1 = col_sums[valid_cols]

        # Retrieve product and adverse event names from index or fallback to DataFrame
        if hasattr(container, "product_index") and hasattr(container, "ae_index"):
            products = np.asarray(container.product_index)[valid_rows]
            aes = np.asarray(container.ae_index)[valid_cols]
        else:
            DATA = container.data
            if min_events > 1:
                DATA = DATA[DATA["events"] >= min_events]
            products = DATA["product_name"].values
            aes = DATA["ae_name"].values
    else:
        # Fallback if only container.data is available
        DATA = container.data
        N = float(container.N)
        if min_events > 1:
            DATA = DATA[DATA["events"] >= min_events]

        n11 = np.asarray(DATA["events"], dtype=np.float64)
        n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
        ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
        products = DATA["product_name"].values
        aes = DATA["ae_name"].values

    # Compute 2x2 contingency cells
    n10 = n1j - n11 + 1e-7
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)

    # Compute log(ROR) and VAR(log(ROR)) using Woolf's method
    log_ror = np.log(n11 * n00 / (n10 * n01))
    var_log_ror = 1.0 / n11 + 1.0 / n10 + 1.0 / n01 + 1.0 / n00
    std_err = np.sqrt(var_log_ror)

    # Compute 95% confidence interval
    log_LB = norm.ppf(0.025, log_ror, std_err)
    log_UB = norm.ppf(0.975, log_ror, std_err)

    # Guard against overflow when log_UB > max_log_value
    max_log_value = np.log(np.finfo(np.float64).max)
    ub_exp = np.full_like(log_UB, np.inf)
    safe_mask = log_UB <= max_log_value
    ub_exp[safe_mask] = np.exp(log_UB[safe_mask])

    # Compute one-sided p-value under H0: ROR = 1
    ror_H0 = 1.0
    pval_uni = 1.0 - norm.cdf(log_ror, np.log(ror_H0), std_err)
    pval_uni = np.clip(pval_uni, 0.0, 1.0)

    # Count signals using decision criterion log_LB > 0
    num_signals = int((log_LB > 0).sum())
    ror_exp = np.exp(log_ror)
    lb_exp = np.exp(log_LB)

    # Build result Container
    RC = Container()
    if has_sparse_cont:
        RC.sparse_ror = csr_matrix((ror_exp, (valid_rows, valid_cols)), shape=container.contingency.shape)
        RC.sparse_lb = csr_matrix((lb_exp, (valid_rows, valid_cols)), shape=container.contingency.shape)

    RC.all_signals = pd.DataFrame(
        {
            "Product": products,
            "Adverse Event": aes,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "ROR": ror_exp,
            "LB(CI 95%)": lb_exp,
            "UB(CI 95%)": ub_exp,
            "p-value": pval_uni,
        },
        index=np.arange(len(n11)),
    ).sort_values(by=["LB(CI 95%)"], ascending=False)

    RC.signals = RC.all_signals.iloc[0:num_signals]
    RC.num_signals = num_signals
    return RC
