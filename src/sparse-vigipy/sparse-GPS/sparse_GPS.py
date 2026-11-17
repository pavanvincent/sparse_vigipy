from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from scipy.special import digamma, gdtr, gammaln, betainc
from scipy.optimize import minimize

from ..sparse-utils.sparse_Container import Container
from ..sparse-utils.sparse_expectations import calculate_expected
from ..sparse-utils.sparse-distribution_funcs.sparse_quantile_funcs import quantiles

EPS = np.finfo(np.float32).eps
BOUNDED_METHODS = {
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
    "COBYQA",
}


def _extract_sparse_or_dense_counts(container):
    """Extract active (n11, n1j, ni1, N, products, aes, rows, cols, shape)
    from a sparse contingency matrix if present, otherwise fallback to container.data."""
    has_sparse_cont = hasattr(container, "contingency") and issparse(container.contingency)

    if has_sparse_cont:
        cont = container.contingency.tocoo()
        N = float(container.N) if hasattr(container, "N") else float(cont.sum())

        mask = cont.data > 0
        valid_rows = cont.row[mask]
        valid_cols = cont.col[mask]
        n11 = np.asarray(cont.data[mask], dtype=np.float64)

        row_sums = np.asarray(container.contingency.sum(axis=1)).ravel()
        col_sums = np.asarray(container.contingency.sum(axis=0)).ravel()

        n1j = row_sums[valid_rows]
        ni1 = col_sums[valid_cols]

        if hasattr(container, "product_index") and hasattr(container, "ae_index"):
            products = np.asarray(container.product_index)[valid_rows]
            aes = np.asarray(container.ae_index)[valid_cols]
        else:
            products = container.data["product_name"].to_numpy()
            aes = container.data["ae_name"].to_numpy()

        return has_sparse_cont, n11, n1j, ni1, N, products, aes, valid_rows, valid_cols, container.contingency.shape

    DATA = container.data
    N = float(container.N)
    n11 = np.asarray(DATA["events"], dtype=np.float64)
    n1j = np.asarray(DATA["product_aes"], dtype=np.float64)
    ni1 = np.asarray(DATA["count_across_brands"], dtype=np.float64)
    products = DATA["product_name"].to_numpy()
    aes = DATA["ae_name"].to_numpy()

    return False, n11, n1j, ni1, N, products, aes, None, None, None


def _optimize_gps_priors(
    container: Container,
    priors: np.ndarray,
    truncate: bool,
    truncate_thres: float,
    n11: np.ndarray,
    expected: np.ndarray,
    N: float,
    minimization_method: str,
    minimization_bounds: tuple[tuple[float, float], ...] | None,
    minimization_options: dict | None,
) -> tuple[np.ndarray, str]:
    """Estimate empirical Bayes hyperprior parameters via maximum likelihood."""
    if minimization_method not in BOUNDED_METHODS:
        minimization_bounds = None
    elif minimization_bounds is None:
        minimization_bounds = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))

    opts = {"maxiter": 500}
    if minimization_options:
        opts.update(minimization_options)

    if not truncate:
        gammaln_n11_1 = gammaln(n11 + 1.0)
        p_out = minimize(
            non_truncated_likelihood,
            x0=priors,
            args=(n11, expected, gammaln_n11_1),
            options=opts,
            method=minimization_method,
            bounds=minimization_bounds,
        )
    else:
        trunc = truncate_thres - 1
        n11_trunc = n11[n11 >= truncate_thres]
        E_trunc = expected[n11 >= truncate_thres]
        gammaln_n11_1 = gammaln(n11_trunc + 1.0)
        p_out = minimize(
            truncated_likelihood,
            x0=priors,
            args=(
                n11_trunc,
                E_trunc,
                trunc,
                gammaln_n11_1,
            ),
            options=opts,
            method=minimization_method,
            bounds=minimization_bounds,
        )

    priors_opt = p_out.x
    if np.any(priors_opt < 0) or priors_opt[4] > 1:
        warnings.warn(
            f"Calculated priors violate distribution constraints. Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. Current priors: {priors_opt}. Numerical instability likely during processing. Considering using a minimization method that supports bounds."
        )
    return priors_opt, p_out.message


def gps(
    container,
    min_events=4,
    truncate=False,
    maxiter=500,
    criterion="EB05>=2",
):
    """Computes signal detection based on Multi-item enabled Gamma Poisson Shrinkage (GPS)
    using sparse contingency structures."""
    relative_risk = 1.0
    decision_metric = "rank"
    decision_thres = 0.05
    ranking_statistic = "log2"
    truncate_thres = 1.0
    prior_init = None
    prior_param = None
    expected_method = "mantel-haentzel"
    method_alpha = 1.0
    minimization_method = "SLSQP"
    minimization_bounds = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))
    minimization_options = {"maxiter": maxiter}

    if prior_init is None:
        prior_init = {
            "alpha1": 0.2041,
            "beta1": 0.05816,
            "alpha2": 1.415,
            "beta2": 1.838,
            "w": 0.0969,
        }
    elif isinstance(prior_init, (list, tuple, np.ndarray)):
        prior_init = {
            "alpha1": float(prior_init[0]),
            "beta1": float(prior_init[1]),
            "alpha2": float(prior_init[2]),
            "beta2": float(prior_init[3]),
            "w": float(prior_init[4]),
        }

    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "decision_metric": decision_metric,
        "decision_thres": decision_thres,
        "ranking_statistic": ranking_statistic,
        "truncate": truncate,
        "truncate_thres": truncate_thres,
        "expected_method": expected_method,
        "method_alpha": method_alpha,
        "minimization_method": minimization_method,
    }

    priors = np.asarray(
        [
            prior_init["alpha1"],
            prior_init["beta1"],
            prior_init["alpha2"],
            prior_init["beta2"],
            prior_init["w"],
        ],
        dtype=np.float64,
    )

    (
        has_sparse_cont,
        n11,
        n1j,
        ni1,
        N,
        products,
        aes,
        valid_rows,
        valid_cols,
        cont_shape,
    ) = _extract_sparse_or_dense_counts(container)

    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    code_convergence = "User-provided priors"

    if prior_param is None:
        priors, code_convergence = _optimize_gps_priors(
            container=container,
            priors=priors,
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11,
            expected=expected,
            N=N,
            minimization_method=minimization_method,
            minimization_bounds=minimization_bounds,
            minimization_options=minimization_options,
        )
    else:
        priors = np.asarray(prior_param, dtype=np.float64)

    computation_params = pd.DataFrame.from_dict(
        {
            "minimization_method": minimization_method,
            "minimization_bounds": minimization_bounds,
            "priori_init": prior_init,
            "method_alpha": method_alpha,
            "expected_method": expected_method,
            "minimization_options": minimization_options,
        },
        orient="index",
        columns=["Value"],
    )

    if min_events > 1:
        mask = n11 >= min_events
        products = products[mask]
        aes = aes[mask]
        expected = expected[mask]
        n1j = n1j[mask]
        ni1 = ni1[mask]
        n11 = n11[mask]
        if has_sparse_cont:
            valid_rows = valid_rows[mask]
            valid_cols = valid_cols[mask]

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    # Posterior probability of the null hypothesis
    _p_post1 = np.clip(priors[1] / (priors[1] + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    _p_post2 = np.clip(priors[3] / (priors[3] + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    gammaln_n11_1_post = gammaln(n11 + 1.0)
    r1, r2 = priors[0], priors[2]
    log_qdb1 = gammaln(n11 + r1) - gammaln_n11_1_post - gammaln(r1) + r1 * np.log(_p_post1) + n11 * np.log(1.0 - _p_post1)
    log_qdb2 = gammaln(n11 + r2) - gammaln_n11_1_post - gammaln(r2) + r2 * np.log(_p_post2) + n11 * np.log(1.0 - _p_post2)
    qdb1 = np.exp(log_qdb1)
    qdb2 = np.exp(log_qdb2)

    _qn_denom = priors[4] * qdb1 + (1.0 - priors[4]) * qdb2
    Qn = np.where(_qn_denom > 0, priors[4] * qdb1 / _qn_denom, priors[4])

    gd1 = gdtr(relative_risk, priors[0] + n11, np.maximum(priors[1] + expected, 1e-10))
    gd2 = gdtr(relative_risk, priors[2] + n11, np.maximum(priors[3] + expected, 1e-10))
    posterior_probability = Qn * gd1 + (1.0 - Qn) * gd2

    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(np.maximum(priors[1] + expected, 1e-10))
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(np.maximum(priors[3] + expected, 1e-10))
    EBlog2 = (np.log(2.0) ** -1) * (Qn * dgterm1 + (1.0 - Qn) * dgterm2)
    ebgm = np.power(2.0, np.asarray(EBlog2, dtype=np.float64))

    # Calculation of the Lower Bound (EB05) and Upper Bound (EB95)
    EB05 = quantiles(
        0.05,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )
    EB95 = quantiles(
        0.95,
        Qn,
        priors[0] + n11,
        priors[1] + expected,
        priors[2] + n11,
        priors[3] + expected,
    )

    # Apply signal decision criterion
    if criterion == "EB05>=2":
        raw_sig_mask = EB05 >= 2.0
    elif criterion in ("EB05 > 1 & EBGM >=2", "EBGM >= 2 and EB05 > 1"):
        raw_sig_mask = (ebgm >= 2.0) & (EB05 > 1.0)
    else:
        raw_sig_mask = np.ones(len(n11), dtype=bool)

    RES = Container(params=True)
    RES.param["input_params"] = input_params
