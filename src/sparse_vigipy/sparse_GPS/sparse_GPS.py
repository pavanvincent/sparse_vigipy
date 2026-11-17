from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse, csr_matrix
from scipy.special import digamma, gdtr, gammaln, betainc
from scipy.optimize import minimize

from ..sparse_utils.sparse_Container import Container
from ..sparse_utils.sparse_expectations import calculate_expected
from ..sparse_utils.sparse_distribution_funcs.sparse_quantile_funcs import quantiles

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


def _get_labels(
    container: Container,
    attr_candidates: tuple[str, ...],
    raw_obj: object,
    size: int,
    prefix: str,
) -> np.ndarray:
    """Retrieve feature/outcome labels from Container attributes, DataFrame columns, or fallback indices."""
    for attr in attr_candidates:
        if hasattr(container, attr) and getattr(container, attr) is not None:
            return np.asarray(getattr(container, attr))
    if isinstance(raw_obj, pd.DataFrame):
        return np.asarray(raw_obj.columns)
    return np.asarray([f"{prefix}_{i}" for i in range(size)])


def _build_contingency_from_container(
    container: Container,
) -> tuple[sparse.coo_matrix, np.ndarray, np.ndarray, float]:
    """Compute the sparse product-event contingency matrix using Container sparse getters."""
    X = container.get_sparse_features()  # shape: (n_reports, n_products)
    Y = container.get_sparse_outcomes()  # shape: (n_reports, n_aes)

    # Sparse matrix multiplication yields the (n_products, n_aes) N_11 count matrix
    mat = (X.T @ Y).astype(np.float64).tocoo()
    mat.eliminate_zeros()

    N = float(container.N) if hasattr(container, "N") and container.N is not None else float(mat.sum())

    products = _get_labels(
        container,
        ("product_names", "products", "feature_names"),
        getattr(container, "product_features", None),
        mat.shape[0],
        "Product",
    )
    aes = _get_labels(
        container,
        ("ae_names", "aes", "outcome_names"),
        getattr(container, "event_outcomes", None),
        mat.shape[1],
        "AE",
    )

    return mat, products, aes, N


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
    """Estimate empirical Bayes hyperprior parameters via maximum likelihood on sparse vectors."""
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
        trunc = truncate_thres - 1.0
        mask_trunc = n11 >= truncate_thres
        n11_trunc = n11[mask_trunc]
        E_trunc = expected[mask_trunc]
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
    return priors_opt, str(p_out.message)


def non_truncated_likelihood(
    p: np.ndarray,
    n11: np.ndarray,
    E: np.ndarray,
    gammaln_n11_1: np.ndarray | None = None,
) -> float:
    """Compute negative log-likelihood for the non-truncated negative binomial mixture."""
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    dnb1 = np.exp(log_dnb1)
    dnb2 = np.exp(log_dnb2)
    term = (w * dnb1 + (1.0 - w) * dnb2) + 1e-7
    return float(np.sum(-np.log(term)))


def truncated_likelihood(
    p: np.ndarray,
    n11: np.ndarray,
    E: np.ndarray,
    truncate: float,
    gammaln_n11_1: np.ndarray | None = None,
) -> float:
    """Compute negative log-likelihood for the truncated negative binomial mixture."""
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    dnb1 = np.exp(log_dnb1)
    dnb2 = np.exp(log_dnb2)
    term1 = w * dnb1 + (1.0 - w) * dnb2

    if truncate == 0:
        pnb1 = np.exp(r1 * np.log(p_nb1))
        pnb2 = np.exp(r2 * np.log(p_nb2))
    else:
        pnb1 = betainc(r1, truncate + 1, p_nb1)
        pnb2 = betainc(r2, truncate + 1, p_nb2)
    term2 = 1.0 - (w * pnb1 + (1.0 - w) * pnb2)

    return float(np.sum(-np.log(np.maximum(term1, 1e-300)) + np.log(np.maximum(term2, 1e-7))))


def gps(
    container: Container,
    min_events: int = 4,
    truncate: bool = False,
    maxiter: int = 500,
    criterion: str = "EB05>=2",
) -> Container:
    """Compute signal detection based on Multi-item Gamma Poisson Shrinkage (GPS) using sparse matrices.

    Clinical Intuition:
        GPS models the true relative risk distribution across all drug-event pairs as a mixture
        of two Gamma distributions: a dominant background null component (capturing non-signals
        centered near RR=1.0) and an elevated risk component. By shrinking observed counts toward
        this empirical Bayesian mixture, GPS stabilizes high-variance small counts while producing
        robust empirical Bayes geometric mean (EBGM) estimates and conservative 5th percentile
        lower bounds (EB05).

    Parameters:
        container: A Container object implementing `get_sparse_features()` and `get_sparse_outcomes()`.
        min_events: Minimum number of events required for a pair to be retained.
        truncate: Whether to truncate likelihoods below a threshold for numerical stability.
        maxiter: Maximum number of iterations in the optimization algorithm.
        criterion: Criterion retained for signal detection ('EB05>=2' or 'EB05 > 1 & EBGM >=2').

    Returns:
        Container containing detected signals, all evaluated pairs, sparse result matrices,
        signal count, and model parameters.
    """
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

    # 1. Build sparse contingency matrix via Container.get_sparse_features() & get_sparse_outcomes()
    mat, products, aes, N = _build_contingency_from_container(container)
    row_totals = np.asarray(mat.sum(axis=1)).ravel()
    col_totals = np.asarray(mat.sum(axis=0)).ravel()

    n11 = mat.data
    rows = mat.row
    cols = mat.col
    n1j = row_totals[rows]
    ni1 = col_totals[cols]

    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)
    code_convergence = "User-provided priors"

    # 2. Prior optimization on non-zero entries
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

    # 3. Sparse filtering (remove cells below min_events)
    if min_events > 1:
        mask = n11 >= min_events
        n11 = n11[mask]
        rows = rows[mask]
        cols = cols[mask]
        expected = expected[mask]
        n1j = n1j[mask]
        ni1 = ni1[mask]

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    # 4. Posterior Bayesian calculations restricted to active coordinates
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
    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1.0 - Qn) * dgterm2)
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

    # 5. Build result container and attach sparse output matrices
    RES = Container(params=True)
    RES.param["input_params"] = input_params
    RES.param["computation_params"] = computation_params
    RES.param["convergence"] = code_convergence
    RES.param["priors"] = priors

    # Store results as sparse matrices (shape = n_products x n_aes)
    RES.sparse_ebgm = sparse.coo_matrix((ebgm, (rows, cols)), shape=mat.shape)
    RES.sparse_eb05 = sparse.coo_matrix((EB05, (rows, cols)), shape=mat.shape)

    RES.all_signals = pd.DataFrame(
        {
            "Product": products[rows],
            "Adverse Event": aes[cols],
            "EBGM": ebgm,
            "EB05": EB05,
            "EB95": EB95,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "p_{H0}": posterior_probability,
        }
    )

    # Sort descending by EB05 before cumulative metric calculation
    RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False).reset_index(drop=True)

    # Compute FDR, Sensitivity (Se), FNR, and Specificity (Sp) on sorted results
    p_h0 = RES.all_signals["p_{H0}"].values
    p_h1 = 1.0 - p_h0

    total_true_signals = np.sum(p_h1)
    total_true_negatives = np.sum(p_h0)

    true_positives_cum = np.cumsum(p_h1)
    false_positives_cum = np.cumsum(p_h0)
    post_range = np.arange(1, len(p_h0) + 1)

    RES.all_signals["FDR"] = false_positives_cum / post_range
    RES.all_signals["Se"] = true_positives_cum / (total_true_signals + 1e-7)
    RES.all_signals["FNR"] = (total_true_signals - true_positives_cum) / (total_true_signals + 1e-7)
    RES.all_signals["Sp"] = (total_true_negatives - false_positives_cum) / (total_true_negatives + 1e-7)

    # Apply signal detection criterion
    if criterion == "EB05>=2":
        signal_mask = RES.all_signals["EB05"] >= 2.0
    elif criterion == "EB05 > 1 & EBGM >=2":
        signal_mask = (RES.all_signals["EBGM"] >= 2.0) & (RES.all_signals["EB05"] > 1.0)
    else:
        signal_mask = pd.Series(True, index=RES.all_signals.index)

    RES.signals = RES.all_signals[signal_mask].copy()
    RES.num_signals = len(RES.signals)

    return RES


def gps_sampled(
    container: Container,
    max_points: int = 150_000,
    seed: int = 0,
    min_events: int = 4,
    truncate: bool = False,
    truncate_thres: float = 1.0,
    maxiter: int = 500,
    criterion: str = "EB05>=2",
) -> Container:
    """Subsampled version of GPS using Container sparse getters for prior estimation."""
    priors = np.asarray([0.2041, 0.05816, 1.415, 1.838, 0.0969], dtype=np.float64)
    bounds = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))

    # 1. Build sparse contingency matrix and compute expected counts
    mat, products, aes, N = _build_contingency_from_container(container)
    row_totals = np.asarray(mat.sum(axis=1)).ravel()
    col_totals = np.asarray(mat.sum(axis=0)).ravel()

    n11 = mat.data
    rows = mat.row
    cols = mat.col
    n1j = row_totals[rows]
    ni1 = col_totals[cols]
    expected = (n1j * ni1) / N

    if truncate:
        mask_fit = n11 >= truncate_thres
        n11_fit, E_fit = n11[mask_fit], expected[mask_fit]
    else:
        n11_fit, E_fit = n11, expected

    total_pairs = len(n11_fit)
    if max_points is not None and total_pairs > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total_pairs, max_points, replace=False)
        n11_fit, E_fit = n11_fit[idx], E_fit[idx]

    gl_fit = gammaln(n11_fit + 1.0)

    if not truncate:
        res = minimize(
            non_truncated_likelihood,
            x0=priors,
            args=(n11_fit, E_fit, gl_fit),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
    else:
        res = minimize(
            truncated_likelihood,
            x0=priors,
            args=(n11_fit, E_fit, truncate_thres - 1.0, gl_fit),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
    priors = res.x
    code_convergence = str(res.message)

    # 2. Filter pairs below min_events
    if min_events > 1:
        mask = n11 >= min_events
        n11 = n11[mask]
        rows = rows[mask]
        cols = cols[mask]
        expected = expected[mask]
        n1j = n1j[mask]
        ni1 = ni1[mask]

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    # 3. Posterior calculations
    a1, b1, a2, b2, w = priors
    p1 = np.clip(b1 / (b1 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    p2 = np.clip(b2 / (b2 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    gl_post = gammaln(n11 + 1.0)

    q1 = np.exp(gammaln(n11 + a1) - gl_post - gammaln(a1) + a1 * np.log(p1) + n11 * np.log(1.0 - p1))
    q2 = np.exp(gammaln(n11 + a2) - gl_post - gammaln(a2) + a2 * np.log(p2) + n11 * np.log(1.0 - p2))
    den = w * q1 + (1.0 - w) * q2
    Qn = np.where(den > 0, w * q1 / np.where(den > 0, den, 1.0), w)

    p_h0 = (
        Qn * gdtr(1.0, a1 + n11, np.maximum(b1 + expected, 1e-10))
        + (1.0 - Qn) * gdtr(1.0, a2 + n11, np.maximum(b2 + expected, 1e-10))
    )
    eblog2 = (
        Qn * (digamma(a1 + n11) - np.log(np.maximum(b1 + expected, 1e-10)))
        + (1.0 - Qn) * (digamma(a2 + n11) - np.log(np.maximum(b2 + expected, 1e-10)))
    ) / np.log(2)
    ebgm = np.power(2.0, np.asarray(eblog2, dtype=np.float64))

    EB05 = quantiles(0.05, Qn, a1 + n11, b1 + expected, a2 + n11, b2 + expected)
    EB95 = quantiles(0.95, Qn, a1 + n11, b1 + expected, a2 + n11, b2 + expected)

    # 4. Build result container
    RES = Container(params=True)
    RES.param["priors"] = priors
    RES.param["convergence"] = code_convergence
    RES.param["sampling_info"] = {
        "max_points": max_points,
        "seed": seed,
        "total_candidate_pairs": total_pairs,
        "fitted_on": len(n11_fit),
    }

    RES.sparse_ebgm = sparse.coo_matrix((ebgm, (rows, cols)), shape=mat.shape)
    RES.sparse_eb05 = sparse.coo_matrix((EB05, (rows, cols)), shape=mat.shape)

    RES.all_signals = pd.DataFrame(
        {
            "Product": products[rows],
            "Adverse Event": aes[cols],
            "EBGM": ebgm,
            "EB05": EB05,
            "EB95": EB95,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "p_{H0}": p_h0,
        }
    )

    RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False).reset_index(drop=True)

    p_null = RES.all_signals["p_{H0}"].values
    p_alt = 1.0 - p_null
    tp_cum = np.cumsum(p_alt)
    fp_cum = np.cumsum(p_null)
    tot_pos = np.sum(p_alt)
    tot_neg = np.sum(p_null)

    RES.all_signals["FDR"] = fp_cum / np.arange(1, len(p_null) + 1)
    RES.all_signals["Se"] = tp_cum / (tot_pos + 1e-7)
    RES.all_signals["FNR"] = (tot_pos - tp_cum) / (tot_pos + 1e-7)
    RES.all_signals["Sp"] = (tot_neg - fp_cum) / (tot_neg + 1e-7)

    if criterion == "EB05>=2":
        signal_mask = RES.all_signals["EB05"] >= 2.0
    elif criterion == "EB05 > 1 & EBGM >=2":
        signal_mask = (RES.all_signals["EBGM"] >= 2.0) & (RES.all_signals["EB05"] > 1.0)
    else:
        signal_mask = pd.Series(True, index=RES.all_signals.index)

    RES.signals = RES.all_signals[signal_mask].copy()
    RES.num_signals = len(RES.signals)
    return RES
