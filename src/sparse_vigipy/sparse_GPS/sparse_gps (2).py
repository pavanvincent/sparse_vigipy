from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import digamma, gdtr, gammaln, betainc
from scipy.optimize import minimize

from ..sparse_utils.sparse_Container import Container
from ..sparse_utils.sparse_expectations import calculate_expected
from ..sparse_utils.sparse_distribution_funcs.sparse_quantile_funcs import quantiles

EPS = float(np.finfo(np.float32).eps)
GPS_INIT = np.array([0.2041, 0.05816, 1.415, 1.838, 0.0969], dtype=np.float64)
GPS_BOUNDS = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))
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


def _extract_sparse_data(
    container: Container,
) -> tuple[sparse.coo_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract sparse contingency matrix (COO), report-level marginal totals, labels, and N."""
    has_individual = hasattr(container, "product_features") and hasattr(container, "event_outcomes")

    if has_individual:
        sf = container.get_sparse_features()
        so = container.get_sparse_outcomes()
        products = np.asarray(container.product_names)
        aes = np.asarray(container.ae_names)
        N = float(container.N) if hasattr(container, "N") and container.N is not None else float(sf.shape[0])

        # Exact report-level marginal counts (matching signal_methods.py & fast_profiles.py)
        row_totals = np.asarray(sf.sum(axis=0), dtype=np.float64).ravel()
        col_totals = np.asarray(so.sum(axis=0), dtype=np.float64).ravel()

        if hasattr(container, "contingency") and sparse.issparse(container.contingency):
            mat = container.contingency.tocoo()
        else:
            mat = sf.astype(np.float64).T.dot(so.astype(np.float64)).tocoo()
        mat.eliminate_zeros()

    elif getattr(container, "type", None) == "contingency" or (
        hasattr(container, "contingency") and sparse.issparse(container.contingency)
    ):
        mat = container.contingency
        if not isinstance(mat, sparse.coo_matrix):
            mat = mat.tocoo()
        mat.eliminate_zeros()

        if hasattr(container, "product_names") and container.product_names is not None:
            products = np.asarray(container.product_names)
        elif hasattr(container, "product_index") and container.product_index is not None:
            products = np.asarray(container.product_index)
        else:
            products = np.asarray(pd.Categorical(container.data["product_name"]).categories)

        if hasattr(container, "ae_names") and container.ae_names is not None:
            aes = np.asarray(container.ae_names)
        elif hasattr(container, "ae_index") and container.ae_index is not None:
            aes = np.asarray(container.ae_index)
        else:
            aes = np.asarray(pd.Categorical(container.data["ae_name"]).categories)

        N = float(container.N)
        row_totals = np.asarray(mat.sum(axis=1), dtype=np.float64).ravel()
        col_totals = np.asarray(mat.sum(axis=0), dtype=np.float64).ravel()
    else:
        raise ValueError("Container must provide 'product_features'/'event_outcomes' or 'contingency'.")

    return mat, row_totals, col_totals, products, aes, N


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


def _optimize_gps_priors(
    priors: np.ndarray,
    truncate: bool,
    truncate_thres: float,
    n11: np.ndarray,
    expected: np.ndarray,
    minimization_method: str,
    minimization_bounds: tuple[tuple[float, float], ...] | None,
    minimization_options: dict | None,
) -> tuple[np.ndarray, str]:
    """Estimate empirical Bayes hyperprior parameters via maximum likelihood on sparse vectors."""
    if minimization_method not in BOUNDED_METHODS:
        minimization_bounds = None
    elif minimization_bounds is None:
        minimization_bounds = GPS_BOUNDS

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
            args=(n11_trunc, E_trunc, trunc, gammaln_n11_1),
            options=opts,
            method=minimization_method,
            bounds=minimization_bounds,
        )

    priors_opt = p_out.x
    if np.any(priors_opt < 0) or priors_opt[4] > 1:
        warnings.warn(
            f"Calculated priors violate distribution constraints. Current priors: {priors_opt}."
        )
    return priors_opt, str(p_out.message)


def _compute_posterior_gps(
    n11: np.ndarray,
    expected: np.ndarray,
    priors: np.ndarray,
    relative_risk: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute EBGM, EB05, EB95, and posterior probability P(H0) from fitted priors."""
    a1, b1, a2, b2, w = priors
    p1 = np.clip(b1 / (b1 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    p2 = np.clip(b2 / (b2 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
    gl_post = gammaln(n11 + 1.0)

    q1 = np.exp(gammaln(n11 + a1) - gl_post - gammaln(a1) + a1 * np.log(p1) + n11 * np.log(1.0 - p1))
    q2 = np.exp(gammaln(n11 + a2) - gl_post - gammaln(a2) + a2 * np.log(p2) + n11 * np.log(1.0 - p2))
    den = w * q1 + (1.0 - w) * q2
    Qn = np.where(den > 0, w * q1 / np.where(den > 0, den, 1.0), w)

    # Exact Gamma CDF P(lambda <= relative_risk | posterior): gdtr(rate, shape, x)
    posterior_probability = (
        Qn * gdtr(np.maximum(b1 + expected, 1e-10), a1 + n11, relative_risk)
        + (1.0 - Qn) * gdtr(np.maximum(b2 + expected, 1e-10), a2 + n11, relative_risk)
    )

    eblog2 = (
        Qn * (digamma(a1 + n11) - np.log(np.maximum(b1 + expected, 1e-10)))
        + (1.0 - Qn) * (digamma(a2 + n11) - np.log(np.maximum(b2 + expected, 1e-10)))
    ) / np.log(2.0)
    ebgm = np.power(2.0, np.asarray(eblog2, dtype=np.float64))

    EB05 = quantiles(0.05, Qn, a1 + n11, b1 + expected, a2 + n11, b2 + expected)
    EB95 = quantiles(0.95, Qn, a1 + n11, b1 + expected, a2 + n11, b2 + expected)

    return ebgm, EB05, EB95, posterior_probability


def _build_gps_result(
    mat_shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    products: np.ndarray,
    aes: np.ndarray,
    n11: np.ndarray,
    n10: np.ndarray,
    n01: np.ndarray,
    n00: np.ndarray,
    ebgm: np.ndarray,
    EB05: np.ndarray,
    EB95: np.ndarray,
    posterior_probability: np.ndarray,
    priors: np.ndarray,
    code_convergence: str,
    criterion: str,
    extra_params: dict | None = None,
) -> Container:
    """Assemble output Container with sparse matrices, metrics (FDR, Se, FNR, Sp), and signals."""
    RES = Container(params=True)
    RES.param["priors"] = priors
    RES.param["convergence"] = code_convergence
    if extra_params:
        RES.param.update(extra_params)

    RES.sparse_ebgm = sparse.coo_matrix((ebgm, (rows, cols)), shape=mat_shape)
    RES.sparse_eb05 = sparse.coo_matrix((EB05, (rows, cols)), shape=mat_shape)

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

    RES.all_signals = RES.all_signals.sort_values(by=["EB05"], ascending=False).reset_index(drop=True)

    p_h0 = RES.all_signals["p_{H0}"].to_numpy(dtype=np.float64)
    p_h1 = 1.0 - p_h0

    total_true_signals = np.sum(p_h1)
    total_true_negatives = np.sum(p_h0)

    true_positives_cum = np.cumsum(p_h1)
    false_positives_cum = np.cumsum(p_h0)
    post_range = np.arange(1, len(p_h0) + 1, dtype=np.float64)

    RES.all_signals["FDR"] = false_positives_cum / post_range
    RES.all_signals["Se"] = true_positives_cum / (total_true_signals + 1e-7)
    RES.all_signals["FNR"] = (total_true_signals - true_positives_cum) / (total_true_signals + 1e-7)
    RES.all_signals["Sp"] = (total_true_negatives - false_positives_cum) / (total_true_negatives + 1e-7)

    if criterion == "EB05>=2":
        signal_mask = RES.all_signals["EB05"] >= 2.0
    elif criterion == "EB05 > 1 & EBGM >=2":
        signal_mask = (RES.all_signals["EBGM"] >= 2.0) & (RES.all_signals["EB05"] > 1.0)
    else:
        signal_mask = pd.Series(True, index=RES.all_signals.index)

    RES.signals = RES.all_signals[signal_mask].copy()
    RES.num_signals = len(RES.signals)
    return RES


def gps(
    container: Container,
    min_events: int = 4,
    truncate: bool = False,
    maxiter: int = 500,
    criterion: str = "EB05>=2",
    prior_param: list[float] | np.ndarray | None = None,
) -> Container:
    """Compute Multi-item Gamma Poisson Shrinkage (GPS) aligned with signal_methods.py."""
    relative_risk = 1.0
    truncate_thres = 1.0
    expected_method = "mantel-haentzel"
    method_alpha = 1.0
    minimization_method = "SLSQP"
    minimization_bounds = GPS_BOUNDS
    minimization_options = {"maxiter": maxiter}

    prior_init = {
        "alpha1": float(GPS_INIT[0]),
        "beta1": float(GPS_INIT[1]),
        "alpha2": float(GPS_INIT[2]),
        "beta2": float(GPS_INIT[3]),
        "w": float(GPS_INIT[4]),
    }

    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "truncate": truncate,
        "truncate_thres": truncate_thres,
        "expected_method": expected_method,
        "method_alpha": method_alpha,
        "minimization_method": minimization_method,
    }

    # 1. Extract sparse contingency matrix and report-level marginal totals
    mat, row_totals, col_totals, products, aes, N = _extract_sparse_data(container)

    n11 = np.asarray(mat.data, dtype=np.float64)
    rows = mat.row
    cols = mat.col
    n1j = row_totals[rows]
    ni1 = col_totals[cols]
    expected = calculate_expected(N, n1j, ni1, n11, expected_method, method_alpha)

    # 2. Prior estimation (or direct reuse of precomputed priors)
    if prior_param is None:
        priors, code_convergence = _optimize_gps_priors(
            priors=GPS_INIT.copy(),
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11,
            expected=expected,
            minimization_method=minimization_method,
            minimization_bounds=minimization_bounds,
            minimization_options=minimization_options,
        )
    else:
        priors = np.asarray(prior_param, dtype=np.float64)
        code_convergence = "User-provided priors"

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

    # 3. Filter cells below min_events
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

    # 4. Posterior calculations & output Container
    ebgm, EB05, EB95, posterior_probability = _compute_posterior_gps(
        n11, expected, priors, relative_risk=relative_risk
    )

    return _build_gps_result(
        mat_shape=mat.shape,
        rows=rows,
        cols=cols,
        products=products,
        aes=aes,
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        ebgm=ebgm,
        EB05=EB05,
        EB95=EB95,
        posterior_probability=posterior_probability,
        priors=priors,
        code_convergence=code_convergence,
        criterion=criterion,
        extra_params={
            "input_params": input_params,
            "computation_params": computation_params,
        },
    )


def gps_sampled(
    container: Container,
    max_points: int = 150_000,
    seed: int = 0,
    min_events: int = 4,
    truncate: bool = False,
    truncate_thres: float = 1.0,
    maxiter: int = 500,
    criterion: str = "EB05>=2",
    prior_param: list[float] | np.ndarray | None = None,
) -> Container:
    """Subsampled GPS strictly matching fit_gps_priors & gps_stats in signal_methods.py."""
    # 1. Extract sparse contingency matrix and report-level marginal totals
    mat, row_totals, col_totals, products, aes, N = _extract_sparse_data(container)

    n11 = np.asarray(mat.data, dtype=np.float64)
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
    if prior_param is None:
        if max_points is not None and total_pairs > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(total_pairs, max_points, replace=False)
            n11_fit, E_fit = n11_fit[idx], E_fit[idx]

        priors, code_convergence = _optimize_gps_priors(
            priors=GPS_INIT.copy(),
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11_fit,
            expected=E_fit,
            minimization_method="SLSQP",
            minimization_bounds=GPS_BOUNDS,
            minimization_options={"maxiter": maxiter},
        )
        fitted_on = len(n11_fit)
    else:
        priors = np.asarray(prior_param, dtype=np.float64)
        code_convergence = "User-provided priors"
        fitted_on = 0

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

    # 3. Posterior calculations & output Container
    ebgm, EB05, EB95, posterior_probability = _compute_posterior_gps(
        n11, expected, priors, relative_risk=1.0
    )

    return _build_gps_result(
        mat_shape=mat.shape,
        rows=rows,
        cols=cols,
        products=products,
        aes=aes,
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        ebgm=ebgm,
        EB05=EB05,
        EB95=EB95,
        posterior_probability=posterior_probability,
        priors=priors,
        code_convergence=code_convergence,
        criterion=criterion,
        extra_params={
            "sampling_info": {
                "max_points": max_points,
                "seed": seed,
                "total_candidate_pairs": total_pairs,
                "fitted_on": fitted_on,
            }
        },
    )
