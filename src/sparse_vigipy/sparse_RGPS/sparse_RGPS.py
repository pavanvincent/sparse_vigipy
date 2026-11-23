from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.special import digamma, gdtr, gammaln, betainc
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from ..sparse_utils.sparse_Container import Container
from ..sparse_utils.sparse_distribution_funcs.sparse_quantile_funcs import quantiles

EPS = float(np.finfo(np.float32).eps)

# Initialisation & bornes pour le prior Gamma simple (DuMouchel & Harpaz, 2012) : (alpha, beta)
RGPS_SINGLE_INIT = np.array([1.0, 1.0], dtype=np.float64)
RGPS_SINGLE_BOUNDS = ((EPS, 50.0), (EPS, 50.0))

# Initialisation & bornes si mélange à 2 Gammas activé (use_mixture=True) : (a1, b1, a2, b2, w)
RGPS_MIX_INIT = np.array([0.2041, 0.05816, 1.415, 1.838, 0.0969], dtype=np.float64)
RGPS_MIX_BOUNDS = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))

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


def _extract_individual_sparse_data(
    container: Container,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.coo_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract individual-level sparse matrices (sf in CSC, so in CSC), contingency COO, marginals, labels, and N."""
    has_individual = hasattr(container, "product_features") and hasattr(container, "event_outcomes")
    if not has_individual:
        raise ValueError(
            "RGPS requires an individual-level Container with 'product_features' and 'event_outcomes' "
            "(e.g., from convert_binary(), from_raw_tables(), from_faers_unique_case(), or from_vaers_unique_case())."
        )

    # CSC format for sf enables O(1) column slicing when pruning zero-cooccurrence drugs per AE
    sf = container.get_sparse_features().tocsc()
    so = container.get_sparse_outcomes().tocsc()

    if hasattr(container, "product_names") and container.product_names is not None:
        products = np.asarray(container.product_names)
    elif hasattr(container, "product_index") and container.product_index is not None:
        products = np.asarray(container.product_index)
    elif isinstance(container.product_features, pd.DataFrame):
        products = np.asarray(container.product_features.columns)
    else:
        products = np.arange(sf.shape[1])

    if hasattr(container, "ae_names") and container.ae_names is not None:
        aes = np.asarray(container.ae_names)
    elif hasattr(container, "ae_index") and container.ae_index is not None:
        aes = np.asarray(container.ae_index)
    elif isinstance(container.event_outcomes, pd.DataFrame):
        aes = np.asarray(container.event_outcomes.columns)
    else:
        aes = np.arange(so.shape[1])

    N = float(container.N) if hasattr(container, "N") and container.N is not None else float(sf.shape[0])

    # Report-level marginal counts matching gps.py
    row_totals = np.asarray(sf.sum(axis=0), dtype=np.float64).ravel()
    col_totals = np.asarray(so.sum(axis=0), dtype=np.float64).ravel()

    if hasattr(container, "contingency") and sparse.issparse(container.contingency):
        mat = container.contingency.tocoo()
    else:
        mat = (sf.astype(np.float64).T @ so.astype(np.float64)).tocoo()
    mat.eliminate_zeros()

    return sf, so, mat, row_totals, col_totals, products, aes, N


def _fit_single_ae_elr(
    ae_idx: int,
    target_prods: np.ndarray,
    sf_csc: sparse.csc_matrix,
    sf_csr_T: sparse.csr_matrix,
    so_csc: sparse.csc_matrix,
    cooc_prods: np.ndarray,
    C_reg: float,
    penalty: str,
    max_lr_iter: int,
    tol: float,
    prune_zero_cooc: bool,
) -> np.ndarray:
    """
    Fit Extended Logistic Regression (ELR) for one adverse event `ae_idx` and return
    adjusted expected counts E_ij^adj for `target_prods`.

    Optimizations:
    1. Column pruning: restricts design matrix X to products having >= 1 co-occurrence
       with `ae_idx` (reduces columns by 90%-98% on FAERS/VAERS).
    2. Sparse matrix-vector dot product on `sf_csr_T[target_prods]` only.
    """
    n_reports = sf_csc.shape[0]
    y_col = so_csc.getcol(int(ae_idx)).toarray().ravel()
    y_bin = (y_col > 0).astype(np.int32)
    n_pos = int(y_bin.sum())

    if n_pos == 0 or n_pos == n_reports:
        p_hat = np.full(n_reports, n_pos / max(n_reports, 1), dtype=np.float64)
        e_target = sf_csr_T[target_prods].dot(p_hat)
        return np.maximum(e_target, 1e-6)

    if prune_zero_cooc and len(cooc_prods) > 0 and len(cooc_prods) < sf_csc.shape[1]:
        X_sub = sf_csc[:, cooc_prods].tocsr()
    else:
        X_sub = sf_csc.tocsr()

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        penalty=penalty,
        C=C_reg,
        solver=solver,
        max_iter=max_lr_iter,
        tol=tol,
    )
    clf.fit(X_sub, y_bin)
    p_hat = clf.predict_proba(X_sub)[:, 1]

    # Compute E_ij^adj only for the target products of this AE
    e_target = sf_csr_T[target_prods].dot(p_hat)
    return np.maximum(e_target, 1e-6)


def _compute_regression_expected(
    sf_csc: sparse.csc_matrix,
    so_csc: sparse.csc_matrix,
    full_contingency_csc: sparse.csc_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    C_reg: float = 1.0,
    penalty: str = "l2",
    max_lr_iter: int = 100,
    tol: float = 1e-3,
    prune_zero_cooc: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute Extended Logistic Regression (ELR) expected counts E_ij^adj in parallel
    across all distinct adverse events in `cols`.
    """
    expected_adj = np.empty(len(rows), dtype=np.float64)

    order = np.argsort(cols)
    cols_sorted = cols[order]
    rows_sorted = rows[order]

    unique_aes, split_starts = np.unique(cols_sorted, return_index=True)
    split_ends = np.append(split_starts[1:], len(cols_sorted))

    sf_csr_T = sf_csc.T.tocsr()

    # Pre-extract co-occurring product indices per AE from full contingency matrix
    tasks = []
    for ae_idx, start, end in zip(unique_aes, split_starts, split_ends):
        target_prods = rows_sorted[start:end]
        if prune_zero_cooc:
            c_start = full_contingency_csc.indptr[ae_idx]
            c_end = full_contingency_csc.indptr[ae_idx + 1]
            cooc_prods = full_contingency_csc.indices[c_start:c_end]
        else:
            cooc_prods = np.empty(0, dtype=np.int32)
        tasks.append((int(ae_idx), target_prods, cooc_prods, start, end))

    if n_jobs == 1 or len(tasks) <= 1:
        results_list = [
            _fit_single_ae_elr(
                ae_idx=ae_idx,
                target_prods=target_prods,
                sf_csc=sf_csc,
                sf_csr_T=sf_csr_T,
                so_csc=so_csc,
                cooc_prods=cooc_prods,
                C_reg=C_reg,
                penalty=penalty,
                max_lr_iter=max_lr_iter,
                tol=tol,
                prune_zero_cooc=prune_zero_cooc,
            )
            for ae_idx, target_prods, cooc_prods, _, _ in tasks
        ]
    else:
        # Threading backend avoids pickling large sparse matrices while sklearn/scipy release the GIL
        results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_single_ae_elr)(
                ae_idx=ae_idx,
                target_prods=target_prods,
                sf_csc=sf_csc,
                sf_csr_T=sf_csr_T,
                so_csc=so_csc,
                cooc_prods=cooc_prods,
                C_reg=C_reg,
                penalty=penalty,
                max_lr_iter=max_lr_iter,
                tol=tol,
                prune_zero_cooc=prune_zero_cooc,
            )
            for ae_idx, target_prods, cooc_prods, _, _ in tasks
        )

    for (_, _, _, start, end), e_vals in zip(tasks, results_list):
        expected_adj[order[start:end]] = e_vals

    return expected_adj


# =============================================================================
# LIKELIHOODS (Single Gamma prior & 2-Gamma Mixture prior)
# =============================================================================

def _single_non_truncated_likelihood(
    p: np.ndarray,
    n11: np.ndarray,
    E: np.ndarray,
    gammaln_n11_1: np.ndarray | None = None,
) -> float:
    """Negative log-likelihood for single-Gamma Poisson (Negative Binomial) model."""
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    alpha, beta = p[0], p[1]
    p_nb = np.clip(beta / (beta + E + 1e-10), 1e-10, 1.0 - 1e-10)
    log_dnb = gammaln(n11 + alpha) - gammaln_n11_1 - gammaln(alpha) + alpha * np.log(p_nb) + n11 * np.log(1.0 - p_nb)
    return float(-np.sum(log_dnb))


def _single_truncated_likelihood(
    p: np.ndarray,
    n11: np.ndarray,
    E: np.ndarray,
    truncate: float,
    gammaln_n11_1: np.ndarray | None = None,
) -> float:
    """Negative log-likelihood for truncated single-Gamma Poisson model."""
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    alpha, beta = p[0], p[1]
    p_nb = np.clip(beta / (beta + E + 1e-10), 1e-10, 1.0 - 1e-10)
    log_dnb = gammaln(n11 + alpha) - gammaln_n11_1 - gammaln(alpha) + alpha * np.log(p_nb) + n11 * np.log(1.0 - p_nb)

    if truncate == 0:
        pnb = np.exp(alpha * np.log(p_nb))
    else:
        pnb = betainc(alpha, truncate + 1, p_nb)
    term2 = 1.0 - pnb
    return float(np.sum(-log_dnb + np.log(np.maximum(term2, 1e-7))))


def _mix_non_truncated_likelihood(
    p: np.ndarray,
    n11: np.ndarray,
    E: np.ndarray,
    gammaln_n11_1: np.ndarray | None = None,
) -> float:
    """Negative log-likelihood for 2-Gamma mixture Poisson model (matching gps.py)."""
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    term = (w * np.exp(log_dnb1) + (1.0 - w) * np.exp(log_dnb2)) + 1e-7
    return float(np.sum(-np.log(term)))


def _mix_truncated_likelihood(
    p: np.ndarray,
    n11: np.ndarray,
    E: np.ndarray,
    truncate: float,
    gammaln_n11_1: np.ndarray | None = None,
) -> float:
    """Negative log-likelihood for truncated 2-Gamma mixture Poisson model (matching gps.py)."""
    if gammaln_n11_1 is None:
        gammaln_n11_1 = gammaln(n11 + 1.0)
    p_nb1 = np.clip(p[1] / (p[1] + E + 1e-10), 1e-10, 1.0 - 1e-10)
    p_nb2 = np.clip(p[3] / (p[3] + E + 1e-10), 1e-10, 1.0 - 1e-10)

    r1, r2, w = p[0], p[2], p[4]
    log_dnb1 = gammaln(n11 + r1) - gammaln_n11_1 - gammaln(r1) + r1 * np.log(p_nb1) + n11 * np.log(1.0 - p_nb1)
    log_dnb2 = gammaln(n11 + r2) - gammaln_n11_1 - gammaln(r2) + r2 * np.log(p_nb2) + n11 * np.log(1.0 - p_nb2)
    term1 = w * np.exp(log_dnb1) + (1.0 - w) * np.exp(log_dnb2)

    if truncate == 0:
        pnb1 = np.exp(r1 * np.log(p_nb1))
        pnb2 = np.exp(r2 * np.log(p_nb2))
    else:
        pnb1 = betainc(r1, truncate + 1, p_nb1)
        pnb2 = betainc(r2, truncate + 1, p_nb2)
    term2 = 1.0 - (w * pnb1 + (1.0 - w) * pnb2)
    return float(np.sum(-np.log(np.maximum(term1, 1e-300)) + np.log(np.maximum(term2, 1e-7))))


def _optimize_rgps_priors(
    priors: np.ndarray,
    use_mixture: bool,
    truncate: bool,
    truncate_thres: float,
    n11: np.ndarray,
    expected: np.ndarray,
    minimization_method: str,
    minimization_bounds: tuple[tuple[float, float], ...] | None,
    minimization_options: dict | None,
) -> tuple[np.ndarray, str]:
    """Estimate empirical Bayes hyperprior parameters via maximum likelihood."""
    if minimization_method not in BOUNDED_METHODS:
        minimization_bounds = None
    elif minimization_bounds is None:
        minimization_bounds = RGPS_MIX_BOUNDS if use_mixture else RGPS_SINGLE_BOUNDS

    opts = {"maxiter": 500}
    if minimization_options:
        opts.update(minimization_options)

    if not truncate:
        gammaln_n11_1 = gammaln(n11 + 1.0)
        obj_fn = _mix_non_truncated_likelihood if use_mixture else _single_non_truncated_likelihood
        p_out = minimize(
            obj_fn,
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
        obj_fn = _mix_truncated_likelihood if use_mixture else _single_truncated_likelihood
        p_out = minimize(
            obj_fn,
            x0=priors,
            args=(n11_trunc, E_trunc, trunc, gammaln_n11_1),
            options=opts,
            method=minimization_method,
            bounds=minimization_bounds,
        )

    priors_opt = p_out.x
    if np.any(priors_opt < 0) or (use_mixture and priors_opt[4] > 1):
        warnings.warn(
            f"Calculated priors violate distribution constraints. Current priors: {priors_opt}."
        )
    return priors_opt, str(p_out.message)


def _compute_posterior_rgps(
    n11: np.ndarray,
    expected: np.ndarray,
    priors: np.ndarray,
    use_mixture: bool = False,
    relative_risk: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute EBRRR (EBGM), RRR05 (EB05), RRR95 (EB95), and posterior probability P(H0)."""
    if use_mixture:
        a1, b1, a2, b2, w = priors
        p1 = np.clip(b1 / (b1 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
        p2 = np.clip(b2 / (b2 + expected + 1e-10), 1e-10, 1.0 - 1e-10)
        gl_post = gammaln(n11 + 1.0)

        q1 = np.exp(gammaln(n11 + a1) - gl_post - gammaln(a1) + a1 * np.log(p1) + n11 * np.log(1.0 - p1))
        q2 = np.exp(gammaln(n11 + a2) - gl_post - gammaln(a2) + a2 * np.log(p2) + n11 * np.log(1.0 - p2))
        den = w * q1 + (1.0 - w) * q2
        Qn = np.where(den > 0, w * q1 / np.where(den > 0, den, 1.0), w)

        posterior_probability = (
            Qn * gdtr(np.maximum(b1 + expected, 1e-10), a1 + n11, relative_risk)
            + (1.0 - Qn) * gdtr(np.maximum(b2 + expected, 1e-10), a2 + n11, relative_risk)
        )

        eblog2 = (
            Qn * (digamma(a1 + n11) - np.log(np.maximum(b1 + expected, 1e-10)))
            + (1.0 - Qn) * (digamma(a2 + n11) - np.log(np.maximum(b2 + expected, 1e-10)))
        ) / np.log(2.0)
        ebrrr = np.power(2.0, np.asarray(eblog2, dtype=np.float64))

        rrr05 = quantiles(0.05, Qn, a1 + n11, b1 + expected, a2 + n11, b2 + expected)
        rrr95 = quantiles(0.95, Qn, a1 + n11, b1 + expected, a2 + n11, b2 + expected)
    else:
        alpha, beta = priors[0], priors[1]
        a_post = alpha + n11
        b_post = np.maximum(beta + expected, 1e-10)

        posterior_probability = gdtr(b_post, a_post, relative_risk)

        eblog2 = (digamma(a_post) - np.log(b_post)) / np.log(2.0)
        ebrrr = np.power(2.0, np.asarray(eblog2, dtype=np.float64))

        Qn_ones = np.ones_like(a_post, dtype=np.float64)
        rrr05 = quantiles(0.05, Qn_ones, a_post, b_post, a_post, b_post)
        rrr95 = quantiles(0.95, Qn_ones, a_post, b_post, a_post, b_post)

    return ebrrr, rrr05, rrr95, posterior_probability


def _build_rgps_result(
    mat_shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    products: np.ndarray,
    aes: np.ndarray,
    n11: np.ndarray,
    n10: np.ndarray,
    n01: np.ndarray,
    n00: np.ndarray,
    expected_adj: np.ndarray,
    ebrrr: np.ndarray,
    rrr05: np.ndarray,
    rrr95: np.ndarray,
    posterior_probability: np.ndarray,
    priors: np.ndarray,
    code_convergence: str,
    criterion: str,
    extra_params: dict | None = None,
) -> Container:
    """Assemble output Container aligned with _build_gps_result in gps.py."""
    RES = Container(params=True)
    RES.param["priors"] = priors
    RES.param["convergence"] = code_convergence
    if extra_params:
        RES.param.update(extra_params)

    RES.sparse_ebgm = sparse.coo_matrix((ebrrr, (rows, cols)), shape=mat_shape)
    RES.sparse_eb05 = sparse.coo_matrix((rrr05, (rows, cols)), shape=mat_shape)
    RES.sparse_ebrrr = RES.sparse_ebgm
    RES.sparse_rrr05 = RES.sparse_eb05

    RES.all_signals = pd.DataFrame(
        {
            "Product": products[rows],
            "Adverse Event": aes[cols],
            "EBRRR": ebrrr,
            "RRR05": rrr05,
            "RRR95": rrr95,
            "EBGM": ebrrr,
            "EB05": rrr05,
            "EB95": rrr95,
            "E_{adj}": expected_adj,
            "N_{11}": n11,
            "N_{10}": n10,
            "N_{01}": n01,
            "N_{00}": n00,
            "p_{H0}": posterior_probability,
        }
    )

    RES.all_signals = RES.all_signals.sort_values(by=["RRR05"], ascending=False).reset_index(drop=True)

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

    if criterion in ("EB05>=2", "RRR05>=2"):
        signal_mask = RES.all_signals["RRR05"] >= 2.0
    elif criterion in ("EB05 > 1 & EBGM >=2", "RRR05 > 1 & EBRRR >=2"):
        signal_mask = (RES.all_signals["EBRRR"] >= 2.0) & (RES.all_signals["RRR05"] > 1.0)
    else:
        signal_mask = pd.Series(True, index=RES.all_signals.index)

    RES.signals = RES.all_signals[signal_mask].copy()
    RES.num_signals = len(RES.signals)
    return RES


def rgps(
    container: Container,
    min_events: int = 4,
    truncate: bool = False,
    maxiter: int = 500,
    criterion: str = "RRR05>=2",
    C_reg: float = 1.0,
    penalty: str = "l2",
    use_mixture: bool = False,
    prune_zero_cooc: bool = True,
    max_lr_iter: int = 100,
    lr_tol: float = 1e-3,
    prior_param: list[float] | np.ndarray | None = None,
    n_jobs: int = -1,
) -> Container:
    """
    Compute Regression-Adjusted Gamma-Poisson Shrinker (RGPS) by DuMouchel & Harpaz (2012),
    with column pruning per AE and multi-core joblib parallelization.
    """
    relative_risk = 1.0
    truncate_thres = 1.0
    minimization_method = "SLSQP"
    minimization_bounds = RGPS_MIX_BOUNDS if use_mixture else RGPS_SINGLE_BOUNDS
    minimization_options = {"maxiter": maxiter}

    init_vec = RGPS_MIX_INIT if use_mixture else RGPS_SINGLE_INIT
    if use_mixture:
        prior_init = {
            "alpha1": float(init_vec[0]),
            "beta1": float(init_vec[1]),
            "alpha2": float(init_vec[2]),
            "beta2": float(init_vec[3]),
            "w": float(init_vec[4]),
        }
    else:
        prior_init = {
            "alpha": float(init_vec[0]),
            "beta": float(init_vec[1]),
        }

    input_params = {
        "relative_risk": relative_risk,
        "min_events": min_events,
        "truncate": truncate,
        "truncate_thres": truncate_thres,
        "C_reg": C_reg,
        "penalty": penalty,
        "use_mixture": use_mixture,
        "prune_zero_cooc": prune_zero_cooc,
        "minimization_method": minimization_method,
    }

    # 1. Extract individual-level sparse matrices and contingency COO
    sf_csc, so_csc, mat, row_totals, col_totals, products, aes, N = _extract_individual_sparse_data(container)
    full_contingency_csc = mat.tocsc()

    n11 = np.asarray(mat.data, dtype=np.float64)
    rows = mat.row
    cols = mat.col
    n1j = row_totals[rows]
    ni1 = col_totals[cols]

    # 2. Compute ELR regression-adjusted expected counts E_ij^adj in parallel
    expected_adj = _compute_regression_expected(
        sf_csc=sf_csc,
        so_csc=so_csc,
        full_contingency_csc=full_contingency_csc,
        rows=rows,
        cols=cols,
        C_reg=C_reg,
        penalty=penalty,
        max_lr_iter=max_lr_iter,
        tol=lr_tol,
        prune_zero_cooc=prune_zero_cooc,
        n_jobs=n_jobs,
    )

    # 3. Prior estimation (or direct reuse of precomputed priors)
    if prior_param is None:
        priors, code_convergence = _optimize_rgps_priors(
            priors=init_vec.copy(),
            use_mixture=use_mixture,
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11,
            expected=expected_adj,
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
            "C_reg": C_reg,
            "penalty": penalty,
            "use_mixture": use_mixture,
            "prune_zero_cooc": prune_zero_cooc,
            "n_jobs": n_jobs,
            "minimization_options": minimization_options,
        },
        orient="index",
        columns=["Value"],
    )

    # 4. Filter cells below min_events
    if min_events > 1:
        mask = n11 >= min_events
        n11 = n11[mask]
        rows = rows[mask]
        cols = cols[mask]
        expected_adj = expected_adj[mask]
        n1j = n1j[mask]
        ni1 = ni1[mask]

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    # 5. Posterior calculations & output Container
    ebrrr, rrr05, rrr95, posterior_probability = _compute_posterior_rgps(
        n11=n11,
        expected=expected_adj,
        priors=priors,
        use_mixture=use_mixture,
        relative_risk=relative_risk,
    )

    return _build_rgps_result(
        mat_shape=mat.shape,
        rows=rows,
        cols=cols,
        products=products,
        aes=aes,
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        expected_adj=expected_adj,
        ebrrr=ebrrr,
        rrr05=rrr05,
        rrr95=rrr95,
        posterior_probability=posterior_probability,
        priors=priors,
        code_convergence=code_convergence,
        criterion=criterion,
        extra_params={
            "input_params": input_params,
            "computation_params": computation_params,
        },
    )


def rgps_sampled(
    container: Container,
    max_points: int = 150_000,
    seed: int = 0,
    min_events: int = 4,
    truncate: bool = False,
    truncate_thres: float = 1.0,
    maxiter: int = 500,
    criterion: str = "RRR05>=2",
    C_reg: float = 1.0,
    penalty: str = "l2",
    use_mixture: bool = False,
    prune_zero_cooc: bool = True,
    max_lr_iter: int = 100,
    lr_tol: float = 1e-3,
    prior_param: list[float] | np.ndarray | None = None,
    n_jobs: int = -1,
) -> Container:
    """
    Subsampled RGPS aligned with gps_sampled() in gps.py, with parallel ELR and column pruning.
    """
    # 1. Extract individual-level sparse matrices and contingency COO
    sf_csc, so_csc, mat, row_totals, col_totals, products, aes, N = _extract_individual_sparse_data(container)
    full_contingency_csc = mat.tocsc()

    n11 = np.asarray(mat.data, dtype=np.float64)
    rows = mat.row
    cols = mat.col
    n1j = row_totals[rows]
    ni1 = col_totals[cols]

    if truncate:
        mask_fit = n11 >= truncate_thres
        n11_fit, rows_fit, cols_fit = n11[mask_fit], rows[mask_fit], cols[mask_fit]
    else:
        n11_fit, rows_fit, cols_fit = n11, rows, cols

    total_pairs = len(n11_fit)
    init_vec = RGPS_MIX_INIT if use_mixture else RGPS_SINGLE_INIT
    minimization_bounds = RGPS_MIX_BOUNDS if use_mixture else RGPS_SINGLE_BOUNDS

    # 2. Subsample pairs for prior estimation (and compute ELR only on needed AEs)
    if prior_param is None:
        if max_points is not None and total_pairs > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(total_pairs, max_points, replace=False)
            n11_fit = n11_fit[idx]
            rows_fit = rows_fit[idx]
            cols_fit = cols_fit[idx]

        E_fit = _compute_regression_expected(
            sf_csc=sf_csc,
            so_csc=so_csc,
            full_contingency_csc=full_contingency_csc,
            rows=rows_fit,
            cols=cols_fit,
            C_reg=C_reg,
            penalty=penalty,
            max_lr_iter=max_lr_iter,
            tol=lr_tol,
            prune_zero_cooc=prune_zero_cooc,
            n_jobs=n_jobs,
        )

        priors, code_convergence = _optimize_rgps_priors(
            priors=init_vec.copy(),
            use_mixture=use_mixture,
            truncate=truncate,
            truncate_thres=truncate_thres,
            n11=n11_fit,
            expected=E_fit,
            minimization_method="SLSQP",
            minimization_bounds=minimization_bounds,
            minimization_options={"maxiter": maxiter},
        )
        fitted_on = len(n11_fit)
    else:
        priors = np.asarray(prior_param, dtype=np.float64)
        code_convergence = "User-provided priors"
        fitted_on = 0

    # 3. Filter pairs below min_events BEFORE computing full ELR expected counts for scoring
    if min_events > 1:
        mask = n11 >= min_events
        n11 = n11[mask]
        rows = rows[mask]
        cols = cols[mask]
        n1j = n1j[mask]
        ni1 = ni1[mask]

    expected_adj = _compute_regression_expected(
        sf_csc=sf_csc,
        so_csc=so_csc,
        full_contingency_csc=full_contingency_csc,
        rows=rows,
        cols=cols,
        C_reg=C_reg,
        penalty=penalty,
        max_lr_iter=max_lr_iter,
        tol=lr_tol,
        prune_zero_cooc=prune_zero_cooc,
        n_jobs=n_jobs,
    )

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)

    # 4. Posterior calculations & output Container
    ebrrr, rrr05, rrr95, posterior_probability = _compute_posterior_rgps(
        n11=n11,
        expected=expected_adj,
        priors=priors,
        use_mixture=use_mixture,
        relative_risk=1.0,
    )

    return _build_rgps_result(
        mat_shape=mat.shape,
        rows=rows,
        cols=cols,
        products=products,
        aes=aes,
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        expected_adj=expected_adj,
        ebrrr=ebrrr,
        rrr05=rrr05,
        rrr95=rrr95,
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
                "C_reg": C_reg,
                "penalty": penalty,
                "use_mixture": use_mixture,
                "prune_zero_cooc": prune_zero_cooc,
                "n_jobs": n_jobs,
            }
        },
    )

