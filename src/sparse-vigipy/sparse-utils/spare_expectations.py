import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.sparse import issparse, csr_matrix
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def __cameron_trivedi_dispersion(y, mu):
    """
    Vectorized dispersion test from Cameron and Trivedi's 1990 paper:

    Cameron, A. and Trivedi, Pravin, (1990), Regression-based tests for
    overdispersion in the Poisson model, Journal of Econometrics,
    46, issue 3, p. 347-364, https://doi.org.

    Arguments:
        y (array-like): Observed event counts
        mu (array-like): Fitted mean values from Poisson GLM

    Returns:
        (np.ndarray) The Cameron-Trivedi response variable
    """
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    return ((y_arr - mu_arr) ** 2 - y_arr) / mu_arr


def __test_dispersion(model, events):
    """
    Apply an auxiliary formula to run linear regression for the dispersion
    test. If the lower_bound is > 0, overdispersion is most likely confirmed.
    The alpha parameter can be utilized in a negative binomial family GLM to
    better model the data.

    Arguments:
        model (statsmodels model): A Poisson family GLM fitted model
        events (array-like): Observed event counts

    Returns:
        alpha: The optimized alpha term for a negative binomial model
        lower_bound: The 5% confidence interval value
        upper_bound: The 95% confidence interval value
    """
    mu = np.asarray(model.mu, dtype=np.float64)
    response = __cameron_trivedi_dispersion(events, mu)

    aux_df = pd.DataFrame({"response": response, "mu": mu})
    results = smf.ols("response ~ mu - 1", aux_df).fit()

    alpha_conf_int = results.conf_int(0.05).loc["mu"]
    alpha = float(results.params["mu"])
    lower_bound = float(alpha_conf_int.loc[0])
    upper_bound = float(alpha_conf_int.loc[1])

    return alpha, lower_bound, upper_bound


def __mh(N, n1j, ni1):
    """
    The Mantel-Haentzel calculation for expected counts.
    Supports 1D arrays, scalars, and SciPy sparse matrices.
    """
    if issparse(n1j) or issparse(ni1):
        # Sparse outer/element-wise product without densifying
        if issparse(n1j) and issparse(ni1):
            return n1j.multiply(ni1) / N
        return csr_matrix(n1j).multiply(ni1) / N

    return np.asarray(n1j, dtype=np.float64) * np.asarray(ni1, dtype=np.float64) / N


def __stats_method(n1j, ni1, n11, family):
    """
    Calculate expected counts via a GLM statistical model on non-zero entries.
    Expected counts are modeled as a function of n1j and ni1.

    Arguments:
        n1j (iterable or sparse): All adverse events for a single product
        ni1 (iterable or sparse): Total count of a particular AE across all products
        n11 (iterable or sparse): Total count of a particular AE for a particular product
        family (statsmodels family): The GLM family

    Returns:
        np.ndarray: The expected counts for n11
    """
    n11_arr = n11.data if issparse(n11) else np.asarray(n11, dtype=np.float64).ravel()
    n1j_arr = n1j.data if issparse(n1j) else np.asarray(n1j, dtype=np.float64).ravel()
    ni1_arr = ni1.data if issparse(ni1) else np.asarray(ni1, dtype=np.float64).ravel()

    # Build design matrix directly with NumPy/statsmodels (avoids formula parsing overhead)
    exog = np.column_stack([np.ones_like(n1j_arr), n1j_arr, ni1_arr])
    model = sm.GLM(n11_arr, exog, family=family).fit()

    if isinstance(family, sm.families.Poisson):
        dispersion = model.pearson_chi2 / model.df_resid
        if dispersion > 2:
            alpha, lb, ub = __test_dispersion(model, n11_arr)
            warning_string = (
                f"Variance does not equal the mean! Data likely overdispersed...\n"
                f"Consider utilizing the negative-binomial family instead of poisson.\n"
                f"Cameron-Trivedi alpha: {alpha:5.4f}, CI: ({lb}, {ub})"
            )
            warnings.warn(warning_string)

    return model.predict(exog)


def test_dispersion(container):
    """
    Return the Cameron-Trivedi dispersion value and the alpha value for use
    in a negative-binomial GLM. It is recommended to avoid Poisson GLMs when
    dispersion > 2.

    Arguments:
        container: A Container object produced by convert() from data_prep.py

    Returns:
        dict: Dispersion value, alpha value, lower bound, and upper bound of alpha
    """
    # Extract non-zero observations directly from sparse contingency matrix if available
    if hasattr(container, "contingency") and issparse(container.contingency):
        coo = container.contingency.tocoo()
        row_sums = np.asarray(container.contingency.sum(axis=1)).ravel()
        col_sums = np.asarray(container.contingency.sum(axis=0)).ravel()

        events = coo.data.astype(np.float64)
        prod_aes = row_sums[coo.row]
        ae_across_brands = col_sums[coo.col]
    else:
        events = container.data["events"].to_numpy(dtype=np.float64)
        prod_aes = container.data["product_aes"].to_numpy(dtype=np.float64)
        ae_across_brands = container.data["count_across_brands"].to_numpy(dtype=np.float64)

    exog = np.column_stack([np.ones_like(prod_aes), prod_aes, ae_across_brands])
    model = sm.GLM(events, exog, family=sm.families.Poisson()).fit()

    dispersion = float(model.pearson_chi2 / model.df_resid)
    alpha, lb, ub = __test_dispersion(model, events)
    return {"dispersion": dispersion, "alpha": alpha, "lb": lb, "ub": ub}


def calculate_expected(N, n1j, ni1, n11, method="mantel-haentzel", alpha=1):
    """
    Calculate the expected counts for disproportionality analysis.

    Arguments:
        N (int or float): Total number of adverse events across all products
        n1j (iterable or sparse): All adverse events for a single product
        ni1 (iterable or sparse): Total count of a particular AE across all products
        n11 (iterable or sparse): Total count of a particular AE for a particular product
        method (str): Method for calculating expected counts
        alpha (float): Dispersion parameter if using a NegativeBinomial family

    Returns:
        np.ndarray or csr_matrix: The expected counts for n11
    """
    supported = ("mantel-haentzel", "negative-binomial", "poisson")
    if method not in supported:
        raise AssertionError(f"{method} not a supported method. Please choose from {supported}")

    if method == "mantel-haentzel":
        return __mh(N, n1j, ni1)
    elif method == "negative-binomial":
        try:
            return __stats_method(n1j, ni1, n11, sm.families.NegativeBinomial(alpha=alpha))
        except PerfectSeparationError:
            print("Perfect separation of data detected. Defaulting to Mantel-Haentzel estimation.")
            return __mh(N, n1j, ni1)
    elif method == "poisson":
        try:
            return __stats_method(n1j, ni1, n11, sm.families.Poisson())
        except PerfectSeparationError:
            print("Perfect separation of data detected. Defaulting to Mantel-Haentzel estimation.")
            return __mh(N, n1j, ni1)
