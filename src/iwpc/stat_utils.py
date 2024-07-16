import numpy as np
from numpy._typing import NDArray


def propagate_uncertainty_through_const_power(A: NDArray, a, A_err: NDArray) -> NDArray:
    """
    Calculates the uncertainty on A^a given A and stderr(A)

    Parameters
    ----------
    A
        A list of estimators of a quantity
    a
        A power
    A_err
        The standard error on A

    Returns
    -------
        The standard error on A ** a
    """
    return np.abs(a * A ** (a - 1) * A_err)


def propagate_uncertainty_through_product(A: NDArray, B: NDArray, cov: NDArray) -> NDArray:
    """
    Parameters
    ----------
    A
        A list of estimators of a quantity
    B
        A list of estimators of a second quantity
    cov
        The covariance matrix of A and B

    Returns
    -------
    NDArray
        The uncertainty on A*B
    """
    return A * B * np.sqrt(B**2 * cov[0, 0] + A**2 * cov[1, 1] + 2 * A * B * cov[0, 1])


def propagate_uncertainty_through_ratio(A: NDArray, B: NDArray, cov: NDArray) -> NDArray:
    """
    Parameters
    ----------
    A
        A list of estimators of a quantity
    B
        A list of estimators of a second quantity
    cov
        The covariance matrix of A and B

    Returns
    -------
    NDArray
        The uncertainty on A/B
    """
    return np.abs(A / B) * np.sqrt(cov[0, 0] / A**2 + cov[1, 1] / B**2 - 2 * cov[0, 1] / A / B)


def normalised_weight_sum_uncertainty(As: NDArray, A_errs: NDArray) -> NDArray:
    """
    Given an array of independent weight sums and uncertainties thereof, calculates the uncertainty in the normalised
    weights

    Parameters
    ----------
    As
        A list of independent weight sums
    A_errs
        The standard error on the weight sums

    Returns
    -------
    NDArray
        The error on As / As.sum()
    """
    cov = np.asarray([[A_errs**2, A_errs**2], [A_errs**2, np.ones_like(A_errs) * (A_errs**2).sum()]])
    return propagate_uncertainty_through_ratio(As, As.sum(), cov)
