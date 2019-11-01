import warnings

import numpy as np
from scipy.linalg import svd, inv


def gauss_seidel(A: np.ndarray, b: np.ndarray, max_it=100, max_res=1e-10, verbose=False):
    """
    Solve a matrix equation in the form Ax = b using the Gauss-Seidel iterative method.
    :param A:
    :param b:
    :return: x
    """
    assert (A.shape[0] == A.shape[1])
    assert (A.shape[0] == len(b))
    n = A.shape[0]
    diag = np.diag(A)
    b.resize(diag.shape)
    b = b / diag
    A = A / diag
    A_L = -np.tril(A, -1)  # Strictly lower triangular
    A_U = -np.triu(A, 1)  # Strictly upper triangular

    its = 0
    res = 1  # Arbitrary
    x = np.zeros((n,))
    while res > max_res and its < max_it:
        for k in range(n):
            x[k] = np.dot(A_L[k], x) + np.dot(A_U[k], x) + b[k]
        its += 1
        res = np.max(np.abs(b - np.dot(A, x)))
        if verbose:
            print(f"it = {its}, res = {res}")

    if res > max_res:
        warnings.warn(f"Gauss-Seidel failed to converge, residual = {res:.3e}")
        return None

    return x


def svd_solve(A: np.ndarray, b: np.ndarray):
    """
    Solve a matrix equation in the form Ax = b using Singular Value Decomposition.
    :param A:
    :param b:
    :return: x
    """
    U, s, Vh = svd(A)
    return np.dot(Vh.T, np.dot(inv(np.diag(s)), np.dot(U.T, b)))
