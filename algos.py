from typing import Callable, Literal

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np

def projection(
    y: npt.NDArray[np.float64],
    constraint_type: Literal["linear", "l_2"],
    constraints: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64]
) -> npt.NDArray[np.float64]:

    x = None
    if constraint_type == "linear":
        x = np.maximum(y, constraints[0])
        x = np.minimum(x, constraints[1])
    elif constraint_type == "l_2":
        centre = constraints[0]
        radius = constraints[1]
        z = y - centre
        x = z / max(radius, np.linalg.norm(z))
        x *= radius
        x += centre
    return x

def projected_gd(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    constraint_type: Literal["linear", "l_2"],
    constraints: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64]
) -> npt.NDArray[np.float64]:

    def P_c(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return projection(y, constraint_type, constraints)

    def sufficient_decrement(
        x: npt.NDArray[np.float64],
        t: np.float64
    ) -> bool:
        x_p = P_c(x - t * d_f(x))
        dec = f(x) - f(x_p)
        bound = t * (np.linalg.norm((x - x_p) / t) ** 2)
        return abs(dec - bound) < 1e-3

    eps = 1e-2
    s = 1
    beta = 0.75

    x_k = point

    for k in range(1000):
        t_k = s
        while not sufficient_decrement(x_k, t_k):
            t_k *= beta

        x_new = P_c(x_k - t_k * d_f(x_k))

        if np.linalg.norm(x_new - x_k) <= eps:
            break
        x_k = x_new

    return x_k

def dual_ascent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    c: list[Callable[[npt.NDArray[np.float64]], np.float64 | float]],
    d_c: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    initial_point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    def grad_x(
        x_k: npt.NDArray[np.float64],
        lambda_k: npt.NDArray[np.float64]
    ):
        grad = d_f(x_k)
        for idx, d_c_idx in enumerate(d_c):
            grad = grad + lambda_k[idx] * d_c_idx(x_k)
        return grad

    def grad_lambda(
        x_k: npt.NDArray[np.float64],
        lambda_k: npt.NDArray[np.float64]
    ):
        grad = np.zeros(len(c))
        for idx, c_idx in enumerate(c):
            grad[idx] = c_idx(x_k)
        return grad

    alpha = 1e-2
    zeros = np.zeros(len(c))

    x_k = initial_point
    lambda_k = np.ones(len(c))

    for k in range(10000):
        x_k = x_k - alpha * grad_x(x_k, lambda_k)
        lambda_k = lambda_k + alpha * grad_lambda(x_k, lambda_k)
        lambda_k = np.maximum(lambda_k, zeros)

    return x_k
