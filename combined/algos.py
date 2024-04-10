from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def armijo_line_search(
    initial_point: npt.NDArray[np.float64],
    d_k: npt.NDArray[np.float64],
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    alpha: np.float64 = 10.0,
    rho: np.float64 = 0.75,
    c: np.float64 = 0.001
) -> np.float64:

    x_k = initial_point
    f_k = f(x_k)
    g_k = d_f(x_k)

    while f(x_k + alpha * d_k) > f_k + c * alpha * np.dot(g_k, d_k):
        alpha = rho * alpha

    return alpha

def wolfe_line_search(
    initial_point: npt.NDArray[np.float64],
    d_k: npt.NDArray[np.float64],
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    c_1: np.float64 = 0.001,
    c_2: np.float64 = 0.1
) -> np.float64:

    alpha = 0
    beta = 1e6
    t = 1

    x_k = initial_point
    f_k = f(x_k)
    g_k = d_f(x_k)

    while True:
        if f(x_k + t * d_k) > f_k + c_1 * t * np.dot(g_k, d_k):
            beta = t
            t = (alpha + beta) / 2
        elif np.dot(d_f(x_k + t * d_k), d_k) < c_2 * np.dot(g_k, d_k):
            alpha = t
            t = (alpha + beta) / 2
        else:
            break

    return t

def descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    approach = [
        "Backtracking",
        "Bisection",
        "Pure",
        "Damped",
        "Levenberg-Marquardt",
        "Combined"
    ]
    functions = [
        steepest_descent_backtracking,
        steepest_descent_bisection,
        pure_newton_method,
        damped_newton_method,
        levenberg_marquardt,
        combined_newton_method
    ]

    f_X_i = [ ]
    d_f_X_i = [ ]
    X_0_i = [ ]
    X_1_i = [ ]

    for function in functions:
        f_x_i, d_f_x_i, x_0_i, x_1_i = function(f, d_f, d2_f, initial_point)
        f_X_i.append(f_x_i)
        d_f_X_i.append(d_f_x_i)
        if len(initial_point) == 2:
            X_0_i.append(x_0_i)
            X_1_i.append(x_1_i)

    plot_graphs(f, initial_point, approach, f_X_i, d_f_X_i, X_0_i, X_1_i)

def steepest_descent_backtracking(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
        d_k = - d_f(x_k)
        alpha_k = armijo_line_search(x_k, d_k, f, d_f)
        x_k = x_k + alpha_k * d_k
        k = k + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    return f_x_i, d_f_x_i, x_0_i, x_1_i

def steepest_descent_bisection(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
        d_k = - d_f(x_k)
        alpha_k = wolfe_line_search(x_k, d_k, f, d_f)
        x_k = x_k + alpha_k * d_k
        k = k + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    return f_x_i, d_f_x_i, x_0_i, x_1_i

def pure_newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
        d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]
        x_k = x_k + d_k
        k = k + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    return f_x_i, d_f_x_i, x_0_i, x_1_i

def damped_newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
        d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]
        t_k = armijo_line_search(x_k, d_k, f, d_f, alpha=1.0)
        x_k = x_k + t_k * d_k
        k = k + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    return f_x_i, d_f_x_i, x_0_i, x_1_i

def levenberg_marquardt(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
        lambda_min = np.min(np.linalg.eigvalsh(d2_f(x_k)))
        if lambda_min <= 0:
            mu_k = - lambda_min + 0.1
            d_k = np.linalg.lstsq(d2_f(x_k) + np.diag(np.full(len(x_k), mu_k)), - d_f(x_k), rcond=None)[0]
        else:
            d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]

        x_k = x_k + d_k
        k = k + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    return f_x_i, d_f_x_i, x_0_i, x_1_i

def combined_newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
        lambda_min = np.min(np.linalg.eigvalsh(d2_f(x_k)))
        if lambda_min <= 0:
            mu_k = - lambda_min + 0.1
            d_k = np.linalg.lstsq(d2_f(x_k) + np.diag(np.full(len(x_k), mu_k)), - d_f(x_k), rcond=None)[0]
        else:
            d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]

        alpha_k = armijo_line_search(x_k, d_k, f, d_f)
        x_k = x_k + alpha_k * d_k
        k = k + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    return f_x_i, d_f_x_i, x_0_i, x_1_i

def plot_graphs(f, initial_point, condition, f_x_i, d_f_x_i, x_0_i, x_1_i):

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i, cond in enumerate(condition):
        plt.plot(range(len(f_x_i[i])), f_x_i[i], label=cond, color=colors[i])
    plt.xlabel("Iteration, k")
    plt.ylabel("Function value, f(x_k)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_vals.png")
    plt.close()

    for i, cond in enumerate(condition):
        plt.plot(range(len(d_f_x_i[i])), d_f_x_i[i], label=cond, color=colors[i])
    plt.xlabel("Iteration, k")
    plt.ylabel("Gradient value, |f'(x_k)|")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_grad.png")
    plt.close()

    if len(initial_point) == 2:
        w0 = np.linspace(initial_point[0] * -5, initial_point[0] * 5, 100)
        w1 = np.linspace(initial_point[1] * -5, initial_point[1] * 5, 100)
        z = np.zeros(shape=(100, 100))
        for i, ii in enumerate(w0):
           for j, jj in enumerate(w1):
               z[i][j] = f(np.array([ii, jj]))
        plt.contour(w0, w1, z)

        for i, cond in enumerate(condition):
            angles_0 = np.array(x_0_i[i])[1:] - np.array(x_0_i[i])[:-1]
            angles_1 = np.array(x_1_i[i])[1:] - np.array(x_1_i[i])[:-1]
            plt.quiver(x_0_i[i][:-1], x_1_i[i][:-1], angles_0, angles_1, label=cond, color=colors[i], angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=20, headlength=15)

        plt.xlabel("x_0")
        plt.ylabel("x_1")
        plt.legend()
        plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_cont.png")
        plt.close()
