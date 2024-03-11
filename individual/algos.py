from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:

    x_k = initial_point
    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    if condition == "Backtracking":
        alpha_0 = 10.0
        rho = 0.75
        c = 0.001
        eps = 1e-6
        k = 0

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            d_k = - d_f(x_k)

            alpha = alpha_0
            while (f(x_k + alpha * d_k) > f(x_k) + c * alpha * d_f(x_k) * d_k).all():
                alpha = rho * alpha

            x_k = x_k + alpha * d_k
            k = k + 1

            f_x_i.append(f(x_k))
            d_f_x_i.append(np.linalg.norm(d_f(x_k)))
            if len(x_k) == 2:
                x_0_i.append(x_k[0])
                x_1_i.append(x_k[1])

    elif condition == "Bisection":
        c_1 = 0.001
        c_2 = 0.1
        alpha_0 = 0
        t = 1
        beta_0 = 1e6
        k = 0
        eps = 1e-6

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            d_k = - d_f(x_k)

            alpha = alpha_0
            beta = beta_0
            while True:
                if (f(x_k + t * d_k) > f(x_k) + c_1 * t * d_f(x_k) * d_k).all():
                    beta = t
                    t = (alpha + beta) / 2
                elif (d_f(x_k + t * d_k) * d_k < c_2 * d_f(x_k) * d_k).all():
                    alpha = t
                    t = (alpha + beta) / 2
                else:
                    break
            x_k = x_k + t * d_k
            k = k + 1

            f_x_i.append(f(x_k))
            d_f_x_i.append(np.linalg.norm(d_f(x_k)))
            if len(x_k) == 2:
                x_0_i.append(x_k[0])
                x_1_i.append(x_k[1])

    plot_graphs(f, initial_point, condition, f_x_i, d_f_x_i, x_0_i, x_1_i)

    return x_k

def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:

    x_k = initial_point
    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    if condition == "Pure":
        eps = 1e-6
        k = 0

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]
            x_k = x_k + d_k
            k = k + 1

            f_x_i.append(f(x_k))
            d_f_x_i.append(np.linalg.norm(d_f(x_k)))
            if len(x_k) == 2:
                x_0_i.append(x_k[0])
                x_1_i.append(x_k[1])

    elif condition == "Damped":
        alpha = 0.001
        beta = 0.75
        eps = 1e-6
        k = 0

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]

            t_k = 1
            while (f(x_k) - f(x_k + t_k * d_k) < - alpha * t_k * d_f(x_k) * d_k).all():
                t_k = beta * t_k

            x_k = x_k + t_k * d_k
            k = k + 1

            f_x_i.append(f(x_k))
            d_f_x_i.append(np.linalg.norm(d_f(x_k)))
            if len(x_k) == 2:
                x_0_i.append(x_k[0])
                x_1_i.append(x_k[1])

    elif condition == "Levenberg-Marquardt":
        eps = 1e-6
        k = 0

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

    elif condition == "Combined":
        alpha_0 = 10.0
        rho = 0.75
        c = 0.001
        eps = 1e-6
        k = 0

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            lambda_min = np.min(np.linalg.eigvalsh(d2_f(x_k)))
            if lambda_min <= 0:
                mu_k = - lambda_min + 0.1
                d_k = np.linalg.lstsq(d2_f(x_k) + np.diag(np.full(len(x_k), mu_k)), - d_f(x_k), rcond=None)[0]
            else:
                d_k = np.linalg.lstsq(d2_f(x_k), - d_f(x_k), rcond=None)[0]

            alpha_k = alpha_0
            while (f(x_k + alpha_k * d_k) > f(x_k) + c * alpha_k * d_f(x_k) * d_k).all():
                alpha_k = rho * alpha_k

            x_k = x_k + alpha_k * d_k
            k = k + 1

            f_x_i.append(f(x_k))
            d_f_x_i.append(np.linalg.norm(d_f(x_k)))
            if len(x_k) == 2:
                x_0_i.append(x_k[0])
                x_1_i.append(x_k[1])

    plot_graphs(f, initial_point, condition, f_x_i, d_f_x_i, x_0_i, x_1_i)

    return x_k

def plot_graphs(f, initial_point, condition, f_x_i, d_f_x_i, x_0_i, x_1_i):

    plt.plot(range(len(f_x_i)), f_x_i)
    plt.xlabel("Iteration, k")
    plt.ylabel("Function value, f(x_k)")
    plt.grid(True)
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_{condition.lower()}_vals.png")
    plt.close()

    plt.plot(range(len(d_f_x_i)), d_f_x_i)
    plt.xlabel("Iteration, k")
    plt.ylabel("Gradient value, |f'(x_k)|")
    plt.grid(True)
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_{condition.lower()}_grad.png")
    plt.close()

    if len(initial_point) == 2:
        w0 = np.linspace(initial_point[0] * -5, initial_point[0] * 5, 100)
        w1 = np.linspace(initial_point[1] * -5, initial_point[1] * 5, 100)
        z = np.zeros(shape=(100, 100))
        for i, ii in enumerate(w0):
           for j, jj in enumerate(w1):
               z[i][j] = f(np.array([ii, jj]))
        plt.contour(w0, w1, z)

        angles_0 = np.array(x_0_i)[1:] - np.array(x_0_i)[:-1]
        angles_1 = np.array(x_1_i)[1:] - np.array(x_1_i)[:-1]
        plt.quiver(x_0_i[:-1], x_1_i[:-1], angles_0, angles_1, angles='xy', width=0.001, headwidth=20, headlength=15)

        plt.xlabel("x_0")
        plt.ylabel("x_1")
        plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_{condition.lower()}_cont.png")
        plt.close()
