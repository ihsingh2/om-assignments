from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np

def line_search(
    initial_point: NDArray[np.float64],
    d_k: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]]
) -> np.float64:

    x_k = initial_point
    alpha = 0
    beta = 1e6
    t = 1
    c_1 = 0.001
    c_2 = 0.1

    while beta - alpha > 1e-8:
        if f(x_k + t * d_k) > f(x_k) + c_1 * t * np.dot(d_f(x_k), d_k):
            beta = t
            t = (alpha + beta) / 2
        elif np.dot(d_f(x_k + t * d_k), d_k) < c_2 * np.dot(d_f(x_k).T, d_k):
            alpha = t
            if beta == 1e6:
                t = 2 * alpha
            else:
                t = (alpha + beta) / 2
        else:
            return t

    return t

def conjugate_descent(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:

    k = 0
    r = 0
    n = len(initial_point)
    eps = 1e-6
    x_k = initial_point
    g_k = d_f(x_k)
    d_k = -g_k

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(g_k) > eps:
        if r == n:
            d_k = -d_f(x_k)
            r = 0

        alpha_k = line_search(x_k, d_k, f, d_f)
        x_k = x_k + alpha_k * d_k

        if (d_f(x_k) == g_k).all():
            r = n
            continue

        if approach == "Hestenes-Stiefel":
            beta_k = np.dot(d_f(x_k), d_f(x_k) - g_k) / np.dot(d_k, d_f(x_k) - g_k)
        elif approach == "Polak-Ribiere":
            beta_k = np.dot(d_f(x_k), d_f(x_k) - g_k) / np.dot(g_k, g_k)
        elif approach == "Fletcher-Reeves":
            beta_k = np.dot(d_f(x_k), d_f(x_k)) / np.dot(g_k, g_k)
        else:
            return None

        g_k = d_f(x_k)
        d_k = -g_k + beta_k * d_k
        k = k + 1
        r = r + 1

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

    plot_graphs(f, initial_point, approach.lower(), f_x_i, d_f_x_i, x_0_i, x_1_i)

    return x_k

def sr1(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point
    g_k = d_f(x_k)
    H_k = np.eye(len(initial_point))

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(g_k) > eps:
        d_k = - H_k.dot(g_k)
        alpha_k = line_search(x_k, d_k, f, d_f)

        delta_k = alpha_k * d_k
        gamma_k = d_f(x_k + alpha_k * d_k) - g_k
        x_k = x_k + alpha_k * d_k
        g_k = d_f(x_k)

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

        z = delta_k - H_k.dot(gamma_k)
        denom = np.dot(z, gamma_k)
        if denom == 0:
            denom = 1e-9
        H_k = H_k + np.outer(z, z) / denom
        k = k + 1

    plot_graphs(f, initial_point, "sr1", f_x_i, d_f_x_i, x_0_i, x_1_i)

    return x_k

def dfp(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point
    g_k = d_f(x_k)
    H_k = np.eye(len(initial_point))

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(g_k) > eps:
        d_k = - H_k.dot(g_k)
        alpha_k = line_search(x_k, d_k, f, d_f)

        delta_k = alpha_k * d_k
        gamma_k = d_f(x_k + alpha_k * d_k) - g_k
        x_k = x_k + alpha_k * d_k
        g_k = d_f(x_k)

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

        term1 = np.outer(delta_k, delta_k) / np.dot(delta_k, gamma_k)
        term2 = H_k.dot(np.outer(gamma_k, gamma_k)).dot(H_k) / np.dot(gamma_k, H_k.dot(gamma_k))
        H_k = H_k + term1 - term2
        k = k + 1

    plot_graphs(f, initial_point, "dfp", f_x_i, d_f_x_i, x_0_i, x_1_i)

    return x_k

def bfgs(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:

    k = 0
    eps = 1e-6
    x_k = initial_point
    g_k = d_f(x_k)
    H_k = np.eye(len(initial_point))

    f_x_i = [ f(x_k), ]
    d_f_x_i = [ np.linalg.norm(d_f(x_k)), ]
    x_0_i = [ x_k[0], ]
    x_1_i = [ x_k[1], ]

    while k <= 1e4 and np.linalg.norm(g_k) > eps:
        d_k = - H_k.dot(g_k)
        alpha_k = line_search(x_k, d_k, f, d_f)

        delta_k = alpha_k * d_k
        gamma_k = d_f(x_k + alpha_k * d_k) - g_k
        x_k = x_k + alpha_k * d_k
        g_k = d_f(x_k)

        f_x_i.append(f(x_k))
        d_f_x_i.append(np.linalg.norm(d_f(x_k)))
        if len(x_k) == 2:
            x_0_i.append(x_k[0])
            x_1_i.append(x_k[1])

        term1 = np.outer(delta_k, delta_k) / np.dot(delta_k, gamma_k)
        term1 *= 1 + (np.dot(gamma_k, H_k.dot(gamma_k)) / np.dot(delta_k, gamma_k))
        term2 = np.matmul(np.outer(delta_k, gamma_k), H_k) + np.matmul(H_k, np.outer(gamma_k, delta_k))
        term2 /= np.dot(delta_k, gamma_k)
        H_k = H_k + term1 - term2
        k = k + 1

    plot_graphs(f, initial_point, "bfgs", f_x_i, d_f_x_i, x_0_i, x_1_i)

    return x_k

def plot_graphs(f, initial_point, condition, f_x_i, d_f_x_i, x_0_i, x_1_i):

    plt.plot(range(len(f_x_i)), f_x_i)
    plt.xlabel("Iteration, k")
    plt.ylabel("Function value, f(x_k)")
    plt.grid(True)
    plt.savefig(f"plots/{f.__name__.lower()}_{np.array2string(initial_point)}_{condition.lower()}_vals.png")
    plt.close()

    plt.plot(range(len(d_f_x_i)), d_f_x_i)
    plt.xlabel("Iteration, k")
    plt.ylabel("Gradient value, |f'(x_k)|")
    plt.grid(True)
    plt.savefig(f"plots/{f.__name__.lower()}_{np.array2string(initial_point)}_{condition.lower()}_grad.png")
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
        plt.quiver(x_0_i[:-1], x_1_i[:-1], angles_0, angles_1, angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=20, headlength=15)

        plt.xlabel("x_0")
        plt.ylabel("x_1")
        plt.savefig(f"plots/{f.__name__.lower()}_{np.array2string(initial_point)}_{condition.lower()}_cont.png")
        plt.close()
