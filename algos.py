from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np




def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    ...

def sr1(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    ...


def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    ...

def bfgs(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    ...
