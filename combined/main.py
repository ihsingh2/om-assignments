import os
from typing import Callable, Literal

from prettytable import PrettyTable
import numpy as np
import numpy.typing as npt

from algos import descent
from functions import (
    trid_function,
    trid_function_derivative,
    trid_function_hessian,
    three_hump_camel_function,
    three_hump_camel_function_derivative,
    three_hump_camel_function_hessian,
    rosenbrock_function,
    rosenbrock_function_derivative,
    rosenbrock_function_hessian,
    styblinski_tang_function,
    styblinski_tang_function_derivative,
    styblinski_tang_function_hessian,
    func_1,
    func_1_derivative,
    func_1_hessian,
)

test_cases = [
    [
        trid_function,
        trid_function_derivative,
        trid_function_hessian,
        np.asarray([-2.0, -2]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        three_hump_camel_function_hessian,
        np.asarray([-2.0, -1]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        rosenbrock_function_hessian,
        np.asarray([-2.0, 2, 2, 2]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        styblinski_tang_function_hessian,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        func_1,
        func_1_derivative,
        func_1_hessian,
        np.asarray([-0.5, 0.5]),
    ]
]


def main():
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    for test_case_num, test_case in enumerate(test_cases):
        descent(
            test_case[0], test_case[1], test_case[2], test_case[3]
        )

if __name__ == "__main__":
    main()
