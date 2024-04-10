## Brief

Implementation of Steepest Descent and Newton's Methods.

## Methods

### 1. Steepest Descent

```math
\textbf{d}_{k} = - \nabla f(\textbf{x}_{k})
```

- Backtracking with Armijo condition

```math
f(\textbf{x}_k + \alpha \textbf{d}_k) \leq f(\textbf{x}) + c \alpha \nabla f(\textbf{x}_k)^T \textbf{d}_k
```

- Bisection method with Wolfe condition

```math
\nabla f(\textbf{x}_k + \alpha \textbf{d}_k)^T \textbf{d}_k \geq c' \nabla f(\textbf{x}_k)^T \textbf{d}_k
```

### 2. Newton's Method

- Pure Newton's Method

```math
\nabla^2 f(\textbf{x}_k) \textbf{d}_k = - \nabla f(\textbf{x}_k)
```

- Damped Newton's Method

  Calculate step size using Armijo condition.

- Levenberg-Marquardt Modification

```math
\mu_k = \begin{cases}
    - \lambda_{min} + 0.1 & \text{if } \lambda_{min} \leq 0 \\
    \textbf{0} & \text{otherwise}
\end{cases}
```

```math
(\nabla^2 f(\textbf{x}_k) + \mu_k \textbf{I}) \textbf{d}_k = - \nabla f(\textbf{x}_k)
```

- Combining Damped Newton's Method, with Levenberg-Marquardt Modification

  Calculate step size using Armijo condition.

## Functions

- Trid Function

```math
f(\textbf{x}) = \sum_{i=1}^d (x_i - 1)^2 - \sum_{i=2}^d x_{i-1} x_i
```

- Three Hump Camel

```math
f(\textbf{x}) = 2 x_1^2 - 1.05 x_1^4 + \frac{x_1^6}{6} + x_1 x_2 + x_2^2
```

- Styblinski-Tang Function

```math
f(\textbf{x}) = \frac{1}{2} \sum_{i=1}^d (x_i^4 - 16 x_i^2 + 5 x_i)
```

- Rosenbrock Function

```math
f(\textbf{x}) = \sum_{i=1}^{d-1} [ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 ]
```

- Root of Square Function

```math
f(\textbf{x}) = \sqrt{1 + x_1^2} + \sqrt{1 + x_2^2}
```

## File Structure

- ``combined/`` - For generating all the plots of different methods, in a single figure, for each test case. No output is printed.

- ``individual/`` - For generating separate outputs and plots, for each pair of test case and method.

- ``report/`` - Summary of the key results.
