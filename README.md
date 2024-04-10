## Brief

Implementation of Conjugate Gradient and Quasi-Newton methods.

## Methods

### 1. Conjugate Gradient Methods

```math
\textbf{d}_{k+1} = - \textbf{g}_{k+1} + \beta_k \textbf{d}_k
```

- Hestenes-Stiefel Approach

```math
\beta_k = \frac{\textbf{g}_{k+1}^T (\textbf{g}_{k+1} - \textbf{g}_k)}{\textbf{d}_k^T (\textbf{g}_{k+1} - \textbf{g}_k)}
```

- Polak-Ribiere Approach

```math
\beta_k = \frac{\textbf{g}_{k+1}^T (\textbf{g}_{k+1} - \textbf{g}_k)}{\textbf{g}_k^T \textbf{g}_k}
```

- Fletcher-Reeves Approach

```math
\beta_k = \frac{\textbf{g}_{k+1}^T \textbf{g}_{k+1}}{\textbf{g}_k^T \textbf{g}_k}
```

### 2. Quasi-Newton Methods

```math
\textbf{d}_{k} = - \textbf{B}_{k} \textbf{g}_k
```

- Rank One Update (SR1)

```math
\textbf{B}_{k+1} = \textbf{B}_{k} + \frac{(\delta_k - \textbf{B}_k \gamma_k)(\delta_k - \textbf{B}_k \gamma_k)^T}{(\delta_k - \textbf{B}_k \gamma_k)^T \gamma_k}
```

- Davidon-Fletcher-Powell Formula (DFP)

```math
\textbf{B}_{k+1} = \textbf{B}_{k} + \frac{\delta_k \delta_k^T}{\delta_k^T \gamma_k} - \frac{\textbf{B}_k \gamma_k \gamma_k^T \textbf{B}_k}{\gamma_k^T \textbf{B}_k \gamma_k}
```

- Broyden-Fletcher-Goldfarb-Shanno Formula (BFGS)

```math
\textbf{B}_{k+1} = \textbf{B}_{k} + \left(1 + \frac{\gamma_k^T \textbf{B}_k \gamma_k}{\delta_k^T \gamma_k}\right) \frac{\delta_k \delta_k^T}{\delta_k^T \gamma_k} - \frac{\delta_k \gamma_k^T \textbf{B}_k + \textbf{B}_k \gamma_k \delta_k^T}{\delta_k^T \gamma_k}
```

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

- Matyas Function

```math
f(\textbf{x}) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2
```

- Rotated Hyper-Ellipsoid Function

```math
f(\textbf{x}) = \sum_{i=1}^d \sum_{j=1}^i x_j^2
```

## File Structure

- ``combined/`` - For generating all the plots of different methods, in a single figure, for each test case. No output is printed.

- ``individual/`` - For generating separate outputs and plots, for each pair of test case and method.

- ``report/`` - Summary of the key results.
