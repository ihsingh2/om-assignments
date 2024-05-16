## Brief

Implementation of Projection Gradient Descent and Dual Ascent algorithm.

## Methods

### 1. Projection Gradient Method

```math
\textbf{x}_{k + 1} = P_c(\textbf{x}_{k} - t_k \nabla f(\textbf{x}_k))
```

### 2. Dual Ascent Method

```math
\textbf{x}_{k + 1} = \textbf{x}_{k} - \alpha \nabla_x \mathcal{L}(\textbf{x}_k, \lambda_k)
```

```math
\lambda_{k + 1} = \lambda_{k} + \alpha \nabla_\lambda \mathcal{L}(\textbf{x}_{k + 1}, \lambda_k) \\
```

```math
\lambda_{k + 1, i} = \max(0, \lambda_{k + 1, i})
```

## Functions

- Trid Function

```math
f(\textbf{x}) = \sum_{i=1}^d (x_i - 1)^2 - \sum_{i=2}^d x_{i-1} x_i
```

- Matyas Function

```math
f(\textbf{x}) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2
```

## Constraints

- Linear

```math
\textbf{lb} \leq \textbf{x} \leq \textbf{ub}
```

- L2

```math
\| \textbf{x} - \textbf{c} \| \leq r
```

## File Structure

- ``report/`` - Summary of the key results.

- ``src/`` - Implementation in Python.
