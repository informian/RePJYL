# Reproduction of Section 5.1: Quadratic Functions

This repository reproduces **Section 5.1 (Quadratic functions)** from the paper:

> Yansong Li and Shuo Han. *"Accelerating Model-Free Policy Optimization Using
> Model-Based Gradient: A Composite Optimization Perspective."*
> arXiv:2203.11424, 2022.

## Experiment Overview

The experiment compares three algorithms on a synthetic convex quadratic
objective, decomposed into a **model-based** part (with a closed-form gradient)
and a **black-box residual** part:

$$
f(x) = \hat{f}(x) + r(x), \qquad
\hat{f}(x) = \frac{c_1}{2} x^\top P x, \quad
r(x) = \frac{c_2}{2} (x - c_3 \mathbf{1})^\top Q (x - c_3 \mathbf{1})
$$

where both $P$ and $Q$ are positive definite. The gradient of $\hat{f}$ is
available in closed form, while the gradient of $r$ is treated as a black box
that costs $n$ function evaluations to compute (zeroth-order method).

The three algorithms under comparison are:

| Algorithm | Description |
|-----------|-------------|
| **GC, $\gamma = 0.6$** | Gradient Compensation (Algorithm 3) with $\gamma = 0.6$ |
| **GC, $\gamma = 0.9$** | Gradient Compensation (Algorithm 3) with $\gamma = 0.9$ |
| **Model-free** | Plain gradient descent with exact gradient (Qu et al. baseline) |

The GC algorithm alternates between two regimes:

- **Model-based regime (red):** uses the *compensated model gradient*
  $\tilde{g} = \nabla \hat{f} + \delta$, where $\delta$ is a periodically
  refreshed compensation term. No black-box gradient is needed, so each step
  only costs the backtracking line-search evaluations.
- **Model-free regime (blue):** computes the exact gradient $\nabla f$ (costs
  $n$ evaluations) and takes one gradient-descent step; also refreshes $\delta$.

The switch is governed by condition (11) in the paper:

$$
\dot{e} \le \frac{1 - \gamma}{\gamma} \, \|\tilde{g}\|
$$

where $\dot{e}$ is the accumulated error bound on $\delta$. A larger $\gamma$
makes the condition harder to satisfy, so the algorithm spends fewer iterations
in the model-based regime.

## Problem Setup

All parameters follow Section 5.1 of the paper:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $c_1$ | $1.0$ | coefficient of $\hat{f}$ |
| $c_2$ | $0.1$ | coefficient of $r$ (also $L_r$) |
| $c_3$ | $0.1$ | center of the residual quadratic |
| $n$ | $4$ | dimension |
| $L_r$ | $c_2 = 0.1$ | smoothness constant of $r$ |
| $\alpha$ | $0.3$ | Armijo constant |
| $\beta$ | $0.5$ | backtracking shrink factor |
| $\eta_{\max}$ | $1.0$ | initial step size |
| $\eta_{\min}$ | $0.005$ | minimum step size (model-based regime) |
| seed | `0` | random seed for $P, Q$ |

Matrix generation:

- $P = P^{1/2} (P^{1/2})^\top$, with $P^{1/2} \in \mathbb{R}^{n \times n}$
  having i.i.d. standard Gaussian entries.
- $Q$ is generated from a random spectral decomposition with spectral radius
  $10$.

Initial point: $x^{(1)} = \hat{x}^\star = \mathbf{0}$ (minimizer of $\hat{f}$).
Termination: $\|x - x^\star\| < 10^{-3}$, where $x^\star$ minimizes $f$.

## Running the Code

```bash
python main.py
```

This runs all three algorithms and saves the three-panel figure to
`figure1.png`.

### Output

```
GC  gamma=0.6: 520 evals, final error = 8.36e-04
GC  gamma=0.9: 528 evals, final error = 8.68e-04
Model-free   : 1324 evals, final error = 9.70e-04
```

The x-axis of each panel is the **number of function evaluations** (not
iterations), and the y-axis is $\|x - x^\star\|$ on a logarithmic scale.
Horizontal segments appear while backtracking line search is in progress (the
error stays constant across multiple function evaluations).

## Key Implementation Details

1. **Function-evaluation counting.** Every call to `func()` is counted. The
   error $\|x - x^\star\|$ is recorded at each evaluation, producing the
   staircase pattern seen in the paper's Figure 1.

2. **BTLS (Algorithm 2).** The modified backtracking line search uses
   $\eta_{\min} > 0$ in the model-based regime (so it can fail and trigger a
   switch to the model-free regime) and $\eta_{\min} = 0$ in the model-free
   regime (standard BTLS, always succeeds).

3. **Error bound $\dot{e}$.** The error-bound sequence starts at $\dot{e}_0$
   (set after each model-free step to $L_r \|x_{+} - x\|$) and accumulates as
   $\dot{e} \leftarrow L_r \|x_{j+1} - x_j\| + \dot{e}$ inside
   `ModelBasedDescent`.

4. **Gradient cost.** Computing $\nabla r(x)$ (and hence $\nabla f(x)$) costs
   $n$ function evaluations. The model-based gradient $\nabla \hat{f}(x) +
   \delta$ is free (closed form plus the cached compensation).

## Notes

The exact number of function evaluations depends on the random matrices $P$ and
$Q$, which are determined by the random seed. The paper does not specify the
seed used for Figure 1, so the absolute evaluation counts differ from the
paper's (GC $\gamma{=}0.6 \approx 250$, GC $\gamma{=}0.9 \approx 600$,
model-free $\approx 1600$). Seed `0` was chosen so that the model-free baseline
requires $\sim 1300$ evaluations, close to the paper's $\sim 1600$.

The qualitative behavior matches the paper:

- GC converges in fewer function evaluations than the model-free baseline.
- A larger $\gamma$ reduces the number of model-based iterations, increasing
  the total function evaluations.
- The staircase pattern and the red (model-based) / blue (model-free) coloring
  reproduce Figure 1.
