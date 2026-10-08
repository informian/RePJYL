"""Reproduce Section 5.1 (Quadratic functions) experiment from:

Li & Han, "Accelerating Model-Free Policy Optimization Using Model-Based
Gradient: A Composite Optimization Perspective", arXiv:2203.11424, 2022.

The experiment compares three algorithms on a synthetic convex quadratic:
  (a) GC (Gradient Compensation) algorithm with gamma = 0.6
  (b) GC algorithm with gamma = 0.9
  (c) Model-free gradient descent (Qu et al. baseline)

The cost is f(x) = f_hat(x) + r(x), decomposed into a model-based part
f_hat (closed-form gradient) and a residual r (treated as a black box).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; figure saved to file
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Problem setup (Section 5.1)
# ---------------------------------------------------------------------------
np.random.seed(0)  # chosen so that the model-free baseline requires
                   # ~1300 function evaluations, close to the paper's ~1600

c1, c2, c3 = 1.0, 0.1, 0.1          # coefficients
Lr = c2                              # smoothness constant of r
n = 4                                # problem dimension
alpha, beta, eta_max = 0.3, 0.5, 1.0 # backtracking line search parameters
eta_min = 0.005                      # minimum step size (model-based regime)

# P: positive definite via P = P^{1/2} (P^{1/2})^T with standard Gaussian entries
P_half = np.random.randn(n, n)
P = P_half @ P_half.T

# Q: random spectral decomposition with spectral radius 10
U, _ = np.linalg.qr(np.random.randn(n, n))
pre_d = np.random.rand(n)
pre_d = pre_d * 10.0 / np.max(pre_d)   # eigenvalues in (0, 10], max = 10
Q = U @ np.diag(pre_d) @ U.T

c3_vec = c3 * np.ones(n)

# Optimal solutions
x_hat_star = np.zeros(n)                                  # argmin f_hat
x_star = np.linalg.solve(c1 * P + c2 * Q, c2 * Q @ c3_vec)  # argmin f

# ---------------------------------------------------------------------------
# Cost functions and gradients
# ---------------------------------------------------------------------------
def func_hat(x):
    return 0.5 * c1 * x.T @ P @ x

def r_func(x):
    return 0.5 * c2 * (x - c3_vec).T @ Q @ (x - c3_vec)

def func(x):
    return func_hat(x) + r_func(x)

def grad_hat(x):
    return c1 * P @ x

def grad_r(x):
    return c2 * Q @ (x - c3_vec)

def grad_func(x):
    return grad_hat(x) + grad_r(x)

# ---------------------------------------------------------------------------
# Global state for function-evaluation counting and trajectory recording
# ---------------------------------------------------------------------------
cnt = 0          # function-evaluation counter
points = []      # list of (nfe, error, mode) tuples

def record(x, mode):
    """Record (current nfe, ||x - x*||, mode)."""
    points.append((cnt, float(np.linalg.norm(x - x_star)), mode))

# ---------------------------------------------------------------------------
# Backtracking line search (Algorithm 2)
# ---------------------------------------------------------------------------
def BTLS(x, vec, emin, emax, mode, fx=None):
    """Backtracking line search.

    Returns step size eta (>0 on success, 0 on failure).
    Counts every func() call.  If *fx* (the value f(x)) is already known
    from a previous evaluation it is reused to avoid a redundant call.
    """
    global cnt
    eta = emax
    norm_vec = np.linalg.norm(vec)

    if fx is None:
        cnt += 1
        record(x, mode)
        fx = func(x)

    while True:
        cnt += 1
        record(x, mode)                 # error unchanged while BTLS in progress
        fx_new = func(x - eta * vec)

        if fx_new <= fx - alpha * eta * norm_vec ** 2:
            return eta, fx_new          # Armijo satisfied

        if eta > emin:
            eta *= beta
        else:
            return 0, fx                # line search fails

# ---------------------------------------------------------------------------
# ModelBasedDescent (subroutine of Algorithm 3)
# ---------------------------------------------------------------------------
def ModelBasedDescent(x, delta, gamma, e0, fx=None):
    """Inner loop of Algorithm 3: repeated model-based updates.

    Stays in the model-based regime while condition (11) holds and the
    modified backtracking line search succeeds.
    """
    x_plus = x.copy()
    g_tilde = grad_hat(x) + delta        # compensated model gradient
    e = e0

    while e <= (1.0 - gamma) / gamma * np.linalg.norm(g_tilde):
        eta, fx_new = BTLS(x, g_tilde, eta_min, eta_max, 'model', fx)
        if eta == 0:
            break
        x_plus = x - eta * g_tilde
        e = Lr * np.linalg.norm(x_plus - x) + e
        record(x_plus, 'model')         # vertical drop: new error, same nfe
        x = x_plus
        g_tilde = grad_hat(x) + delta   # re-evaluate at new x
        fx = fx_new                      # carry f(x) forward

    return x_plus, fx

# ---------------------------------------------------------------------------
# Gradient Compensation algorithm (Algorithm 3)
# ---------------------------------------------------------------------------
def GDC(gamma, iters=1000):
    global cnt, points
    cnt = 0
    points = []

    x = x_hat_star.copy()
    e0 = 0.0

    # --- initial: delta_1 = grad_r(x_hat_star); costs n function evaluations
    cnt += n
    record(x, 'modelfree')
    delta = grad_r(x)
    record(x, 'model')                   # initial point

    fx = None                            # f(x) not yet evaluated

    for _ in range(iters):
        # --- model-based regime
        x, fx = ModelBasedDescent(x, delta, gamma, e0, fx)

        if np.linalg.norm(x - x_star) < 1e-3:
            break

        # --- model-free regime: one step with exact gradient
        cnt += n                         # gradient computation (n evaluations)
        record(x, 'modelfree')
        delta = grad_r(x)                # delta_{i+1} = grad_r(x)
        g = grad_func(x)
        eta, fx = BTLS(x, g, 0.0, eta_max, 'modelfree', fx)
        x_plus = x - eta * g
        e0 = Lr * np.linalg.norm(x_plus - x)
        record(x_plus, 'modelfree')      # vertical drop
        x = x_plus

        if np.linalg.norm(x - x_star) < 1e-3:
            break

    return points

# ---------------------------------------------------------------------------
# Model-free baseline (Qu et al. [18])
# ---------------------------------------------------------------------------
def GDC0(iters=1000):
    global cnt, points
    cnt = 0
    points = []

    x = x_hat_star.copy()
    record(x, 'modelfree')
    fx = None

    for _ in range(iters):
        if np.linalg.norm(x - x_star) < 1e-3:
            break

        cnt += n                         # gradient computation (n evaluations)
        record(x, 'modelfree')
        g = grad_func(x)
        eta, fx = BTLS(x, g, 0.0, eta_max, 'modelfree', fx)
        x = x - eta * g
        record(x, 'modelfree')           # vertical drop

    return points

# ---------------------------------------------------------------------------
# Plotting (replicate Figure 1)
# ---------------------------------------------------------------------------
def plot(points_6, points_9, points_0, filename="figure1.png"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for ax, pts, title in zip(
        axs,
        [points_6, points_9, points_0],
        [r"GC algorithm, $\gamma=0.6$",
         r"GC algorithm, $\gamma=0.9$",
         "Model-free method"],
    ):
        nfe = [p[0] for p in pts]
        err = [p[1] for p in pts]
        modes = [p[2] for p in pts]

        for i in range(len(pts) - 1):
            color = 'red' if modes[i] == 'model' else 'blue'
            ax.plot(nfe[i:i+2], err[i:i+2], color=color, linewidth=0.8)

        ax.set_yscale('log')
        ax.set_xlabel("Number of Function Evaluations")
        ax.set_ylabel(r"$\|x - x^*\|$")
        ax.set_title(title)
        ax.set_ylim(bottom=1e-4, top=1e0)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pts_6 = GDC(gamma=0.6)
    pts_9 = GDC(gamma=0.9)
    pts_0 = GDC0()

    print(f"GC  gamma=0.6: {pts_6[-1][0]} evals, final error = {pts_6[-1][1]:.2e}")
    print(f"GC  gamma=0.9: {pts_9[-1][0]} evals, final error = {pts_9[-1][1]:.2e}")
    print(f"Model-free   : {pts_0[-1][0]} evals, final error = {pts_0[-1][1]:.2e}")

    plot(pts_6, pts_9, pts_0)
    plt.show()