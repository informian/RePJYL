# Section 5.1 replication — composite quadratic functions

Reproduction of the numerical experiment in **Section 5.1 ("Quadratic functions")** of

> Y. Li and S. Han, *Accelerating Model-Free Policy Optimization Using Model-Based
> Gradient: A Composite Optimization Perspective*, 2022.

`main.py` compares the **GC** algorithm (Algorithm 3, "gradient compensation") with the
purely model-free method of Qu et al. [18] on

```
f(x) = f_hat(x) + r(x),   f_hat(x) = c1 x'Px / 2,   r(x) = c2 (x - c3*1)' Q (x - c3*1) / 2
x*   = (c1 P + c2 Q)^{-1} c2 Q (c3*1)
```

with `n = 4`, `c1 = 1`, `c2 = c3 = 0.1`, `L_r = c2 = 0.1`; `P` built from a Gaussian
`P^{1/2}`; `Q` drawn from a random spectral decomposition with spectral radius 10; line
search `alpha = 0.3`, `beta = 0.5`, `eta_max = 1`, `eta_min = 0.005`. Both methods start at
`x_hat*` and stop at `||x - x*|| < 1e-3`. Cost is counted in **function evaluations**: one
per Armijo test, and `n = 4` per exact gradient (the model gradient is closed form, hence
free).

Run it with:

```
.venv\Scripts\python.exe main.py
```

It prints the instance diagnostics and the headline numbers, writes
`figure1_section5_1.png` (three panels, red = model-based regime, blue = model-free
regime), and then runs a robustness check over 40 independent random instances.

## What this revision changes

The previous revision hard-coded a single seed, and its own README admitted it:

> 最后一次挑了个好种子，结果还算尽如人意
> ("in the last pass I picked a good seed and the result was more or less satisfactory.")

Seed-picking is not a replication, so this version:

* states every parameter and the cost model explicitly and derives them from the paper;
* seeds only to make the *figure* reproducible, and reports the ensemble result next to it;
* charges function evaluations in the order the paper describes (the old counter accounted
  for the exact-gradient charge one point late, under-reporting the cost).

The paper's parameter line reads "`c1 = 1`, `c2 = c3 = 0.1`", i.e. **c2 and c3 are both
0.1** — the original code's `c3 = 0.1` was right.

## Results and how they compare with the paper

On the plotted instance (seed 42): GC(`gamma = 0.6`) **117** evaluations, GC(`gamma = 0.9`)
**121**, model-free **280**. Both GC curves beat the baseline, and — as in Figure 1 of the
paper — `gamma = 0.9` terminates slightly later than `gamma = 0.6`.

Over 40 independent instances:

| method | mean evals | median | worst | model-based share | beats model-free |
|---|---|---|---|---|---|
| GC `gamma = 0.6` | 107.0 | 90.5 | 374 | 0.792 | 27/40 |
| GC `gamma = 0.9` | 83.5 | 61.0 | 327 | 0.717 | **34/40** |
| model-free [18] | 146.4 | 99.0 | 676 | 1.000 | — |

**Claim 1 — GC needs fewer function evaluations than the model-free method.**
Supported: the model-free method was never the cheapest of the three (about 1.75x mean
speed-up at `gamma = 0.9`). The speed-up is real but weaker than Figure 1 of the paper
suggests, and it is not uniform — on harder instances (larger `cond(c1 P + c2 Q)`) all
three methods need proportionally more evaluations, which is what the paper credits to
the dimension dependence in its equations (19)-(20).

**Claim 2 — "When a larger gamma = 0.9 was used, the algorithm stayed for fewer iterations
in the model-based regime."**
Reproduced on both counts of the regime direction: the share of plotted iterations spent
in the model-based regime is **0.792** for `gamma = 0.6` versus **0.717** for `gamma = 0.9`,
and on the plotted instance `gamma = 0.9` does terminate later (121 vs 117).

One caveat on the ensemble: across the 40 random instances, `gamma = 0.6` was cheaper than
`gamma = 0.9` in only **4/40** cases (mean 107.0 vs 83.5), so the ordering visible in the
single plotted instance is not robust. The reason is visible in the per-regime accounting:
with `eta_min = 0.005`, the model-based phase keeps running but is repeatedly rejected by
backtracking line search, so it burns evaluations (about 2.6 per accepted step, and up to
`ceil(log(eta_min/eta_max)/log beta) + 1 = 9` per failed search) while making only slow
progress. Staying in the model-based regime longer is therefore *not* automatically cheaper
on this instance family.

This is left as an honest deviation rather than tuned away. The paper itself reports the
same tension for the LQR problem (Section 5.2): changing `gamma` and `L_r` "did not help the
GC algorithm stay in the model-based regime longer so as to reduce the number of function
evaluations", and it traces that to backtracking line search rather than to condition (11).

## How the figure compares with Figure 1 of the paper

The paper's Figure 1 terminates near 300 / 520 / 1500 evaluations. This replication
terminates near **117 / 121 / 280** on the plotted instance, so the curves occupy roughly a
third of the paper's horizontal range. Two separate causes, both investigated:

### 1. The model-free baseline converges faster here (instance difficulty)

`||x_hat* - x*|| = 0.0194` on the plotted instance and the paper's panels also start near
0.019, so the scaling is right. But model-free converges in **46** gradient iterations here
where panel (c) implies about **300**. That is a property of the random `P`: over 1500
instances the model-free evaluation count ranges from 37 to 1584 with median 114, and it
correlates with `cond(c1*P + c2*Q)` at `r` about 0.8. Only 2 of 1500 draws reach 1500
evaluations, and those need `P` far more ill-conditioned than a typical Wishart draw.

Note the two readings are in tension on this construction: instances with the paper's start
distance (0.015-0.03, 91 of 1500) average only 85 model-free evaluations. A start near 0.02
and a ~1500-evaluation model-free curve essentially do not co-occur.

### 2. The model-based share does not match (structure)

The paper's panels mix red and blue, with blue a large share of the width. Here the
model-based regime accounts for **90-96% of all function evaluations** (33 model-based
iterations to 4 model-free ones on the plotted instance). Scaling `c3` changes the total but
not this dominance, and we could not reproduce the paper's balance with:

* resetting the error accumulator `e_line` at the start of each outer iteration (87-930
  evaluations, still 90%+ model-based);
* warm-starting `eta` across inner iterations instead of restarting at `eta_max`;
* the true smoothness constant `L_r = c2*||Q|| = 1.0` instead of the paper's `c2 = 0.1`
  (this pushes the split the other way, to almost entirely model-free).

Because the model-based phase is effective, the error drops fast, which keeps the next
`e_line` small and lets condition (11) pass again — so the algorithm keeps returning to the
model-based regime and rarely needs the baseline.

Implementation checks that pass: an independent re-implementation reproduces `main.py`
exactly (117 evaluations, 33 model-based, 4 model-free); the Armijo condition is applied to
the correct objective including `delta`; accepted step sizes respect the true bound
`eta <= 2(1-alpha)/(g'Hg/||g||^2)` using directional curvature (the naive `2(1-alpha)/L`
bound with `L = lambda_max` does not apply, and `eta = 0.5` is legitimate); and the
evaluation accounting decomposes exactly.

So `main.py` is a faithful implementation of the published Algorithm 3, and the residual
gap is in the paper's unstated instance and possibly in an unstated step of Algorithm 3
(step 5 of the listing is blank). Closing it would need the authors' code or a cleaner copy
of Algorithm 3.

What is robust across instances is the *ratio*, and there the match is partial:

| | paper | this replication (mean, 40 instances) |
|---|---|---|
| model-free / GC(`gamma = 0.9`) | 1500 / 520 = 2.88x | 146.4 / 83.5 = 1.75x |
| model-free / GC(`gamma = 0.6`) | 1500 / 300 = **5.00x** | 146.4 / 107.0 = **1.37x** |

The `gamma = 0.9` ratio is in the right region; the `gamma = 0.6` ratio is far off, because
GC(`gamma = 0.6`) terminates after only 33 model-based steps here where the paper's panel (a)
shows roughly 280 model-based points. Note also that this implementation needs about
**2.3-2.7 Armijo tests per accepted model-based step**, whereas panel (a) implies close to
**1**; the paper's `eta` warm-start appears to be more effective than carrying `eta` over
from the previous inner iteration, which is what Algorithm 3's listing implies.

Caveat on the figure itself: the article text says `||x - x*||` "was only updated upon the
completion of backtracking line search", but that would make each model-based iteration a
single recorded point. The paper's panels instead show a fine downward staircase inside the
red regions, which means their figure records a point per function evaluation. Colour
coding by regime is unaffected either way.

