Optimal Control Problems
========================

Basics
------

Suppose that we are given a discrete-time stochastic dynamics model of the form
```math
x_{k+1} = f(x_k, u_k, w_k),
```
where $x_k \in \mathbb{R}^n$ is the state, $u_k \in \mathbb{R}^m$ is the control input, and 
$w_k \in \mathbb{R}^r$ is the stochastic noise variable drawn from some probability distribution
on $\mathbb{R}^r$.

Subject to the dynamics constraint, we are interested in minimizing an objective function $J$ over 
some finite horizon $N$:
```math
J(x_{0:N}, u_{0:N-1}) \triangleq \sum_{k=0}^{N - 1} c(k, x_k, u_k) + h(x_N),
```
where $c(k, x_k, u_k) \geq 0$ is the stage cost at time $k$ and $h(x_N) \geq 0$ is the terminal cost.

Although not explicitly written above, the actual value of $J$ is dependent on the history of stochastic 
noise $w_{0:N-1}$. This means that the objective $J$ itself is a random variable whose value cannot 
be determined without actually observing the outcome. Therefore, Stochastic Opimal Control seeks to minimize a 
statistic associated with the objective, such as the expectation $\mathbb{E}[J]$.

Supported Problem Types
-----------------------

With RATiLQR.jl, you can formulate various types of stochastic optimal control problems with different objective 
statistics and dynamics models.

### 1. Standard Problem with Gaussian Noise

- Objective: $\mathbb{E}[J]$
- Dynamics: $f(x_k, u_k, w_k) = f(x_k, u_k) + w_k, ~ w_k \sim \mathcal{N}(0, W_k)$
- Definition: [`FiniteHorizonAdditiveGaussianProblem`](@ref)
- Solvers: [`ILEQGSolver`](@ref)

### 2. Standard Problem with Arbitrary Noise

- Objective: $\mathbb{E}[J]$
- Dynamics: $f(x_k, u_k, w_k)$ where $w_k$ has an arbitrary distribution.
- Definition: [`FiniteHorizonGenerativeProblem`](@ref)
- Solvers: [`PETSSolver`](@ref)

### 3. Risk-Sensitive Problem with Gaussian Noise

- Objective: $\frac{1}{\theta}\log\left(\mathbb{E}[\exp(\theta J)]\right)$ where $\theta > 0$
  denotes the risk-sensitivity parameter and is user-specified.
- Dynamics: $f(x_k, u_k, w_k) = f(x_k, u_k) + w_k, ~ w_k \sim \mathcal{N}(0, W_k)$
- Definition: [`FiniteHorizonAdditiveGaussianProblem`](@ref)
- Solvers: [`ILEQGSolver`](@ref)

### 4. Distributionally-Robust Problem with Arbitrary Noise but with Gaussian Noise Model

- Objective: $\max_{p \in \Xi} \mathbb{E}_p [J]$ where $p$ denotes the true, potentially unknown 
  distribution over the noise variables $w_{0:N-1}$ and $\Xi$ is the ambiguity set that encodes how
  uncertain we are about the true distribution. Note that $p$ may not be Gaussian.
  In our formulation, $\Xi$ is defined by a KL divergence 
  bound between the true distribution $p(w_{0:N-1})$ and the Gaussian model 
  $q(w_{0:N-1}) \triangleq \prod_{k = 0}^{N - 1} \mathcal{N}(0, W_k)$
  as follows:
  ```math
  \Xi \triangleq \{p: \mathbb{D}_\mathrm{KL}(p \Vert q) \leq d\},
  ```
  where $d > 0$ is a user-specified constant that defines an upper bound.
- Dynamics: $f(x_k, u_k, w_k) = f(x_k, u_k) + w_k$ where $w_{0:N-1}$ can have an arbitrary distribution
  as long as it is within the ambiguity set defined by the Gaussian model.
- Definition: [`FiniteHorizonAdditiveGaussianProblem`](@ref)
- Solvers: [`RATiLQRSolver`](@ref)


Problem Definition APIs
-----------------------

```@docs
OptimalControlProblem
FiniteHorizonAdditiveGaussianProblem
FiniteHorizonGenerativeProblem
```

