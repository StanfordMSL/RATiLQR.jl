Getting Started
===============

Installation
------------

This package is developed and tested on Julia v1.5.2, but the code should be combatible with any Julia v1.x. To install the package, run the following commands in the Julia REPL.

```julia
import Pkg
Pkg.add(url="https://github.com/StanfordMSL/RATiLQR.jl.git")
```
Example
-------

In this example, we are going to optimize an [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) objective with 
imperfect knowlege of the probability distribution governing the stochastic state transitions of the system. Specifically, we seek to minimize
the following objective:

```math
\max_{p \in \Xi} \mathbb{E}_p \left[\frac{1}{2} \sum_{k=0}^{N - 1} \left(x_k^{\mathrm{T}}Qx_k + u_k^{\mathrm{T}}R u_k\right) +
\frac{1}{2}x_N^{\mathrm{T}}Rx_N\right]
```
```math
\text{subject to } x_{k+1} = x_k + 0.1 \times u_k + w_k \in \mathbb{R}^2, ~~ (w_0,...,w_{N-1}) \sim p(w_{0:N-1}),
```
where the ambiguity set $\Xi \triangleq \{p: \mathbb{D}_\mathrm{KL}(p \Vert q) \leq d\}$ for the true but unknown distribution $p$
is defined by a reference (Gaussian) distribution $q(w_{0:N-1}) \triangleq \prod_{k = 0}^{N - 1} \mathcal{N}(0, W)$ and an upper-bound
$d \geq 0$ on the KL divergence between $p$ and $q$. This problem is an instance of a type of problems called Distributionally 
Robust Optimal Contol, and this particular problem can be solved by the RATiLQR.jl package.

With the RAT iLQR Solver, we can find a locally optimal, affine state-feedback 
policy of the form:
```math
u_k = \pi_k(x) \triangleq L_k(x - x_k) + l_k 
```
to approximately minimize the objective.

The following Julia code defines the problem using our Problem Definition API and performs the local optimization:
```julia
# We are going to use 4 worker processes to distribute the Cross Entropy sampling.
# If you are not doing distributed processing you don't need @everywhere macro.
using Distributed
addprocs(4);

@everywhere using LinearAlgebra, Random
@everywhere using RATiLQR


#=============== PROBLEM DEFINITION ===============#
# generic 2D single integrator dynamics function with parameter dt.
@everywhere function f(x, u, dt, f_returns_jacobian::Bool)
    @assert length(x) == length(u) == 2;
    x_next = x + dt*u;
    if f_returns_jacobian
        A = Matrix(1.0LinearAlgebra.I, 2, 2); # dx_next/dx
        B = Matrix(dt*LinearAlgebra.I, 2, 2); # dx_next/du
        return x_next, A, B;
    else
        return x_next;
    end
end

# generic quadratic cost function with parameters Q and R.
@everywhere function c(x, u, Q, R)
    return 1/2*x'*Q*x + 1/2*u'*R*u;
end

# define your own model.
@everywhere mutable struct SingleIntegratorLQRModel
    # Model parameters
    dt::Float64        # discrete time inverval [s].
    Q::Matrix{Float64} # 2-by-2 state cost weight matrix.
    R::Matrix{Float64} # 2-by-2 control cost weight matrix.

    # Member functions needed to define the optimal control problem
    f::Union{Nothing, Function}  # Noiseless dynamics function x_{k+1} = f(x_k, u_k)
    c::Union{Nothing, Function}  # stage cost function c(k, x_k, u_k)
    h::Union{Nothing, Function}  # terminal cost function c(x_N)
    W::Union{Nothing, Function}  # Gaussian covariance matrix function W(k)

    function SingleIntegratorLQRModel(dt=0.1, Q=Matrix(1.0I, 2, 2), R=Matrix(0.01I, 2, 2))
        this = new(dt, Q, R, nothing, nothing, nothing, nothing); # leave member functions nothing.

        # Define member functions below so their formats are compatible with the OptimalControlProblem type.
        this.f = (x, u, f_returns_jacobian=false) -> f(x, u, this.dt, f_returns_jacobian);
        this.c = (k, x, u) -> c(x, u, this.Q, this.R); # it is ok if c is not dependent on time k.
        this.h = x -> c(x, zeros(2), this.Q, this.R);
        this.W = k -> Matrix(0.1*this.dt*I, 2, 2); # it is ok if W is not dependent on time k.

        return this
    end
end


#=============== OPTIMIZATION ===============#
# Instantiate a model.
model = SingleIntegratorLQRModel();

# Define an OptimalControlProblem.
N = 10; # final time index. Note that the initial time is 0.
problem = FiniteHorizonRiskSensitiveOptimalControlProblem(model.f, model.c, model.h, model.W, N)

# Instantiate a RAT iLQR solver.
solver = CrossEntropyBilevelOptimizationSolver();

# Solve problem to obtain a feedback control policy: π_k(x) = L_k(x - x_k) + l_k.
rng = MersenneTwister(12345); # pseudo random number generator for the Cross Entropy method
x_0 = [5.0, 5.0]; # initial state
u_array = [zeros(2) for ii = 1 : N]; # initial guess for the nominal control schedule
kl_bound = 0.1; # kl divergence bound between the true unknown noise distribution and the Gaussian model.

θ_opt, x_array, l_array, L_array, value, θ_min, θ_max =
    solve!(solver, problem, x_0, u_array, rng, kl_bound=kl_bound);
```
