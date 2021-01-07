#///////////////////////////////////////
#// File Name: optimal_control_problems.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/10/28
#// Description: Optimal control problem definitions.
#///////////////////////////////////////
"""
    OptimalControlProblem

Abstract base type for an Optimal Control Problem.
"""
abstract type OptimalControlProblem end

# Optimal Control Problem for Risk-Sensitive iLEQG (or Risk-Neutral iLQG)
"""
    FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N) <: OptimalControlProblem

A finite horizon, stochastic optimal control problem where the dynamics function is subject to additive Gaussian noise w ~ N(0, W).

# Arguments
- `f(x, u, f_returns_jacobian=false)` -- deterministic dynamics function
  - `x` is a state vector and `u` is a control input vector.
  - The third positional argument `f_returns_jacobian` determines whether the user computes
    and returns the Jacobians, and should default to `false`. If `true`, the return value
    must be augmented with matrices `A` and `B`, where `A = dx_next/dx` and `B = dx_next/du`.
    Otherwise the return value is the (noiseless) next state `x_next`.
- `c(k, x, u)` -- stage cost function
  - `k::Int >= 0` is a time index where `k == 0` is the initial time.
  -  We assume that `c` is non-negative.
- `h(x)` -- terminal cost function
  -  We assume that `h` is non-negative.
- `W(k)` -- covariance matrix function
  - Returns a symmetric positive semidefinite matrix that represents the covariance matrix for
    additive Gaussian noise w ~ N(0, W).
  - `k::Int >= 0` is a time index where `k == 0` is the initial time.
- `N::Int64` -- final time index
  - Note that `0` is the initial time index.

# Notes
- Functions `f`, `c`, and `h` should be written generically enough to accept
  the state `x` and the input `u` of type `Vector{<:Real}`. This is to ensure that
  ForwardDiff can compute Jacobians and Hessians for iLQG/iLEQG.

# Example
```julia
import LinearAlgebra;

function f(x, u, f_returns_jacobian=false)
    x_next = x + u; # 2D single integrator dynamics
    if f_returns_jacobian
        A = Matrix(1.0LinearAlgebra.I, 2, 2); # dx_next/dx
        B = Matrix(1.0LinearAlgebra.I, 2, 2); # dx_next/du
        return x_next, A, B;
    else
        return x_next;
    end
end

c(k, x, u) = k/2*x'*x + k/2*u'*u  # time-dependent quadratic stage cost
N = 10;
h(x) = N/2*x'*x; # quadratic terminal cost
W(k) = Matrix(0.1LinearAlgebra.I, 2, 2);

problem = FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N);
```
"""
mutable struct FiniteHorizonRiskSensitiveOptimalControlProblem <: OptimalControlProblem
    f::Function # Dynamics function x_{t+1} = f(x_t, u_t)
    c::Function # Stage cost function c(k, x, u) where the time index k == 0 is the initial timestep
    h::Function # Terminal cost function h(x)
    W::Function # Additive process noise covariance matrix W(k) where the time index k == 0 is the initial timestep
    N::Int64 # horizon length > 0
end

# Optimal Control Problem for Sampling-based MPC methods
"""
    FiniteHorizonGenerativeOptimalControlProblem(f_stochastic, c, h, N) <: OptimalControlProblem

A finite horizon, stochastic optimal control problem where the dynamics function is stochastic and generative.

# Arguments
- `f_stochastic(x, u, rng, use_true_model=false)` -- stochastic dynamics function
  - `x` is a state vector and `u` is a control input vector.
  - The third positional argument `rng` is a random seed.
  - The fourth positional argument `use_true_model` determines whether a solver has access
    to the true stochastic dynamics and defaults to `false`.
  - The return value is the (noisy) next state `x_next`.

- `c(k, x, u)` -- stage cost function
  - `k::Int >= 0` is a time index where `k == 0` is the initial time.
  -  We assume that `c` is non-negative.
- `h(x)` -- terminal cost function
  -  We assume that `h` is non-negative.
- `N::Int64` -- final time index
  - Note that `0` is the initial time index.

# Example
```julia
import Distributions;
import LinearAlgebra;

function f_stochastic(x, u, rng, use_true_model=false)
    Σ_1 = Matrix(0.5LinearAlgebra.I, 2, 2);

    if use_true_model  # accurate GMM model
        Σ_2 = Matrix(1.0LinearAlgebra.I, 2, 2)
        d = Distributions.MixtureModel([Distributions.MvNormal(zeros(2), Σ_1),
                                        Distributions.MvNormal(ones(2), Σ_2)],
                                        [0.5, 0.5]);
    else  # inaccurate Gaussian model
        d = Distributions.MvNormal(zeros(2), Σ_1);
    end

    x_next = x + u + Distributions.rand(rng, d); # 2D single integrator dynamics
    return x_next;
end

c(k, x, u) = k/2*x'*x + k/2*u'*u  # time-dependent quadratic stage cost
N = 10;
h(x) = N/2*x'*x; # quadratic terminal cost
W(k) = Matrix(0.1LinearAlgebra.I, 2, 2);

problem = FiniteHorizonGenerativeOptimalControlProblem(f_stochastic, c, h, N);
```
"""
mutable struct FiniteHorizonGenerativeOptimalControlProblem <: OptimalControlProblem
    f_stochastic::Function # Stochastic dynamics function x_{t+1} = f_stochastic(x_t, u_t, rng, use_true_model=false)
    c::Function # Stage cost function c(k, x, u) where the time index k == 0 is the initial timestep
    h::Function # Terminal cost function h(x)
    N::Int64 # horizon length > 0
end
