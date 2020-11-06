#///////////////////////////////////////
#// File Name: optimal_control_problems.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/10/28
#// Description: Optimal control problem definitions.
#///////////////////////////////////////

abstract type OptimalControlProblem end

# Optimal Control Problem for Risk-Sensitive iLEQG (or Risk-Neutral iLQG)
mutable struct FiniteHorizonRiskSensitiveOptimalControlProblem <: OptimalControlProblem
    f::Function # Dynamics function x_{t+1} = f(x_t, u_t)
    c::Function # Stage cost function c(k, x, u) where the time index k == 0 is the initial timestep
    h::Function # Terminal cost function h(x)
    W::Function # Additive process noise covariance matrix W(k) where the time index k == 0 is the initial timestep
    N::Int64 # horizon length > 0
end

# Optimal Control Problem for Sampling-based MPC methods
mutable struct FiniteHorizonGenerativeOptimalControlProblem <: OptimalControlProblem
    f_stochastic::Function # Stochastic dynamics function x_{t+1} = f_stochastic(x_t, u_t, rng, use_true_model=false)
    c::Function # Stage cost function c(k, x, u) where the time index k == 0 is the initial timestep
    h::Function # Terminal cost function h(x)
    N::Int64 # horizon length > 0
end
