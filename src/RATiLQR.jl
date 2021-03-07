#///////////////////////////////////////
#// File Name: RATiLQR.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/10/28
#// Description: Julia package for RAT iLQR and baseline methods
#///////////////////////////////////////

module RATiLQR

using Distributions
using Distributed
import Future.randjump
using ForwardDiff
using LinearAlgebra
using Printf
using Random
using Statistics

# Optimal Control Problems
export
    OptimalControlProblem,
    FiniteHorizonAdditiveGaussianProblem,
    FiniteHorizonGenerativeProblem
include("optimal_control_problems.jl")

# iterative Linear-Exponential-Quadratic-Gaussian
export
    simulate_dynamics,
    integrate_cost,
    ILEQGSolver,
    initialize!,
    ApproximationResult,
    approximate_model,
    DynamicProgrammingResult,
    solve_approximate_dp!,
    solve_approximate_dp,
    increase_μ_and_Δ!,
    decrease_μ_and_Δ!,
    line_search!,
    step!,
    solve!
include("ileqg.jl")

# Risk Auto-Tuning iterative LQR (i.e. Cross Entropy + iLEQG)
export
    RATiLQRSolver,
    compute_value_worker,
    compute_cost,
    compute_cost_serial,
    get_positive_samples,
    step!,
    solve!
include("rat_ilqr.jl")

# PETS (Cross Entropy Method with Stochastic Sampling of Dynamics)
export
    PETSSolver,
    initialize!,
    compute_cost_worker,
    compute_cost,
    compute_cost_serial,
    get_elite_samples,
    compute_new_distribution,
    step!,
    solve!
include("pets.jl")

end # module
