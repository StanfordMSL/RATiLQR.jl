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
    FiniteHorizonRiskSensitiveOptimalControlProblem,
    FiniteHorizonGenerativeOptimalControlProblem
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
    regularize!,
    line_search!,
    step!,
    solve!
include("ileqg.jl")

# Cross Entropy Bilevel Optimization (i.e. RATiLQR)
export
    CrossEntropyBilevelOptimizationSolver,
    compute_value_worker,
    compute_cost,
    compute_cost_serial,
    get_positive_samples,
    step!,
    solve!
include("cross_entropy_bilevel_optimization.jl")

# Nelder-Mead Simplex Bilevel Optimization
export
    NelderMeadBilevelOptimizationSolver,
    compute_cost_worker,
    initialize!,
    step!,
    solve!
include("nelder_mead_bilevel_optimization.jl")

# PETS
export
    CrossEntropyDirectOptimizationSolver,
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
