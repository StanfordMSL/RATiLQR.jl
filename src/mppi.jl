#///////////////////////////////////////
#// File Name: mppi.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/08/08
#// Description: MPPI (a.k.a. PDDM) algorithm
#///////////////////////////////////////

using Random
using Distributions
using LinearAlgebra

import StatsFuns: softmax

"""
    MPPISolver(μ_init_array::Vector{Vector{Float64}},
               Σ::Matrix{Float64}; kwargs...)

MPPI Solver initialized with `μ_init_array = [μ_0,...,μ_{N-1}]` and
`Σ`, where the initial control distribution at time
`k` is a Gaussian distribution `Distributions.MvNormal(μ_k, Σ_k)`.

# Optional Keyword Arguments
- `num_control_samples::Int64` -- number of Monte Carlo samples for the control
  trajectory. Default: `10`.
- `deterministic_dynamics::Bool` -- determinies whether to use deterministic prediction
  for the dynamics. If `true`, `num_trajectory_samples` must be 1. Default: `false`.
- `num_trajectory_samples::Int64` -- number of Monte Carlo samples for the state
  trajectory. Default: `10`.
- `weighting_factor::Float64` -- reward-weighting factor for mean updates. Default: `1.0`.
- `iter_max::Int64` -- maximum iteration number. Default: `5`.
- `filtering_coefficient::Float64` -- filtering coefficient in (0, 1), used to
  correlate successive control updates. If `filtering_coefficient` is `0.0`, the
  control noise at the current time step is the same as the control noise at the
  previous time step. If it is `1.0`, the control noise is independent across
  time steps. Default: `0.5`.
- `mean_carry_over::Bool` -- save `μ_array` of the last iteration and use it to
  initialize `μ_array` in the next call to `solve!`. Default: `false`.
"""
mutable struct MPPISolver
    num_control_samples::Int64
    deterministic_dynamics::Bool
    num_trajectory_samples::Int64
    weighting_factor::Float64
    iter_max::Int64
    filtering_coefficient::Float64
    mean_carry_over::Bool
    Σ::Matrix{Float64}

    # action_distributions
    μ_init_array::Vector{Vector{Float64}}
    μ_array::Vector{Vector{Float64}}
    N::Int64 # control sequence length > 0 (must be the same as N in FiniteHorizonGenerativeProblem)
    iter_current::Int64
end

function MPPISolver(μ_init_array::Vector{Vector{Float64}},
                    Σ::Matrix{Float64};
                    num_control_samples=10,
                    deterministic_dynamics=false,
                    num_trajectory_samples=10,
                    weighting_factor=1.0,
                    iter_max=5,
                    filtering_coefficient=0.5,
                    mean_carry_over=false)

    if deterministic_dynamics
        @assert num_trajectory_samples == 1 "num_trajectory_samples must to be 1"
    end
    μ_array = copy(μ_init_array);
    N = length(μ_array);
    iter_current = 0;

    return MPPISolver(num_control_samples, deterministic_dynamics,
                      num_trajectory_samples, weighting_factor,
                      iter_max, filtering_coefficient, mean_carry_over,
                      copy(Σ), copy(μ_init_array), μ_array, N, iter_current);
end;

function initialize!(solver::MPPISolver)
    solver.iter_current = 0;
    solver.μ_array = copy(solver.μ_init_array);
end

function update_mean!(solver::MPPISolver,
                      control_sequence_array::Vector{Vector{Vector{Float64}}},
                      cost_array::Vector{Float64})
    @assert length(cost_array) == solver.num_control_samples;
    @assert length(cost_array) == length(control_sequence_array);
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);

    weight_array = softmax(-solver.weighting_factor .* cost_array);
    solver.μ_array = sum(weight_array .* control_sequence_array);
end

function step!(solver::MPPISolver,
               problem::FiniteHorizonGenerativeProblem,
               x::Vector{Float64},
               rng::AbstractRNG,
               verbose=true, serial=false)
    solver.iter_current += 1;
    if verbose
        println("**MPPI iteration $(solver.iter_current)")
    end
    control_sequence_array = Vector{Vector{Vector{Float64}}}(undef, solver.num_control_samples);
    cost_array = Vector{Float64}(undef, solver.num_control_samples);
    # action sequence sampling
    if verbose
        println("****Drawing $(solver.num_control_samples) samples of control sequences of length $(solver.N)");
    end
    β = solver.filtering_coefficient;
    for ii = 1 : solver.num_control_samples # for-loop over action sequences
        # sample an action sequence from current distribution;
        control_sequence_array[ii] = Vector{Vector{Float64}}(undef, solver.N);
        noise_sequence_array = Vector{Vector{Float64}}(undef, solver.N);
        for tt = 1 : solver.N # for-loop over time steps
            d = MvNormal(zeros(size(solver.Σ, 1)), solver.Σ);
            u = rand(rng, d);
            if tt == 1
                noise_sequence_array[tt] = β*u;
            else
                noise_sequence_array[tt] = β*u + (1 - β)*noise_sequence_array[tt - 1];
            end
        end
        control_sequence_array[ii] = noise_sequence_array .+ solver.μ_array;
    end;

    # cost computation
    if verbose
        println("****Evaluating costs of sampled control sequences")
    end
    if !serial
        # note that the resulting cost_array is different from the serial version, due to randjump.
        cost_array = compute_cost(solver, problem, x, control_sequence_array, rng);
    else
        cost_array = compute_cost_serial(solver, problem, x, control_sequence_array, rng);
    end
    if verbose
        println(cost_array)
    end

    # update mean
    if verbose
        println("****Updating control sequence mean")
    end
    update_mean!(solver, control_sequence_array, cost_array);
end;

"""
    solve!(solver::MPPISolver,
           problem::FiniteHorizonGenerativeProblem, x_0::Vector{Float64},
           rng::AbstractRNG; verbose=true, serial=true)

Given `problem` and `solver` (i.e. an MPPI Solver), solve stochastic optimal
control with current state `x_0`.

# Return Values (Ordered)
- `μ_array::Vector{Vector{Float64}}` -- array of means `[μ_0,...,μ_{N-1}]` for
  the final distribution for the control schedule.

# Notes
- Returns an open-loop control policy.
- If `serial` is `true`, Monte Carlo sampling of the Cross Entropy method is serialized
  on a single process. If `false` it is distributed on all the available worker processes.
  We recommend to leave this to `true` as distributed processing can be slower for
  this algorithm.
"""
function solve!(solver::MPPISolver,
                problem::FiniteHorizonGenerativeProblem,
                x_0::Vector{Float64},
                rng::AbstractRNG;
                verbose=true, serial=true)
    #setting serial=true by default as it turned out this is much faster than serial=false
    initialize!(solver);
    if solver.deterministic_dynamics
        @assert solver.num_trajectory_samples == 1 "num_trajectory_samples must to be 1"
    end
    while solver.iter_current < solver.iter_max
        step!(solver, problem, x_0, rng, verbose, serial)
    end
    if solver.mean_carry_over
        # time indices are shifted, assuming that the algorithm runs in a receding-horizon fashion.
        solver.μ_init_array[1:end-1] = copy(solver.μ_array[2:end])
    end
    return copy(solver.μ_array)
end;
