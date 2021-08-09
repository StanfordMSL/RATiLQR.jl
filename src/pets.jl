#///////////////////////////////////////
#// File Name: pets.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/06
#// Description: PETS algorithm
#///////////////////////////////////////

using Random
using Distributions
using LinearAlgebra

using Distributed
import Future.randjump


"""
    PETSSolver(μ_init_array::Vector{Vector{Float64}},
    Σ_init_array::Vector{Matrix{Float64}}; kwargs...)

PETS Solver initialized with `μ_init_array = [μ_0,...,μ_{N-1}]` and
`Σ_init_array = [Σ_0,...,Σ_{N-1}]`, where the initial control distribution at time
`k` is a Gaussian distribution `Distributions.MvNormal(μ_k, Σ_k)`.

# Optional Keyword Arguments
- `num_control_samples::Int64` -- number of Monte Carlo samples for the control
  trajectory. Default: `10`.
- `deterministic_dynamics::Bool` -- determinies whether to use deterministic prediction
  for the dynamics. If `true`, `num_trajectory_samples` must be 1. Default: `false`.
- `num_trajectory_samples::Int64` -- number of Monte Carlo samples for the state
  trajectory. Default: `10`.
- `num_elite::Int64` -- number of elite samples. Default: `3`.
- `iter_max::Int64` -- maximum iteration number. Default: `5`.
- `smoothing_factor::Float64` -- smoothing factor in (0, 1), used to update the
  mean and the variance of the Cross Entropy distribution for the next iteration.
  If `smoothing_factor` is `0.0`, the updated distribution is independent of the
  previous iteration. If it is `1.0`, the updated distribution is the same as the
  previous iteration. Default: `0.1`.
- `mean_carry_over::Bool` -- save `μ_array` of the last iteration and use it to
  initialize `μ_array` in the next call to `solve!`. Default: `false`.
"""
mutable struct PETSSolver # a.k.a. "PETS"
    # CE solver parameters
    num_control_samples::Int64
    deterministic_dynamics::Bool
    num_trajectory_samples::Int64
    num_elite::Int64
    iter_max::Int64
    smoothing_factor::Float64
    mean_carry_over::Bool

    # action_distributions
    μ_init_array::Vector{Vector{Float64}}
    Σ_init_array::Vector{Matrix{Float64}}
    μ_array::Vector{Vector{Float64}}
    Σ_array::Vector{Matrix{Float64}}
    N::Int64 # control sequence length > 0 (must be the same as N in FiniteHorizonGenerativeProblem)
    iter_current::Int64
end

function PETSSolver(μ_init_array::Vector{Vector{Float64}},
                    Σ_init_array::Vector{Matrix{Float64}};
                    num_control_samples=10,
                    deterministic_dynamics=false,
                    num_trajectory_samples=10,
                    num_elite=3,
                    iter_max=5,
                    smoothing_factor=0.1,
                    mean_carry_over=false)

    @assert length(μ_init_array) == length(Σ_init_array)
    if deterministic_dynamics
        @assert num_trajectory_samples == 1 "num_trajectory_samples must to be 1"
    end
    μ_array, Σ_array = copy(μ_init_array), copy(Σ_init_array);
    N = length(μ_init_array);
    iter_current = 0;

    return PETSSolver(num_control_samples, deterministic_dynamics,
                      num_trajectory_samples, num_elite,
                      iter_max, smoothing_factor, mean_carry_over,
                      copy(μ_init_array), copy(Σ_init_array),
                      μ_array, Σ_array, N, iter_current)
end;

function initialize!(solver::PETSSolver)
    solver.iter_current = 0;
    solver.μ_array = copy(solver.μ_init_array);
    solver.Σ_array = copy(solver.Σ_init_array);
end

function compute_cost_worker(solver::Union{PETSSolver, MPPISolver},
                             problem::FiniteHorizonGenerativeProblem,
                             x::Vector{Float64}, #initial state
                             u_array::Vector{Vector{Float64}},
                             rng::AbstractRNG)
     # trajectory sampling & cost function integration
     x_history_array = Vector{Vector{Vector{Float64}}}(undef, solver.num_trajectory_samples);
     sampled_cost_array = Vector{Float64}(undef, solver.num_trajectory_samples);
     for kk = 1 : solver.num_trajectory_samples # for-loop over trajectory samples
         x_history_array[kk] = Vector{Vector{Float64}}(undef, solver.N + 1);
         x_history_array[kk][1] = copy(x);
         sampled_cost_array[kk] = 0.0;
         for tt = 1 : solver.N # for-loop over time steps
             # get stage cost and perform stochastic transition
             sampled_cost_array[kk] += problem.c(tt - 1, x_history_array[kk][tt], u_array[tt]);
             x_history_array[kk][tt + 1] =
                problem.f_stochastic(x_history_array[kk][tt], u_array[tt], rng,
                                     solver.deterministic_dynamics);
         end
         # add terminal cost
         sampled_cost_array[kk] += problem.h(x_history_array[kk][end]);
     end
     # get average cost for this control sequnece u_array
     cost = mean(sampled_cost_array);
end

function compute_cost(solver::Union{PETSSolver, MPPISolver},
                      problem::FiniteHorizonGenerativeProblem,
                      x::Vector{Float64}, #initial state
                      control_sequence_array::Vector{Vector{Vector{Float64}}},
                      rng::AbstractRNG)
    @assert length(control_sequence_array) == solver.num_control_samples;
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);

    if nprocs() > 1
        proc_id_array = 2 .+ [mod(ii, nprocs() - 1) for ii = 0 : solver.num_control_samples - 1]
    else
        proc_id_array = [1 for ii = 0 : solver.num_control_samples - 1]
    end
    cost_array = Vector{Float64}(undef, solver.num_control_samples);
    rng_array = let m = rng
                    [m; accumulate(randjump, fill(big(10)^20, nprocs()-1), init=m)]
                end;
    @sync begin
        for ii = 1 : solver.num_control_samples
            @inbounds @async cost_array[ii] =
                remotecall_fetch(compute_cost_worker, proc_id_array[ii],
                                 solver, problem, x, control_sequence_array[ii],
                                 rng_array[proc_id_array[ii]])
        end
    end
    return cost_array
end

function compute_cost_serial(solver::Union{PETSSolver, MPPISolver},
                             problem::FiniteHorizonGenerativeProblem,
                             x::Vector{Float64}, #initial state
                             control_sequence_array::Vector{Vector{Vector{Float64}}},
                             rng::AbstractRNG)
    @assert length(control_sequence_array) == solver.num_control_samples;
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);

    cost_array = Vector{Float64}(undef, solver.num_control_samples);
    for ii = 1 : solver.num_control_samples # for-loop over action sequences
        # trajectory sampling & cost function integration
        x_history_array_ii = Vector{Vector{Vector{Float64}}}(undef, solver.num_trajectory_samples);
        sampled_cost_array_ii = Vector{Float64}(undef, solver.num_trajectory_samples);
        for kk = 1 : solver.num_trajectory_samples # for-loop over trajectory samples
            x_history_array_ii[kk] = Vector{Vector{Float64}}(undef, solver.N + 1);
            x_history_array_ii[kk][1] = copy(x);
            sampled_cost_array_ii[kk] = 0.0;
            for tt = 1 : solver.N # for-loop over time steps
                # get stage cost and perform stochastic transition
                sampled_cost_array_ii[kk] += problem.c(tt - 1, x_history_array_ii[kk][tt], control_sequence_array[ii][tt]);
                x_history_array_ii[kk][tt + 1] =
                    problem.f_stochastic(x_history_array_ii[kk][tt], control_sequence_array[ii][tt],
                                         rng, solver.deterministic_dynamics);
            end
            # add terminal cost
            sampled_cost_array_ii[kk] += problem.h(x_history_array_ii[kk][end]);
        end
        # get average cost for this control sequnece
        cost_array[ii] = mean(sampled_cost_array_ii);
    end
    return cost_array
end

function get_elite_samples(solver::PETSSolver,
                           control_sequence_array::Vector{Vector{Vector{Float64}}},
                           cost_array::Vector{Float64})
    @assert length(cost_array) == solver.num_control_samples;
    @assert length(cost_array) == length(control_sequence_array);
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);

    control_sequence_cost_pair_array = collect(zip(control_sequence_array, cost_array));
    control_sequence_cost_pair_sorted_array = sort(control_sequence_cost_pair_array, by=x->x[2]);
    control_sequence_elite_array = [pair[1] for pair in control_sequence_cost_pair_sorted_array[1 : solver.num_elite]];

    return control_sequence_elite_array
end

function compute_new_distribution(solver::PETSSolver,
                                  control_sequence_elite_array::Vector{Vector{Vector{Float64}}})
    @assert length(control_sequence_elite_array) == solver.num_elite
    @assert all([length(sequence) == solver.N for sequence in control_sequence_elite_array]);

    μ_new_array = similar(solver.μ_array);
    Σ_new_array = similar(solver.Σ_array);

    mean_array, cov_array = similar(μ_new_array), similar(Σ_new_array);

    for tt = 1 : solver.N # for-loop over time steps
        mean_array[tt] = mean([elite[tt] for elite in control_sequence_elite_array]);
        cov_array[tt] = Diagonal(var([elite[tt] for elite in control_sequence_elite_array]));

        μ_new_array[tt] = (1.0 - solver.smoothing_factor).*mean_array[tt] + solver.smoothing_factor.*solver.μ_array[tt];
        Σ_new_array[tt] = (1.0 - solver.smoothing_factor).*cov_array[tt] + solver.smoothing_factor.*solver.Σ_array[tt];
    end
    return μ_new_array, Σ_new_array
end

function step!(solver::PETSSolver,
               problem::FiniteHorizonGenerativeProblem,
               x::Vector{Float64},
               rng::AbstractRNG,
               verbose=true, serial=false)
    solver.iter_current += 1;
    if verbose
        println("**CE iteration $(solver.iter_current)")
    end
    control_sequence_array = Vector{Vector{Vector{Float64}}}(undef, solver.num_control_samples);
    cost_array = Vector{Float64}(undef, solver.num_control_samples);
    # action sequence sampling
    if verbose
        println("****Drawing $(solver.num_control_samples) samples of control sequences of length $(solver.N)")
    end
    for ii = 1 : solver.num_control_samples # for-loop over action sequences
        # sample an action sequence from current distribution;
        control_sequence_array[ii] = Vector{Vector{Float64}}(undef, solver.N)
        for tt = 1 : solver.N # for-loop over time steps
            d = MvNormal(solver.μ_array[tt], solver.Σ_array[tt]);
            u = rand(rng, d);
            control_sequence_array[ii][tt] = u;
        end
    end

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

    # compute elite samples
    if verbose
        println("****Choosing $(solver.num_elite) elite samples")
    end
    control_sequence_elite_array = get_elite_samples(solver, control_sequence_array, cost_array);

    # update CEM distribution with smoothing
    if verbose
        println("****Updating control sequence distribution")
    end
    μ_new_array, Σ_new_array = compute_new_distribution(solver, control_sequence_elite_array);
    solver.μ_array = μ_new_array;
    solver.Σ_array = Σ_new_array;
end;

"""
    solve!(solver::PETSSolver,
    problem::FiniteHorizonGenerativeProblem, x_0::Vector{Float64},
    rng::AbstractRNG; verbose=true, serial=true)

Given `problem` and `solver` (i.e. a PETS Solver), solve stochastic optimal
control with current state `x_0`.

# Return Values (Ordered)
- `μ_array::Vector{Vector{Float64}}` -- array of means `[μ_0,...,μ_{N-1}]` for
  the final Cross Entropy distribution for the control schedule.
- `Σ_array::Vector{Matrix{Float64}}` -- array of covariance matrices `[Σ_0,...,Σ_{N-1}]`
  for the final Cross Entropy distribution for the control schedule.

# Notes
- Returns an open-loop control policy.
- If `serial` is `true`, Monte Carlo sampling of the Cross Entropy method is serialized
  on a single process. If `false` it is distributed on all the available worker processes.
  We recommend to leave this to `true` as distributed processing can be slower for
  this algorithm.
"""
function solve!(solver::PETSSolver,
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
    return copy(solver.μ_array), copy(solver.Σ_array)
end
