#///////////////////////////////////////
#// File Name: rat_cem.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/03/08
#// Description: Risk Auto-Tuning Cross Entropy Method (RAT CEM)
#///////////////////////////////////////

using Random
using DataStructures
using Distributions
using LinearAlgebra

using Distributed
import Future.randjump
import StatsFuns.logsumexp


"""
    RATCEMSolver(μ_init_array::Vector{Vector{Float64}},
    Σ_init_array::Vector{Matrix{Float64}}; kwargs...)

RAT CEM Solver initialized with `μ_init_array = [μ_0,...,μ_{N-1}]` and
`Σ_init_array = [Σ_0,...,Σ_{N-1}]`, where the initial control distribution at time
`k` is a Gaussian distribution `Distributions.MvNormal(μ_k, Σ_k)`.

# Optional Keyword Arguments
- `num_control_samples::Int64` -- number of Monte Carlo samples for the control
  trajectory. Default: `10`.
- `deterministic_dynamics::Bool` -- determinies whether to use deterministic prediction
  for the dynamics. If `true`, `num_trajectory_samples` must be 1. Default: `false`.
- `num_trajectory_samples::Int64` -- number of Monte Carlo samples for the state
  trajectory. Default: `10`.
- `μ_θ_init::Float64` -- initial mean parameter `μ_θ` for the risk-sensitivity.
  Default: `1.0`.
- `σ_θ_init::Float64` -- initial covariance parameter `σ_θ` for the risk-sensitivity.
  Default: `2.0`.
- `num_risk_samples::Int64` -- number of Monte Carlo samples for the risk-sensitivity.
  Default: `10`.
- `num_elite::Int64` -- number of elite samples. Default: `10`.
- `iter_max::Int64` -- maximum iteration number. Default: `5`.
- `smoothing_factor::Float64` -- smoothing factor in (0, 1), used to update the
  mean and the variance of the Cross Entropy distribution for the next iteration.
  If `smoothing_factor` is `0.0`, the updated distribution is independent of the
  previous iteration. If it is `1.0`, the updated distribution is the same as the
  previous iteration. Default: `0.1`.
- `mean_carry_over::Bool` -- save `μ_array` & `μ_θ` of the last iteration and use it to
  initialize `μ_array` & `μ_θ` in the next call to `solve!`. Default: `false`.
"""
mutable struct RATCEMSolver
    # CE solver parameters
    num_control_samples::Int64
    deterministic_dynamics::Bool
    num_trajectory_samples::Int64
    num_risk_samples::Int64
    num_elite::Int64
    iter_max::Int64
    smoothing_factor::Float64
    mean_carry_over::Bool

    # action distributions
    μ_init_array::Vector{Vector{Float64}}
    Σ_init_array::Vector{Matrix{Float64}}
    μ_array::Vector{Vector{Float64}}
    Σ_array::Vector{Matrix{Float64}}
    # risk_param distributions
    μ_θ_init::Float64
    σ_θ_init::Float64
    μ_θ::Float64
    σ_θ::Float64
    N::Int64 # control sequence length > 0 (must be the same as N in FiniteHorizonGenerativeProblem)
    iter_current::Int64
end;

function RATCEMSolver(μ_init_array::Vector{Vector{Float64}},
                      Σ_init_array::Vector{Matrix{Float64}};
                      num_control_samples=10,
                      deterministic_dynamics=false,
                      num_trajectory_samples=10,
                      μ_θ_init=1.0,
                      σ_θ_init=2.0,
                      num_risk_samples=10,
                      num_elite=10,
                      iter_max=5,
                      smoothing_factor=0.1,
                      mean_carry_over=false)

    @assert length(μ_init_array) == length(Σ_init_array);
    if deterministic_dynamics
        @assert num_trajectory_samples == 1 "num_trajectory_samples must to be 1";
    end
    μ_array, Σ_array = copy(μ_init_array), copy(Σ_init_array);
    μ_θ, σ_θ = μ_θ_init, σ_θ_init;
    N = length(μ_init_array);
    iter_current = 0;

    return RATCEMSolver(num_control_samples, deterministic_dynamics,
                        num_trajectory_samples, num_risk_samples,
                        num_elite, iter_max, smoothing_factor,
                        mean_carry_over,
                        copy(μ_init_array), copy(Σ_init_array),
                        μ_array, Σ_array,
                        μ_θ_init, σ_θ_init, μ_θ, σ_θ,
                        N, iter_current)
end;

function initialize!(solver::RATCEMSolver)
    solver.iter_current = 0;
    solver.μ_array = copy(solver.μ_init_array);
    solver.Σ_array = copy(solver.Σ_init_array);
    solver.μ_θ = solver.μ_θ_init;
    solver.σ_θ = solver.σ_θ_init;
end

function compute_cost_worker(solver::RATCEMSolver,
                             problem::FiniteHorizonGenerativeProblem,
                             x::Vector{Float64}, #initial state
                             u_array::Vector{Vector{Float64}},
                             rng::AbstractRNG)::Vector{Float64}
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
     return sampled_cost_array
end

function compute_cost(solver::RATCEMSolver,
                      problem::FiniteHorizonGenerativeProblem,
                      x::Vector{Float64}, #initial state
                      control_sequence_array::Vector{Vector{Vector{Float64}}},
                      rng::AbstractRNG)::Vector{Vector{Float64}}
    @assert length(control_sequence_array) == solver.num_control_samples;
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);

    if nprocs() > 1
        proc_id_array = 2 .+ [mod(ii, nprocs() - 1) for ii = 0 : solver.num_control_samples - 1]
    else
        proc_id_array = [1 for ii = 0 : solver.num_control_samples - 1]
    end
    sampled_cost_arrays = Vector{Vector{Float64}}(undef, solver.num_control_samples);
    rng_array = let m = rng
                    [m; accumulate(randjump, fill(big(10)^20, nprocs()-1), init=m)]
                end;
    @sync begin
        for ii = 1 : solver.num_control_samples
            @inbounds @async sampled_cost_arrays[ii] =
                remotecall_fetch(compute_cost_worker, proc_id_array[ii],
                                 solver, problem, x, control_sequence_array[ii],
                                 rng_array[proc_id_array[ii]])
        end
    end
    return sampled_cost_arrays
end

function compute_cost_serial(solver::RATCEMSolver,
                             problem::FiniteHorizonGenerativeProblem,
                             x::Vector{Float64}, #initial state
                             control_sequence_array::Vector{Vector{Vector{Float64}}},
                             rng::AbstractRNG)::Vector{Vector{Float64}}
    @assert length(control_sequence_array) == solver.num_control_samples;
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);

    sampled_cost_arrays = Vector{Vector{Float64}}(undef, solver.num_control_samples);
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
        sampled_cost_arrays[ii] = sampled_cost_array_ii;
    end
    return sampled_cost_arrays
end

function compute_risk(sampled_cost_array::Vector{Float64},
                      θ::Float64)
    risk = logsumexp(θ.*sampled_cost_array) - log(length(sampled_cost_array));
    risk /= θ;
    return risk;
end

function get_elite_samples(solver::RATCEMSolver,
                           control_sequence_array::Vector{Vector{Vector{Float64}}},
                           θ_array::Vector{Float64},
                           sampled_cost_arrays::Vector{Vector{Float64}},
                           kl_bound::Float64)
    @assert length(sampled_cost_arrays) == length(control_sequence_array) == solver.num_control_samples
    @assert all([length(cost_array) == solver.num_trajectory_samples for cost_array in sampled_cost_arrays])
    @assert length(θ_array) == solver.num_risk_samples

    obj_matrix = Matrix{Float64}(undef, solver.num_control_samples, solver.num_risk_samples);
    pq = PriorityQueue{Tuple{Int64, Int64}, Float64}(Base.Order.Reverse);
    u_elite_idx_array = Vector{Int64}();
    θ_elite_idx_array = Vector{Int64}();
    for u_idx = 1 : size(obj_matrix, 1)
        for θ_idx = 1 : size(obj_matrix, 2)
            obj_matrix[u_idx, θ_idx] = compute_risk(sampled_cost_arrays[u_idx], θ_array[θ_idx]);
            obj_matrix[u_idx, θ_idx] += kl_bound/θ_array[θ_idx];
            if length(pq) < solver.num_elite
                enqueue!(pq, (u_idx, θ_idx) => obj_matrix[u_idx, θ_idx])
            else
                if obj_matrix[u_idx, θ_idx] < peek(pq)[2]
                    dequeue!(pq)
                    enqueue!(pq, (u_idx, θ_idx) => obj_matrix[u_idx, θ_idx])
                end
            end
        end
    end
    while !isempty(pq)
        u_idx, θ_idx = peek(pq)[1];
        push!(u_elite_idx_array, u_idx)
        push!(θ_elite_idx_array, θ_idx)
        dequeue!(pq);
    end

    #control_sequence_elite_array = control_sequence_array[u_elite_idx_array];
    #θ_elite_array = θ_array[θ_elite_idx_array]
    return u_elite_idx_array, θ_elite_idx_array, obj_matrix
end

function compute_new_distribution(solver::RATCEMSolver,
                                  control_sequence_elite_array::Vector{Vector{Vector{Float64}}},
                                  θ_elite_array::Vector{Float64})
    @assert length(control_sequence_elite_array) == solver.num_elite
    @assert all([length(sequence) == solver.N for sequence in control_sequence_elite_array]);
    @assert length(θ_elite_array) == solver.num_elite

    μ_new_array = similar(solver.μ_array);
    Σ_new_array = similar(solver.Σ_array);

    mean_array, cov_array = similar(μ_new_array), similar(Σ_new_array);

    for tt = 1 : solver.N # for-loop over time steps
        mean_array[tt] = mean([elite[tt] for elite in control_sequence_elite_array]);
        var_tt = var([elite[tt] for elite in control_sequence_elite_array]);
        if all([elem == 0.0 for elem in var_tt])
            # if all elite samples are the same control sequence, then the variance becomes 0.
            # To prevent the Gaussian from collapsing, add small positive term to var.
            var_tt += 1e-6*ones(size(var_tt))
        end
        cov_array[tt] = Diagonal(var_tt);

        μ_new_array[tt] = (1.0 - solver.smoothing_factor).*mean_array[tt] + solver.smoothing_factor.*solver.μ_array[tt];
        Σ_new_array[tt] = (1.0 - solver.smoothing_factor).*cov_array[tt] + solver.smoothing_factor.*solver.Σ_array[tt];
    end

    μ_θ_new = sum(θ_elite_array)/solver.num_elite;
    σ_θ_new = sqrt(sum((θ_elite_array .- μ_θ_new).^2)/solver.num_elite)
    if σ_θ_new == 0.0
        # if all elite samples are the same risk value, then the std becomes 0.
        # To prevent the Gaussian rom collapsing, add small positive term to std.
        σ_θ_new += 1e-3
    end

    return μ_new_array, Σ_new_array, μ_θ_new, σ_θ_new
end

function step!(solver::RATCEMSolver,
               problem::FiniteHorizonGenerativeProblem,
               x::Vector{Float64},
               kl_bound::Float64,
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
        # sample and action sequence from current distribution;
        control_sequence_array[ii] = Vector{Vector{Float64}}(undef, solver.N)
        for tt = 1 : solver.N # for-loop over time steps
            d = MvNormal(solver.μ_array[tt], solver.Σ_array[tt]);
            u = rand(rng, d);
            control_sequence_array[ii][tt] = u;
        end
    end

    # objective computation & elite selection
    if verbose
        println("****Evaluating risks of sampled control sequences")
    end
    if !serial
        # note that the resulting sampled_cost_arrays is different from the serial version, due to randjump.
        sampled_cost_arrays = compute_cost(solver, problem, x, control_sequence_array, rng);
    else
        sampled_cost_arrays = compute_cost_serial(solver, problem, x, control_sequence_array, rng);
    end
    θ_array = get_positive_samples(solver.μ_θ, solver.σ_θ, solver.num_risk_samples, rng);
    control_sequence_elite_idx_array, θ_elite_idx_array, obj_matrix =
    get_elite_samples(solver, control_sequence_array, θ_array, sampled_cost_arrays, kl_bound);
    if verbose
        println(obj_matrix)
    end
    if verbose
        println("****Choosing $(solver.num_elite) elite samples")
    end

    # update CEM distribution with smoothing
    control_sequence_elite_array = control_sequence_array[control_sequence_elite_idx_array];
    θ_elite_array = θ_array[θ_elite_idx_array];
    if verbose
        println("****Updating control sequence & risk distributions")
    end
    μ_new_array, Σ_new_array, μ_θ_new, σ_θ_new =
        compute_new_distribution(solver, control_sequence_elite_array, θ_elite_array);
    solver.μ_array = μ_new_array;
    solver.Σ_array = Σ_new_array;
    solver.μ_θ = μ_θ_new;
    solver.σ_θ = σ_θ_new;
end;

"""
    solve!(solver::RATCEMSolver,
    problem::FiniteHorizonGenerativeProblem, x_0::Vector{Float64},
    rng::AbstractRNG; kl_bound::Float64, verbose=true, serial=true)

Given `problem` and `solver` (i.e. a RAT CEM Solver), solve distributionally robust
control with current state `x_0` under the KL divergence bound of `kl_bound` (>= 0).

# Return Values (Ordered)
- `μ_array::Vector{Vector{Float64}}` -- array of means `[μ_0,...,μ_{N-1}]` for
  the final Cross Entropy distribution for the control schedule.
- `Σ_array::Vector{Matrix{Float64}}` -- array of covariance matrices `[Σ_0,...,Σ_{N-1}]`
  for the final Cross Entropy distribution for the control schedule.
- `μ_θ::Float64` -- mean for the risk-sensitivity parameter of the final Cross Entropy
  distribution.
- `σ_θ::Float64` -- std for the risk-sensitivity parameter of the final Cross Entropy
  distribution.

# Notes
- Returns an open-loop control policy.
- If `kl_bound` is 0.0, the solver reduces to PETS.
- If `serial` is `true`, Monte Carlo sampling of the Cross Entropy method is serialized
  on a single process. If `false` it is distributed on all the available worker processes.
  We recommend to leave this to `true` as distributed processing can be slower for
  this algorithm.
"""
function solve!(solver::RATCEMSolver,
                problem::FiniteHorizonGenerativeProblem,
                x_0::Vector{Float64},
                rng::AbstractRNG;
                kl_bound::Float64, verbose=true, serial=true)
    @assert kl_bound >= 0 "KL Divergence Bound must be non-negative"
    initialize!(solver);
    if solver.deterministic_dynamics
        @assert solver.num_trajectory_samples == 1 "num_trajectory_samples must to be 1"
    end
    if kl_bound > 0
        while solver.iter_current < solver.iter_max
            step!(solver, problem, x_0, kl_bound, rng, verbose, serial)
        end
        if solver.mean_carry_over
            # time indices are shifted, assuming that the algorithm runs in a receding-horizon fashion.
            solver.μ_init_array[1:end-1] = copy(solver.μ_array[2:end])
            solver.μ_θ_init = solver.μ_θ
        end
    else
        pets_solver = PETSSolver(solver.μ_init_array, solver.Σ_init_array,
                                 num_control_samples=solver.num_control_samples,
                                 deterministic_dynamics=solver.deterministic_dynamics,
                                 num_trajectory_samples=solver.num_trajectory_samples,
                                 num_elite=solver.num_elite,
                                 iter_max=solver.iter_max,
                                 smoothing_factor=solver.smoothing_factor,
                                 mean_carry_over=solver.mean_carry_over);
        μ_array, Σ_array = solve!(pets_solver, problem, x_0, rng,
                                  verbose=verbose, serial=serial);
        solver.μ_init_array = pets_solver.μ_init_array;
        solver.Σ_init_array = pets_solver.Σ_init_array;
        solver.μ_array = pets_solver.μ_array;
        solver.Σ_array = pets_solver.Σ_array;
        solver.μ_θ = 0.0;
        solver.σ_θ = 0.0;
        if solver.mean_carry_over
            solver.μ_θ_init = solver.μ_θ
        end
        solver.iter_current = pets_solver.iter_current;
    end
    return copy(solver.μ_array), copy(solver.Σ_array), solver.μ_θ, solver.σ_θ
end
