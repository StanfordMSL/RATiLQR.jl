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

mutable struct CrossEntropyDirectOptimizationSolver # a.k.a. "PETS"
    # CE solver parameters
    num_control_samples::Int64
    num_trajectory_samples::Int64
    num_elite::Int64
    iter_max::Int64
    smoothing_factor::Float64

    # action_distributions
    μ_init_array::Vector{Vector{Float64}}
    Σ_init_array::Vector{Matrix{Float64}}
    μ_array::Vector{Vector{Float64}}
    Σ_array::Vector{Matrix{Float64}}
    N::Int64 # control sequence length > 0 (must be the same as N in FiniteHorizonGenerativeOptimalControlProblem)
    iter_current::Int64
end

function CrossEntropyDirectOptimizationSolver(μ_init_array::Vector{Vector{Float64}},
                                              Σ_init_array::Vector{Matrix{Float64}};
                                              num_control_samples=10,
                                              num_trajectory_samples=10,
                                              num_elite=3,
                                              iter_max=5,
                                              smoothing_factor=0.1)

    @assert length(μ_init_array) == length(Σ_init_array)
    μ_array, Σ_array = copy(μ_init_array), copy(Σ_init_array);
    N = length(μ_init_array);
    iter_current = 0;

    return CrossEntropyDirectOptimizationSolver(num_control_samples, num_trajectory_samples, num_elite,
                                                iter_max, smoothing_factor, μ_init_array, Σ_init_array,
                                                μ_array, Σ_array, N, iter_current)
end;

function initialize!(direct_solver::CrossEntropyDirectOptimizationSolver)
    direct_solver.iter_current = 0;
    direct_solver.μ_array = copy(direct_solver.μ_init_array);
    direct_solver.Σ_array = copy(direct_solver.Σ_init_array);
end

function compute_cost_worker(direct_solver::CrossEntropyDirectOptimizationSolver,
                              problem::FiniteHorizonGenerativeOptimalControlProblem,
                              x::Vector{Float64}, #initial state
                              u_array::Vector{Vector{Float64}},
                              rng::AbstractRNG, use_true_model=false)
     # trajectory sampling & cost function integration
     x_history_array = Vector{Vector{Vector{Float64}}}(undef, direct_solver.num_trajectory_samples);
     sampled_cost_array = Vector{Float64}(undef, direct_solver.num_trajectory_samples);
     for kk = 1 : direct_solver.num_trajectory_samples # for-loop over trajectory samples
         x_history_array[kk] = Vector{Vector{Float64}}(undef, direct_solver.N + 1);
         x_history_array[kk][1] = copy(x);
         sampled_cost_array[kk] = 0.0;
         for tt = 1 : direct_solver.N # for-loop over time steps
             # get stage cost and perform stochastic transition
             sampled_cost_array[kk] += problem.c(tt - 1, x_history_array[kk][tt], u_array[tt]);
             x_history_array[kk][tt + 1] = problem.f_stochastic(x_history_array[kk][tt], u_array[tt], rng, use_true_model);
         end
         # add terminal cost
         sampled_cost_array[kk] += problem.h(x_history_array[kk][end]);
     end
     # get average cost for this control sequnece u_array
     cost = mean(sampled_cost_array);
end

function compute_cost(direct_solver::CrossEntropyDirectOptimizationSolver,
                      problem::FiniteHorizonGenerativeOptimalControlProblem,
                      x::Vector{Float64}, #initial state
                      control_sequence_array::Vector{Vector{Vector{Float64}}},
                      rng::AbstractRNG, use_true_model=false)
    @assert length(control_sequence_array) == direct_solver.num_control_samples;
    @assert all([length(sequence) == direct_solver.N for sequence in control_sequence_array]);

    if nprocs() > 1
        proc_id_array = 2 .+ [mod(ii, nprocs() - 1) for ii = 0 : direct_solver.num_control_samples - 1]
    else
        proc_id_array = [1 for ii = 0 : direct_solver.num_control_samples - 1]
    end
    cost_array = Vector{Float64}(undef, direct_solver.num_control_samples);
    rng_array = let m = rng
                    [m; accumulate(randjump, fill(big(10)^20, nprocs()-1), init=m)]
                end;
    @sync begin
        for ii = 1 : direct_solver.num_control_samples
            @inbounds @async cost_array[ii] =
                remotecall_fetch(compute_cost_worker, proc_id_array[ii],
                                 direct_solver, problem, x, control_sequence_array[ii],
                                 rng_array[proc_id_array[ii]], use_true_model)
        end
    end
    return cost_array
end

function compute_cost_serial(direct_solver::CrossEntropyDirectOptimizationSolver,
                             problem::FiniteHorizonGenerativeOptimalControlProblem,
                             x::Vector{Float64}, #initial state
                             control_sequence_array::Vector{Vector{Vector{Float64}}},
                             rng::AbstractRNG, use_true_model=false)
    @assert length(control_sequence_array) == direct_solver.num_control_samples;
    @assert all([length(sequence) == direct_solver.N for sequence in control_sequence_array]);

    cost_array = Vector{Float64}(undef, direct_solver.num_control_samples);
    for ii = 1 : direct_solver.num_control_samples # for-loop over action sequences
        # trajectory sampling & cost function integration
        x_history_array_ii = Vector{Vector{Vector{Float64}}}(undef, direct_solver.num_trajectory_samples);
        sampled_cost_array_ii = Vector{Float64}(undef, direct_solver.num_trajectory_samples);
        for kk = 1 : direct_solver.num_trajectory_samples # for-loop over trajectory samples
            x_history_array_ii[kk] = Vector{Vector{Float64}}(undef, direct_solver.N + 1);
            x_history_array_ii[kk][1] = copy(x);
            sampled_cost_array_ii[kk] = 0.0;
            for tt = 1 : direct_solver.N # for-loop over time steps
                # get stage cost and perform stochastic transition
                sampled_cost_array_ii[kk] += problem.c(tt - 1, x_history_array_ii[kk][tt], control_sequence_array[ii][tt]);
                x_history_array_ii[kk][tt + 1] = problem.f_stochastic(x_history_array_ii[kk][tt], control_sequence_array[ii][tt], rng, use_true_model);
            end
            # add terminal cost
            sampled_cost_array_ii[kk] += problem.h(x_history_array_ii[kk][end]);
        end
        # get average cost for this control sequnece
        cost_array[ii] = mean(sampled_cost_array_ii);
    end
    return cost_array
end

function get_elite_samples(direct_solver::CrossEntropyDirectOptimizationSolver,
                           control_sequence_array::Vector{Vector{Vector{Float64}}},
                           cost_array::Vector{Float64})
    @assert length(cost_array) == direct_solver.num_control_samples;
    @assert length(cost_array) == length(control_sequence_array);
    @assert all([length(sequence) == direct_solver.N for sequence in control_sequence_array]);

    control_sequence_cost_pair_array = collect(zip(control_sequence_array, cost_array));
    control_sequence_cost_pair_sorted_array = sort(control_sequence_cost_pair_array, by=x->x[2]);
    control_sequence_elite_array = [pair[1] for pair in control_sequence_cost_pair_sorted_array[1 : direct_solver.num_elite]];

    return control_sequence_elite_array
end

function compute_new_distribution(direct_solver::CrossEntropyDirectOptimizationSolver,
                                  control_sequence_elite_array::Vector{Vector{Vector{Float64}}})
    @assert length(control_sequence_elite_array) == direct_solver.num_elite
    @assert all([length(sequence) == direct_solver.N for sequence in control_sequence_elite_array]);

    μ_new_array = similar(direct_solver.μ_array);
    Σ_new_array = similar(direct_solver.Σ_array);

    mean_array, cov_array = similar(μ_new_array), similar(Σ_new_array);

    for tt = 1 : direct_solver.N # for-loop over time steps
        mean_array[tt] = mean([elite[tt] for elite in control_sequence_elite_array]);
        cov_array[tt] = Diagonal(var([elite[tt] for elite in control_sequence_elite_array]));

        μ_new_array[tt] = (1.0 - direct_solver.smoothing_factor).*mean_array[tt] + direct_solver.smoothing_factor.*direct_solver.μ_array[tt];
        Σ_new_array[tt] = (1.0 - direct_solver.smoothing_factor).*cov_array[tt] + direct_solver.smoothing_factor.*direct_solver.Σ_array[tt];
    end
    return μ_new_array, Σ_new_array
end

function step!(direct_solver::CrossEntropyDirectOptimizationSolver,
               problem::FiniteHorizonGenerativeOptimalControlProblem,
               x::Vector{Float64},
               rng::AbstractRNG,
               use_true_model=false, verbose=true, serial=false)
    direct_solver.iter_current += 1;
    if verbose
        println("**CE iteration $(direct_solver.iter_current)")
    end
    control_sequence_array = Vector{Vector{Vector{Float64}}}(undef, direct_solver.num_control_samples);
    cost_array = Vector{Float64}(undef, direct_solver.num_control_samples);
    # action sequence sampling
    if verbose
        println("****Drawing $(direct_solver.num_control_samples) samples of control sequences of length $(direct_solver.N)")
    end
    for ii = 1 : direct_solver.num_control_samples # for-loop over action sequences
        # sample and action sequence from current distribution;
        control_sequence_array[ii] = Vector{Vector{Float64}}(undef, direct_solver.N)
        for tt = 1 : direct_solver.N # for-loop over time steps
            d = MvNormal(direct_solver.μ_array[tt], direct_solver.Σ_array[tt]);
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
        cost_array = compute_cost(direct_solver, problem, x, control_sequence_array, rng, use_true_model);
    else
        cost_array = compute_cost_serial(direct_solver, problem, x, control_sequence_array, rng, use_true_model);
    end
    if verbose
        println(cost_array)
    end

    # compute elite samples
    if verbose
        println("****Choosing $(direct_solver.num_elite) elite samples")
    end
    control_sequence_elite_array = get_elite_samples(direct_solver, control_sequence_array, cost_array);

    # update CEM distribution with smoothing
    if verbose
        println("****Updating control sequence distribution")
    end
    μ_new_array, Σ_new_array = compute_new_distribution(direct_solver, control_sequence_elite_array);
    direct_solver.μ_array = μ_new_array;
    direct_solver.Σ_array = Σ_new_array;
end;


function solve!(direct_solver::CrossEntropyDirectOptimizationSolver,
                problem::FiniteHorizonGenerativeOptimalControlProblem,
                x_0::Vector{Float64},
                rng::AbstractRNG;
                use_true_model=false, verbose=true, serial=true)
    #setting serial=true by default as it turned out this is much faster than serial=false
    initialize!(direct_solver);
    while direct_solver.iter_current < direct_solver.iter_max
        step!(direct_solver, problem, x_0, rng, use_true_model, verbose, serial)
    end
    return copy(direct_solver.μ_array), copy(direct_solver.Σ_array)
end
