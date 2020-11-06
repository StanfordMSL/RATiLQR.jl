#///////////////////////////////////////
#// File Name: cross_entropy_bilevel_optimization.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/10/28
#// Description: Cross Entropy Bilevel Optimization (RATiLQR) algorithm
#///////////////////////////////////////

using Random
using Distributions
using LinearAlgebra
using Distributed


"""
RATiLQR Solver
"""
mutable struct CrossEntropyBilevelOptimizationSolver
    # ileqg solver parameters
    μ_min_ileqg::Float64           # Minimum damping parameter for regularization
    Δ_0_ileqg::Float64             # Minimum Modification factor for μ
    λ_ileqg::Float64               # Multiplicative factor for line search step parameter in (0, 1)
    d_ileqg::Float64               # Convergence error norm thresholds
    iter_max_ileqg::Int64          # Maximum iteration
    β_ileqg::Float64               # Armijo search condition
    ϵ_init_auto_ileqg::Bool        # Automatic initialization of ϵ_init from the previous iLEQG iteration.
    ϵ_init_ileqg::Float64          # Initial step size for backtracking line search
    ϵ_min_ileqg::Float64           # Minimum step size for backtracking line search
    f_returns_jacobian::Bool       # Whether the dynamics funcion f also returns jacobians or not

    # CE solver parameters
    num_samples::Int64             # Number of CE samples
    num_elite::Int64               # Number of elite samples
    iter_max::Int64                # Maximum iteration
    λ::Float64                     # Multiplicative factor for μ and σ in (0, 1)
    use_θ_max::Bool                # Use maximum feasible θ instead of optimal one

    # CE solver mutable parameters
    μ_init::Float64                # Initial mean parameter
    σ_init::Float64                # Initial standard deviation parameter
    μ::Float64                     # Current mean parameter
    σ::Float64                     # Current standard deviation parameter
    θ_max::Float64                 # Maximum valid θ encountered so far
    θ_min::Float64                 # Minimum valid θ encountered so far
    iter_current::Int64            # Current CE iteration
end

function CrossEntropyBilevelOptimizationSolver(;μ_min_ileqg=1e-6,
                                               Δ_0_ileqg=2.0,
                                               λ_ileqg=0.5,
                                               d_ileqg=1e-2,
                                               iter_max_ileqg=100,
                                               β_ileqg=1e-4,
                                               adaptive_ϵ_init_ileqg=false,
                                               ϵ_init_ileqg=1.0,
                                               ϵ_min_ileqg=1e-6,
                                               μ_init=1.0,
                                               σ_init=2.0,
                                               num_samples=10,
                                               num_elite=3,
                                               iter_max=5,
                                               λ=0.5,
                                               f_returns_jacobian=false,
                                               use_θ_max=false)

    μ, σ = μ_init, σ_init
    θ_max, θ_min = 0.0, Inf
    iter_current = 0

    return CrossEntropyBilevelOptimizationSolver(μ_min_ileqg, Δ_0_ileqg, λ_ileqg, d_ileqg,
                                                 iter_max_ileqg, β_ileqg, adaptive_ϵ_init_ileqg,
                                                 ϵ_init_ileqg, ϵ_min_ileqg, f_returns_jacobian,
                                                 num_samples, num_elite, iter_max, λ, use_θ_max,
                                                 μ_init, σ_init, μ, σ, θ_max, θ_min, iter_current)
end


"""
Initialize RATiLQR Solver
"""
function initialize!(ce_solver::CrossEntropyBilevelOptimizationSolver)
    ce_solver.iter_current = 0;
    ce_solver.μ, ce_solver.σ = ce_solver.μ_init, ce_solver.σ_init
    ce_solver.θ_max = 0.0;
    ce_solver.θ_min = Inf;
end


"""
Compute iLEQG value on a worker process
"""
function compute_value_worker(ce_solver::CrossEntropyBilevelOptimizationSolver,
                              problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                              x::Vector{Float64}, u_array::Vector{Vector{Float64}},
                              θ::Float64)
    ileqg = ILEQGSolver(problem,
                        μ_min=ce_solver.μ_min_ileqg,
                        Δ_0=ce_solver.Δ_0_ileqg,
                        λ=ce_solver.λ_ileqg,
                        d=ce_solver.d_ileqg,
                        iter_max=ce_solver.iter_max_ileqg,
                        β=ce_solver.β_ileqg,
                        adaptive_ϵ_init=ce_solver.ϵ_init_auto_ileqg,
                        ϵ_init=ce_solver.ϵ_init_ileqg,
                        ϵ_min=ce_solver.ϵ_min_ileqg,
                        f_returns_jacobian=ce_solver.f_returns_jacobian)
    initialize!(ileqg, problem, x, u_array)
    value = 0.0
    try
        value = solve!(ileqg, problem, x, u_array, θ=θ, verbose=false)[4];
    catch
        value = Inf
    end
    return value
end


"""
Compute iLEQG values in parallel on multiple worker processes
"""
function compute_cost(ce_solver::CrossEntropyBilevelOptimizationSolver,
                      problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                      x::Vector{Float64},
                      u_array::Vector{Vector{Float64}},
                      θ_array::Vector{Float64},
                      kl_bound::Float64)
    num_samples = length(θ_array)
    if nprocs() > 1
        proc_id_array = 2 .+ [mod(ii, nprocs() - 1) for ii = 0 : num_samples - 1]
    else
        proc_id_array = [1 for ii = 0 : num_samples - 1]
    end
    value_array = Vector{Float64}(undef, num_samples)
    @sync begin
        for ii = 1 : num_samples
            @inbounds @async value_array[ii] =
                remotecall_fetch(compute_value_worker, proc_id_array[ii],
                                ce_solver, problem, x, u_array, θ_array[ii])
        end
    end
    cost_array = value_array .+ kl_bound./θ_array
    return cost_array
end

# For debugging only. It should work in the same way as compute_cost
function compute_cost_serial(ce_solver::CrossEntropyBilevelOptimizationSolver,
                             problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                             x::Vector{Float64},
                             u_array::Vector{Vector{Float64}},
                             θ_array::Vector{Float64},
                             kl_bound::Float64)
    @assert length(θ_array) == ce_solver.num_samples
    cost_array = Vector{Float64}(undef, ce_solver.num_samples)
    for ii = 1 : ce_solver.num_samples
        ileqg_solver = ILEQGSolver(problem,
                                   μ_min=ce_solver.μ_min_ileqg,
                                   Δ_0=ce_solver.Δ_0_ileqg,
                                   λ=ce_solver.λ_ileqg,
                                   d=ce_solver.d_ileqg,
                                   iter_max=ce_solver.iter_max_ileqg,
                                   β=ce_solver.β_ileqg,
                                   adaptive_ϵ_init=ce_solver.ϵ_init_auto_ileqg,
                                   ϵ_init=ce_solver.ϵ_init_ileqg,
                                   ϵ_min=ce_solver.ϵ_min_ileqg,
                                   f_returns_jacobian=ce_solver.f_returns_jacobian)
        initialize!(ileqg_solver, problem, x, u_array)
        try
            cost_array[ii] = solve!(ileqg_solver, problem, x, u_array, θ=θ_array[ii], verbose=false)[4] +
                             kl_bound/θ_array[ii]
        catch
            cost_array[ii] = Inf
        end
    end
    return cost_array
end


"""
Get positive θ samples
"""
function get_positive_samples(μ::Float64, σ::Float64, num_samples::Int64, rng::AbstractRNG)
    θ_array = Float64[];
    d = Normal(μ, σ)
    while true
        θ_sampled = rand(rng, d);
        if θ_sampled > 0.0
            push!(θ_array, θ_sampled)
        end
        if length(θ_array) >= num_samples
            break
        end
    end
    return θ_array
end


"""
Single iteration of RATiLQR
"""
function step!(ce_solver::CrossEntropyBilevelOptimizationSolver,
               problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
               x::Vector{Float64},
               u_array::Vector{Vector{Float64}},
               kl_bound::Float64,
               rng::AbstractRNG,
               verbose=true, serial=false)
    ce_solver.iter_current += 1;
    if verbose
        println("**CE iteration $(ce_solver.iter_current)")
    end
    θ_array = Float64[];
    costs_array = Float64[];
    while true
        if ce_solver.iter_current == 1
            # draw from N(μ_init, σ_init)
            # if too few valid samples, then adjust μ_init, σ_init and redraw.
            # if all samples are valid, then increase μ_init, σ_init for the next iteration.
            if verbose
                println("****Drawing $(ce_solver.num_samples) positive samples of θ ~ N($(round(ce_solver.μ_init,digits=4)), $(round(ce_solver.σ_init,digits=4)))");
            end
            θ_array = get_positive_samples(ce_solver.μ_init, ce_solver.σ_init, ce_solver.num_samples, rng);
        else
            if verbose
                println("****Drawing $(ce_solver.num_samples) positive samples of θ ~ N($(round(ce_solver.μ,digits=4)), $(round(ce_solver.σ,digits=4)))");
            end
            θ_array = get_positive_samples(ce_solver.μ, ce_solver.σ, ce_solver.num_samples, rng);
        end
        if verbose
            println("****Evaluating costs of sampled points")
        end
        if !serial
            costs_array = compute_cost(ce_solver, problem, x, u_array, θ_array, kl_bound)
        else
            costs_array = compute_cost_serial(ce_solver, problem, x, u_array, θ_array, kl_bound)
        end
        if verbose
            println(costs_array)
        end
        num_inf = sum(isinf.(costs_array));
        num_valid = ce_solver.num_samples - num_inf;
        if ce_solver.iter_current == 1 && num_valid < max(ce_solver.num_elite, ce_solver.num_samples*ce_solver.λ)
            if verbose
                println("******$(num_inf)/$(ce_solver.num_samples) samples are Inf. Redrawing samples")
            end
            ce_solver.μ_init *= ce_solver.λ
            ce_solver.σ_init *= ce_solver.λ
        elseif ce_solver.iter_current == 1 && num_valid == ce_solver.num_samples
            ce_solver.μ_init /= ce_solver.λ
            ce_solver.σ_init /= ce_solver.λ
            if verbose
                println("******Increasing μ_init to $(ce_solver.μ_init) and σ_init to $(ce_solver.σ_init)")
            end
            break
        elseif num_valid >= max(ce_solver.num_elite, ce_solver.num_samples*ce_solver.λ)
            if verbose
                println("******$(num_valid)/$(ce_solver.num_samples) samples are valid")
            end
            break
        end
    end

    for ii = 1 : length(θ_array)
        if isinf(costs_array[ii])
            continue
        else
            if θ_array[ii] < ce_solver.θ_min
                ce_solver.θ_min = θ_array[ii]
            elseif θ_array[ii] > ce_solver.θ_max
                ce_solver.θ_max = θ_array[ii]
            end
        end
    end

    θ_cost_pair_array = collect(zip(θ_array, costs_array));
    θ_cost_pair_sorted_array = sort(θ_cost_pair_array, by=x->x[2]);
    θ_elite_array = [pair[1] for pair in θ_cost_pair_sorted_array[1 : ce_solver.num_elite]];
    μ_new = sum(θ_elite_array)/ce_solver.num_elite;
    σ_new = sqrt(sum((θ_elite_array .- μ_new).^2)/ce_solver.num_elite)
    if verbose
        println("****Updated with μ_new: $(round(μ_new,digits=4)) and σ_new: $(round(σ_new,digits=4))")
    end
    ce_solver.μ, ce_solver.σ = μ_new, σ_new;
end


"""
Solve RATiLQR
"""
function solve!(ce_solver::CrossEntropyBilevelOptimizationSolver,
                problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                x_0::Vector{Float64}, u_array::Vector{Vector{Float64}}, rng::AbstractRNG;
                kl_bound::Float64, verbose=true, serial=false)
    @assert kl_bound >= 0 "KL Divergence Bound must be non-negative"
    initialize!(ce_solver);
    if kl_bound > 0
        while ce_solver.iter_current < ce_solver.iter_max
            step!(ce_solver, problem, x_0, u_array, kl_bound, rng, verbose, serial)
        end
        θ_min, θ_max = ce_solver.θ_min, ce_solver.θ_max;
        if ce_solver.use_θ_max
            θ_opt = θ_max
            if verbose
                println("Using θ_max = $(round(θ_max, digits=4)) in place of μ = $(round(ce_solver.μ, digits=4))")
            end
        else
            θ_opt = ce_solver.μ
        end
        if verbose
            println("Maximum iteration number reached. θ_opt = $(round(θ_opt,digits=4)) from [$(round(θ_min,digits=4)), $(round(θ_max,digits=4))]")
        end
    else
        # kl_bound is 0. In this case the optimal controller is the risk-neutral (iLQG) controller.
        θ_opt = 0.0;
    end
    while true
        try
            ileqg_solver = ILEQGSolver(problem,
                                       μ_min=ce_solver.μ_min_ileqg,
                                       Δ_0=ce_solver.Δ_0_ileqg,
                                       λ=ce_solver.λ_ileqg,
                                       d=ce_solver.d_ileqg,
                                       iter_max=ce_solver.iter_max_ileqg,
                                       β=ce_solver.β_ileqg,
                                       adaptive_ϵ_init=ce_solver.ϵ_init_auto_ileqg,
                                       ϵ_init=ce_solver.ϵ_init_ileqg,
                                       ϵ_min=ce_solver.ϵ_min_ileqg,
                                       f_returns_jacobian=ce_solver.f_returns_jacobian)
            initialize!(ileqg_solver, problem, x_0, u_array)
            x_array, l_array, L_array, value, ~ = solve!(ileqg_solver, problem, x_0, u_array, θ=θ_opt, verbose=verbose)
            if kl_bound > 0
                return θ_opt, x_array, l_array, L_array, value + kl_bound/θ_opt, θ_min, θ_max
            else
                return θ_opt, x_array, l_array, L_array, value, 0.0, 0.0
            end
        catch
            @warn "θ_opt == $(θ_opt) resulted in neurotic breakdown. Re-trying with θ_opt == $(max(0.0, θ_opt - ce_solver.σ))"
            θ_opt = max(0.0, θ_opt - ce_solver.σ)
        end
    end
end
