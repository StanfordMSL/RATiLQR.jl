#///////////////////////////////////////
#// File Name: nelder_mead_bilevel_optimization.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/05
#// Description: Nelder-Mead Simplex Bilevel Optimization algorithm
#///////////////////////////////////////

using Printf


"""
    NelderMeadBilevelOptimizationSolver(kwargs...)

RAT iLQR++ (i.e. Nelder-Mead Simplex Method + iLEQG) Solver.

# Optional Keyword Arguments
## iLEQG Solver Parameters
- `μ_min_ileqg::Float64` -- minimum value for Hessian regularization parameter `μ` (> 0).
  Default: `1e-6`.
- `Δ_0_ileqg::Float64` -- minimum multiplicative modification factor (> 0) for `μ`.
  Default: `2.0`.
- `λ_ileqg::Float64` -- multiplicative modification factor in (0, 1) for line search
  step size `ϵ`. Default: `0.5`.
- `d_ileqg::Float64` -- convergence error norm threshold (> 0). If the maximum l2
  norm of the change in nominal control over the horizon is less than `d`, the
  solver is considered to be converged. Default: `1e-2`.
- `iter_max_ileqg::Int64` -- maximum iteration number. Default: 100.
- `ϵ_init_ileqg::Float64` -- initial step size in (`ϵ_min`, 1] to start the
  backtracking line search with. If `adaptive_ϵ_init` is `true`, then this
  value is overridden by the solver's adaptive initialization functionality
  after the first iLEQG iteration. If `adaptive_ϵ_init` is `false`, the
  specified value of `ϵ_init` is used across all the iterations as the initial
  step size. Default:`1.0`.
- `adaptive_ϵ_init_ileqg::Bool` -- if `true`, `ϵ_init` is adaptively changed based on
  the last step size `ϵ` of the previous iLEQG iteration. Default: `false`.
   - If the first line search iterate `ϵ_init_prev` in the previous iLEQG
     iteration is successful, then `ϵ_init` for the next iLEQG iteration is set
     to `ϵ_init = ϵ_init_prev / λ` so that the initial line search step increases.
   - Otherwise `ϵ_init = ϵ_last` where `ϵ_last` is the line search step accepted
     in the previous iLEQG iteration.
- `ϵ_min_ileqg::Float64` -- minimum value of step size `ϵ` to terminate the line
  search. When `ϵ_min` is reached, the last candidate nominal trajectory is accepted
  regardless of the Armijo condition and the current iLEQG iteration is
  finished. Default: `1e-6`.
- `f_returns_jacobian::Bool` -- if `true`, Jacobian matrices of the dynamics function
  are user-provided. This can reduce computation time since automatic
  differentiation is not used. Default: `false`.

## Nelder-Mead Simplex Solver Parameters
- `α::Float64` -- reflection parameter. Default: `1.0`.
- `β::Float64` -- expansion parameter. Default: `2.0`.
- `γ::Float64` -- contraction parameter. Default: `0.5`.
- `ϵ::Float64` -- convergence parameter. The algorithm is said to have convergeced
   if the standard deviation of the objective values at the vertices of the simplex
   is below `ϵ`. Default: `1e-2`.
- `λ::Float64` -- multiplicative modification factor in (0, 1) for `θ_high_init` and
  `θ_low_init`, which is repeatedly applied in case the objective value is infinity
  until a feasible region is find. Default: `0.5`.
- `θ_high_init::Float64` -- Initial guess for `θ_high`. Default: `3.0`.
- `θ_low_init::Float64` -- Initial guess for `θ_low`. Default: `1e-8`.
- `iter_max::Int64` -- maximum iteration number. Default: `100`.

# Notes
- The Nelder-Mead Simplex method maintains a 1D simplex (i.e. a line segment that
  consists of 2 points, `θ_high` and `θ_low`) to search for the optimal risk-sensitivity
  parameter `θ`. `θ_high` and `θ_low` refer to the verteces of the simplex with the highest
  and the lowest objective values, respectively.
- The initial guesses `θ_high_init` and `θ_low_init`, which may be modified during optimization,
  are stored internally in the solver and carried over to the next call to `solve!`.
"""
mutable struct NelderMeadBilevelOptimizationSolver
    # ileqg solver parameters
    μ_min_ileqg::Float64           # Minimum damping parameter for regularization
    Δ_0_ileqg::Float64             # Minimum Modification factor for μ
    λ_ileqg::Float64               # Multiplicative factor for line search step parameter in (0, 1)
    d_ileqg::Float64               # Convergence error norm thresholds
    iter_max_ileqg::Int64          # Maximum iteration
    # β_ileqg::Float64               # Armijo search condition
    ϵ_init_auto_ileqg::Bool        # Automatic initialization of ϵ_init from the previous iLEQG iteration.
    ϵ_init_ileqg::Float64          # Initial step size for backtracking line search
    ϵ_min_ileqg::Float64           # Minimum step size for backtracking line search
    f_returns_jacobian::Bool       # Whether the dynamics funcion f also returns jacobians or not

    # Nelder-Mead Simplex solver parameters
    α::Float64           # Reflection parameter
    β::Float64           # Expansion parameter
    γ::Float64           # Contraction parameter
    ϵ::Float64           # Convergence parameter
    λ::Float64           # Multiplicative factor for θ_high and θ_low in (0, 1)
    iter_max::Int64      # Maximum iteration

    # Nelder-Mead Simplex solver mutable parameters
    θ_high_init::Float64
    θ_low_init::Float64
    iter_current::Int64  # Current Nelder-Mead iteration
    θ_high::Float64
    θ_low::Float64
    c_high::Union{Nothing, Float64}
    c_low::Union{Nothing, Float64}
end

function NelderMeadBilevelOptimizationSolver(;μ_min_ileqg=1e-6,
                                              Δ_0_ileqg=2.0,
                                              λ_ileqg=0.5,
                                              d_ileqg=1e-2,
                                              iter_max_ileqg=100,
                                              # β_ileqg=1e-4,
                                              adaptive_ϵ_init_ileqg=false,
                                              ϵ_init_ileqg=1.0,
                                              ϵ_min_ileqg=1e-6,
                                              f_returns_jacobian=false,
                                              α=1.0,
                                              β=2.0,
                                              γ=0.5,
                                              ϵ=1e-2,
                                              λ=0.5,
                                              iter_max=100,
                                              θ_high_init=3.0,
                                              θ_low_init=1e-8)
    θ_high, θ_low = θ_high_init, θ_low_init;
    iter_current = 0

    return NelderMeadBilevelOptimizationSolver(μ_min_ileqg, Δ_0_ileqg, λ_ileqg, d_ileqg,
                                               iter_max_ileqg, adaptive_ϵ_init_ileqg,
                                               ϵ_init_ileqg, ϵ_min_ileqg, f_returns_jacobian,
                                               α, β, γ, ϵ, λ, iter_max, θ_high_init, θ_low_init,
                                               iter_current, θ_high, θ_low, nothing, nothing)
end


"""
Compute Nelder-Mead cost on a worker process
"""
function compute_cost_worker(nm_solver::NelderMeadBilevelOptimizationSolver,
                             problem::FiniteHorizonAdditiveGaussianProblem,
                             x::Vector{Float64}, u_array::Vector{Vector{Float64}},
                             θ::Float64, kl_bound::Float64)
    ileqg = ILEQGSolver(problem,
                        μ_min=nm_solver.μ_min_ileqg,
                        Δ_0=nm_solver.Δ_0_ileqg,
                        λ=nm_solver.λ_ileqg,
                        d=nm_solver.d_ileqg,
                        iter_max=nm_solver.iter_max_ileqg,
                        # β=nm_solver.β_ileqg,
                        adaptive_ϵ_init=nm_solver.ϵ_init_auto_ileqg,
                        ϵ_init=nm_solver.ϵ_init_ileqg,
                        ϵ_min=nm_solver.ϵ_min_ileqg,
                        f_returns_jacobian=nm_solver.f_returns_jacobian)
    #initialize!(ileqg, problem, x, u_array, θ)
    cost = 0.0
    try
        cost = solve!(ileqg, problem, x, u_array, θ=θ, verbose=false)[4] +
               kl_bound/θ;
    catch
        cost = Inf
    end
    return cost
end


"""
Initialize NelderMeadBilevelOptimization solver
"""
function initialize!(nm_solver::NelderMeadBilevelOptimizationSolver)
    nm_solver.iter_current = 0;
    nm_solver.θ_low = nm_solver.θ_low_init;
    nm_solver.θ_high = nm_solver.θ_high_init;
end


"""
Single iteration of NelderMeadBilevelOptimization solver
"""
function step!(nm_solver::NelderMeadBilevelOptimizationSolver,
               problem::FiniteHorizonAdditiveGaussianProblem,
               x::Vector{Float64},
               u_array::Vector{Vector{Float64}},
               kl_bound::Float64,
               verbose=true)
    nm_solver.iter_current += 1;
    if verbose
        println("**Nelder-Mead iteration $(nm_solver.iter_current)")
    end
    if nm_solver.c_high < nm_solver.c_low
        nm_solver.θ_low, nm_solver.θ_high = nm_solver.θ_high, nm_solver.θ_low;
        nm_solver.c_low, nm_solver.c_high = nm_solver.c_high, nm_solver.c_low;
    end
    if verbose
        println("****(θ_low, c_low) == ($(round(nm_solver.θ_low, digits=4)), $(round(nm_solver.c_low, digits=3)))")
        println("****(θ_high, c_high) == ($(round(nm_solver.θ_high, digits=4)), $(round(nm_solver.c_high, digits=3)))")
    end

    θ_m = nm_solver.θ_low;
    # reflection
    θ_r = θ_m + nm_solver.α*(θ_m - nm_solver.θ_high);
    θ_r = max(nm_solver.θ_low_init, θ_r)
    c_r = compute_cost_worker(nm_solver, problem, x, u_array, θ_r, kl_bound);
    if verbose
        println("****Reflection point: (θ_r, c_r) == ($(round(θ_r, digits=4)), $(round(c_r, digits=3)))")
    end

    if c_r < nm_solver.c_low
        # expansion
        θ_e = θ_m + nm_solver.β*(θ_r - θ_m);
        θ_e = max(nm_solver.θ_low_init, θ_e)
        c_e = compute_cost_worker(nm_solver, problem, x, u_array, θ_e, kl_bound);
        if verbose
            println("****Expansion point: (θ_e, c_e) == ($(round(θ_e, digits=4)), $(round(c_e, digits=3)))")
        end
        if c_e < c_r
            if verbose
                println("****(θ_high, c_high) <-- (θ_e, c_e)")
            end
            nm_solver.θ_high = θ_e;
            nm_solver.c_high = c_e;
        else
            if verbose
                println("****(θ_high, c_high) <-- (θ_r, c_r)")
            end
            nm_solver.θ_high = θ_r;
            nm_solver.c_high = c_r;
        end
    else
        if verbose
            println("****(θ_high, c_high) <-- (θ_r, c_r)")
        end
        if c_r < nm_solver.c_high
            nm_solver.θ_high = θ_r;
            nm_solver.c_high = c_r;
        end
        # contraction
        θ_c = θ_m + nm_solver.γ*(nm_solver.θ_high - θ_m);
        θ_c = max(nm_solver.θ_low_init, θ_c)
        c_c = compute_cost_worker(nm_solver, problem, x, u_array, θ_c, kl_bound);
        if verbose
            println("****Contraction point: (θ_c, c_c) == ($(round(θ_c, digits=4)), $(round(c_c, digits=3)))")
        end
        if c_c > nm_solver.c_high
            nm_solver.θ_high = (nm_solver.θ_high + nm_solver.θ_low)/2;
            nm_solver.c_high = compute_cost_worker(nm_solver, problem, x, u_array, nm_solver.θ_high, kl_bound);
            if verbose
                println("****Shrinking θ_high: (θ_high, c_high) == ($(round(nm_solver.θ_high, digits=4)), $(round(nm_solver.c_high, digits=3)))")
            end
        else
            if verbose
                println("****(θ_high, c_high) <-- (θ_c, c_c)")
            end
            nm_solver.θ_high = θ_c;
            nm_solver.c_high = c_c;
        end
    end
end


"""
    solve!(nm_solver::NelderMeadBilevelOptimizationSolver,
    problem::FiniteHorizonAdditiveGaussianProblem, x_0::Vector{Float64},
    u_array::Vector{Vector{Float64}}; kl_bound::Float64, verbose=true)

Given `problem` and `nm_solver` (i.e. a RAT iLQR++ Solver), solve distributionally robust
control with current state `x_0` and nominal control schedule `u_array = [u_0, ..., u_{N-1}]`
under the KL divergence bound of `kl_bound` (>= 0).

# Return Values (Ordered)
- `θ_opt::Float64` -- optimal risk-sensitivity parameter.
- `x_array::Vector{Vector{Float64}}` -- nominal state trajectory `[x_0,...,x_N]`.
- `l_array::Vector{Vector{Float64}}` -- nominal control schedule `[l_0,...,l_{N-1}]`.
- `L_array::Vector{Matrix{Float64}}` -- feedback gain schedule `[L_0,...,L_{N-1}]`.
- `value::Float64` -- optimal cost-to-go (i.e. objective value) found by the solver.

# Notes
- Returns a time-varying affine state-feedback policy `π_k` of the form
  `π_k(x) = L_k(x - x_k) + l_k`.
- If `kl_bound` is 0.0, the solver reduces to iLQG.
"""
function solve!(nm_solver::NelderMeadBilevelOptimizationSolver,
                problem::FiniteHorizonAdditiveGaussianProblem,
                x_0::Vector{Float64}, u_array::Vector{Vector{Float64}};
                kl_bound::Float64, verbose=true)
    @assert kl_bound >= 0 "KL Divergence Bound must be non-negative"
    initialize!(nm_solver);
    if kl_bound > 0
        if isnothing(nm_solver.c_high)
            while true
                nm_solver.c_high = compute_cost_worker(nm_solver, problem, x_0, u_array, nm_solver.θ_high, kl_bound)
                if !isinf(nm_solver.c_high)
                    break;
                else
                    nm_solver.θ_high *= nm_solver.λ;
                    nm_solver.θ_high_init *= nm_solver.λ;
                end
            end
        end
        if isnothing(nm_solver.c_low)
            while true
                nm_solver.c_low = compute_cost_worker(nm_solver, problem, x_0, u_array, nm_solver.θ_low, kl_bound)
                if !isinf(nm_solver.c_low)
                    break;
                else
                    nm_solver.θ_low *= nm_solver.λ;
                    nm_solver.θ_low_init *= nm_solver.λ;
                end
            end
        end

        while true
            step!(nm_solver, problem, x_0, u_array, kl_bound, verbose)
            # compute stdev
            c_mean = (nm_solver.c_low + nm_solver.c_high)/2;
            stdev = sqrt(0.5*((nm_solver.c_high - c_mean)^2 + (nm_solver.c_low - c_mean)^2));
            stdev_str = @sprintf "%3.3f" stdev
            if stdev < nm_solver.ϵ
                if verbose
                    println("Nelder-Mead Converged. stdev == $(stdev_str)")
                end
                break;
            end
            if nm_solver.iter_current == nm_solver.iter_max
                if verbose
                    println("Maximum iteration number reached. stdev == $(stdev_str)")
                end
                break;
            end
        end
        θ_opt = nm_solver.θ_low;
        if verbose
            println("θ_opt == $(round(θ_opt, digits=4))")
        end
        #nm_solver.θ_high_init = nm_solver.θ_high;
    else
        # kl_bound is 0. In this case the optimal controller is the risk-neutral (iLQG) controller.
        θ_opt = 0.0;
    end
    ileqg_solver = ILEQGSolver(problem,
                               μ_min=nm_solver.μ_min_ileqg,
                               Δ_0=nm_solver.Δ_0_ileqg,
                               λ=nm_solver.λ_ileqg,
                               d=nm_solver.d_ileqg,
                               iter_max=nm_solver.iter_max_ileqg,
                               # β=nm_solver.β_ileqg,
                               adaptive_ϵ_init=nm_solver.ϵ_init_auto_ileqg,
                               ϵ_init=nm_solver.ϵ_init_ileqg,
                               ϵ_min=nm_solver.ϵ_min_ileqg,
                               f_returns_jacobian=nm_solver.f_returns_jacobian)
    #initialize!(ileqg_solver, problem, x_0, u_array, θ_opt)
    x_array, l_array, L_array, value, ~ = solve!(ileqg_solver, problem, x_0, u_array, θ=θ_opt, verbose=verbose)
    if kl_bound > 0
        return θ_opt, x_array, l_array, L_array, value + kl_bound/θ_opt
    else
        return θ_opt, x_array, l_array, L_array, value
    end
end
