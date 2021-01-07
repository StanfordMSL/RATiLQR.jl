#///////////////////////////////////////
#// File Name: ileqg.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/10/28
#// Description: Iterative Linear-Exponential-Quadratic-Gaussian algorithm
#///////////////////////////////////////

using LinearAlgebra
using ForwardDiff
using Printf
using Random
using Distributions


"""
Simulate noiseless dynamics x_{t+1} = f(x_t, u_t) with u_array from x_0.
"""
function simulate_dynamics(problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                           x_0::Vector{Float64}, u_array::Vector{Vector{Float64}};
                           f_returns_jacobian=false)
    @assert problem.N == length(u_array)
    x_array = Vector{Vector{Float64}}(undef, problem.N + 1)
    x_array[1] = copy(x_0);
    if f_returns_jacobian
        A_array = Vector{Matrix{Float64}}(undef, problem.N)
        B_array = Vector{Matrix{Float64}}(undef, problem.N)
        for ii = 1 : problem.N
            x_array[ii + 1], A_array[ii], B_array[ii] =
                problem.f(x_array[ii], u_array[ii], true)
        end
        return x_array, A_array, B_array
    else
        for ii = 1 : problem.N
            x_array[ii + 1] = problem.f(x_array[ii], u_array[ii])
        end
        return x_array
    end
end;


"""
Simulate noisy dynamics x_{t+1} = f(x_t, u_t) + w_t with w_t ~ N(0, W(t)) and u_array from x_0.
"""
function simulate_dynamics(problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                           x_0::Vector{Float64}, u_array::Vector{Vector{Float64}},
                           rng::AbstractRNG)
    @assert problem.N == length(u_array)
    x_array = Vector{Vector{Float64}}(undef, problem.N + 1)
    x_array[1] = x_0;
    for ii = 1 : problem.N
        d = MvNormal(zeros(length(x_0)), problem.W(ii - 1));
        x_array[ii + 1] = problem.f(x_array[ii], u_array[ii]) + rand(rng, d)
    end
    return x_array
end;


"""
Simulate noiseless dynamics x_{t+1} = f(x_t, u_t) with x_array and state feedback u = L*x + l
"""
function simulate_dynamics(problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                           x_array::Vector{Vector{Float64}},
                           l_array::Vector{Vector{Float64}},
                           L_array::Vector{Matrix{Float64}};
                           f_returns_jacobian=false)
    @assert problem.N == length(l_array) && problem.N == length(L_array)
    u_array = Vector{Vector{Float64}}(undef, problem.N)
    x_array_new = Vector{Vector{Float64}}(undef, problem.N + 1)
    x_array_new[1] = copy(x_array[1]);
    if f_returns_jacobian
        A_array = Vector{Matrix{Float64}}(undef, problem.N)
        B_array = Vector{Matrix{Float64}}(undef, problem.N)
        for ii = 1 : problem.N
            u_array[ii] = l_array[ii] + L_array[ii]*(x_array_new[ii] - x_array[ii])
            x_array_new[ii + 1], A_array[ii], B_array[ii] =
                problem.f(x_array_new[ii], u_array[ii], true)
        end
        return x_array_new, u_array, A_array, B_array
    else
        for ii = 1 : problem.N
            u_array[ii] = l_array[ii] + L_array[ii]*(x_array_new[ii] - x_array[ii])
            x_array_new[ii + 1] = problem.f(x_array_new[ii], u_array[ii])
        end
        return x_array_new, u_array
    end
end


"""
Simulate noisy dynamics x_{t+1} = f(x_t, u_t) + w_t with w_t ~ N(0, W(t)), x_array and state feedback u = L*x + l
"""
function simulate_dynamics(problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                           x_array::Vector{Vector{Float64}},
                           l_array::Vector{Vector{Float64}},
                           L_array::Vector{Matrix{Float64}},
                           rng::AbstractRNG)
    @assert problem.N == length(l_array) && problem.N == length(L_array)
    u_array = Vector{Vector{Float64}}(undef, problem.N)
    x_array_new = Vector{Vector{Float64}}(undef, problem.N + 1)
    x_array_new[1] = copy(x_array[1]);
    for ii = 1 : problem.N
        d = MvNormal(zeros(length(x_array_new[ii])), problem.W(ii - 1));
        u_array[ii] = l_array[ii] + L_array[ii]*(x_array_new[ii] - x_array[ii])
        x_array_new[ii + 1] = problem.f(x_array_new[ii], u_array[ii]) + rand(rng, d)
    end
    return x_array_new, u_array
end


"""
Integrate cost given state and control trajectories
"""
function integrate_cost(problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                        x_array::Vector{Vector{Float64}}, u_array::Vector{Vector{Float64}})
    @assert problem.N == length(u_array) && problem.N + 1 == length(x_array)
    cost = 0.0;
    for ii = 1 : problem.N
        cost += problem.c(ii - 1, x_array[ii], u_array[ii])
    end
    cost += problem.h(x_array[end])
    return cost
end;


"""
    iLEQGSolver(problem::FiniteHorizonRiskSensitiveOptimalControlProblems, kwargs...)

iLQG and iLEQG Solver for `problem`.

# Optional Keyword Arguments
- `μ_min::Float64` -- minimum value for Hessian regularization parameter `μ` (> 0).
  Default: `1e-6`.
- `Δ_0::Float64` -- minimum multiplicative modification factor (> 0) for `μ`.
  Default: `2.0`.
- `λ::Float64` -- multiplicative modification factor in (0, 1) for line search
  step size `ϵ`. Default: `0.5`.
- `d::Float64` -- convergence error norm threshold (> 0). If the maximum l2
  norm of the change in nominal control over the horizon is less than `d`, the
  solver is considered to be converged. Default: `1e-2`.
- `iter_max::Int64` -- maximum iteration number. Default: 100.
- `β::Float64` -- Armijo condition number (>= 0) defining a sufficient decrease
  for backtracking line search. If `β == 0`, then any cost-to-go improvement is
  considered a sufficient decrease. Default: `1e-4`.
- `ϵ_init::Float64` -- initial step size in (`ϵ_min`, 1] to start the
  backtracking line search with. If `adaptive_ϵ_init` is `true`, then this
  value is overridden by the solver's adaptive initialization functionality
  after the first iLEQG iteration. If `adaptive_ϵ_init` is `false`, the
  specified value of `ϵ_init` is used across all the iterations as the initial
  step size. Default:`1.0`.
- `adaptive_ϵ_init::Bool` -- if `true`, `ϵ_init` is adaptively changed based on
  the last step size `ϵ` of the previous iLEQG iteration. Default: `false`.
   - If the first line search iterate `ϵ_init_prev` in the previous iLEQG
     iteration is successful, then `ϵ_init` for the next iLEQG iteration is set
     to `ϵ_init = ϵ_init_prev / λ` so that the initial line search step increases.
   - Otherwise `ϵ_init = ϵ_last` where `ϵ_last` is the line search step accepted
     in the previous iLEQG iteration.
- `ϵ_min::Float64` -- minimum value of step size `ϵ` to terminate the line
  search. When `ϵ_min` is reached, the last candidate nominal trajectory is accepted
  regardless of the Armijo condition and the current iLEQG iteration is
  finished. Default: `1e-6`.
- `f_returns_jacobian::Bool` -- if `true`, Jacobian matrices of the dynamics function
  are user-provided. This can reduce computation time since automatic
  differentiation is not used. Default: `false`.
"""
mutable struct ILEQGSolver
    μ_min::Float64   # Minimum damping parameter for regularization
    μ::Float64       # Damping parameter for regularization
    Δ_0::Float64     # Minimum Modification factor for μ
    Δ::Float64       # Modification factor for μ
    λ::Float64       # Multiplicative factor for line search step parameter
    d::Float64       # Convergence error norm thresholds
    iter_max::Int64  # Maximum iteration
    β::Float64       # Armijo search condition
    ϵ_init_auto::Bool # Automatic initialization of ϵ_init from the previous iLEQG iteration.
    ϵ_init::Float64  # Initial step size for backtracking line search
    ϵ_min::Float64   # Minimum step size for backtracking line search
    f_returns_jacobian::Bool

    x_array::Vector{Vector{Float64}} # Nominal state trajectory
    l_array::Vector{Vector{Float64}} # Linear control schedule
    L_array::Vector{Matrix{Float64}} # Feedback gain schedule

    A_array::Union{Nothing, Vector{Matrix{Float64}}} # Sequence of Jacobians [dx_1/dx_0, ..., dx_{T}/dx_{T-1}]
    B_array::Union{Nothing, Vector{Matrix{Float64}}} # Sequence of Jacobians [dx_1/du_0, ..., dx_{T}/du_0]

    iter_current::Int64 # current iteration number
    d_current::Float64  # current error norm
    ϵ_history::Array{Tuple{Float64, Float64}} # Array of (ϵ, -value_improvement)
    ϵ_init_init::Float64 # initial value of ϵ_init
end;

function ILEQGSolver(problem::FiniteHorizonRiskSensitiveOptimalControlProblem;
                     μ_min=1e-6, Δ_0=2.0, λ=0.5, d=1e-2, iter_max=100,
                     β=1e-4, ϵ_init=1.0, adaptive_ϵ_init=false,
                     ϵ_min=1e-6, f_returns_jacobian=false)
    @assert 0 < λ < 1 "λ has to be in (0, 1)"
    @assert d > 0 "d > 0 is necessary"
    @assert μ_min > 0 "μ_min > 0 is necessary"
    @assert Δ_0 > 0 "Δ_0 > 0 is necessary"
    @assert β >= 0 "β >= 0 is necessary"
    @assert 0 < ϵ_init <= 1 "ϵ_init has to be in (0, 1]"
    @assert ϵ_init > ϵ_min "ϵ_init > ϵ_min is necessary"
    @assert 0 < ϵ_min < 1 "ϵ_min has to be in (0, 1)"

    x_array = Vector{Vector{Float64}}(undef, problem.N + 1)
    l_array = Vector{Vector{Float64}}(undef, problem.N)
    L_array = Vector{Matrix{Float64}}(undef, problem.N)
    return ILEQGSolver(μ_min, μ_min, Δ_0, Δ_0, λ, d, iter_max, β, adaptive_ϵ_init, ϵ_init, ϵ_min, f_returns_jacobian,
                       x_array, l_array, L_array, nothing, nothing, 0, Inf, Tuple{Float64, Float64}[], ϵ_init)
end;


"""
Initialize iLEQG solver
"""
function initialize!(ileqg::ILEQGSolver, problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                     x_0::Vector{Float64}, u_array::Vector{Vector{Float64}})
    ileqg.μ, ileqg.Δ = ileqg.μ_min, ileqg.Δ_0;
    ileqg.d_current = Inf;
    ileqg.iter_current = 0;
    ileqg.ϵ_init = ileqg.ϵ_init_init
    ileqg.ϵ_history = Tuple{Float64, Float64}[];
    if ileqg.f_returns_jacobian
        ileqg.x_array, ileqg.A_array, ileqg.B_array =
            simulate_dynamics(problem, x_0, u_array, f_returns_jacobian=true)
    else
        ileqg.x_array = simulate_dynamics(problem, x_0, u_array, f_returns_jacobian=false);
        ileqg.A_array, ileqg.B_array = nothing, nothing
    end
    ileqg.l_array = copy(u_array);
    x_dim, u_dim = length(ileqg.x_array[1]), length(ileqg.l_array[1])
    for ii = 1 : problem.N
        ileqg.L_array[ii] = zeros(u_dim, x_dim)
    end
end


"""
Model Approximation
"""
struct ApproximationResult
    q_array::Vector{Float64}
    q_vec_array::Vector{Vector{Float64}}
    Q_array::Vector{Matrix{Float64}}
    r_array::Vector{Vector{Float64}}
    R_array::Vector{Matrix{Float64}}
    P_array::Vector{Matrix{Float64}}
    A_array::Vector{Matrix{Float64}}
    B_array::Vector{Matrix{Float64}}
    W_array::Vector{Matrix{Float64}}
end


"""
Linearize the dynamics and quadratize the costs around the nominal trajectory.
"""
function approximate_model(problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                           u_array::Vector{Vector{Float64}},
                           x_array::Vector{Vector{Float64}},
                           A_array_input::Union{Nothing, Vector{Matrix{Float64}}}=nothing,
                           B_array_input::Union{Nothing, Vector{Matrix{Float64}}}=nothing)
    f, c, h, W = problem.f, problem.c, problem.h, problem.W;
    # linearization & quadratization functions
    fx(x, u) = ForwardDiff.jacobian(a -> f(a, u), x);
    fu(x, u) = ForwardDiff.jacobian(b -> f(x, b), u);
    cx(k, x, u) = ForwardDiff.gradient(a -> c(k, a, u), x);
    cu(k, x, u) = ForwardDiff.gradient(b -> c(k, x, b), u);
    cux(k, x, u) = ForwardDiff.jacobian(a -> ForwardDiff.gradient(b -> c(k, a, b), u), x);
    cxx(k, x, u) = Symmetric(ForwardDiff.hessian(a -> c(k, a, u), x));
    cuu(k, x, u) = Symmetric(ForwardDiff.hessian(b -> c(k, x, b), u));
    hx(x) = ForwardDiff.gradient(a -> h(a), x);
    hxx(x) = Symmetric(ForwardDiff.hessian(a -> h(a), x));

    # compute approximations
    @assert problem.N == length(u_array)
    N = problem.N;
    if !isnothing(A_array_input)
        @assert length(A_array_input) == N
    end
    if !isnothing(B_array_input)
        @assert length(B_array_input) == N
    end
    q_array = Vector{Float64}(undef, N + 1)
    q_vec_array = Vector{Vector{Float64}}(undef, N + 1)
    Q_array = Vector{Matrix{Float64}}(undef, N + 1)
    r_array = Vector{Vector{Float64}}(undef, N)
    R_array = Vector{Matrix{Float64}}(undef, N)
    P_array = Vector{Matrix{Float64}}(undef, N)
    A_array = Vector{Matrix{Float64}}(undef, N)
    B_array = Vector{Matrix{Float64}}(undef, N)
    W_array = Vector{Matrix{Float64}}(undef, N)
    # Threads.@threads for ii = 1 : N
    for ii = 1 : N
        @inbounds x, u = x_array[ii], u_array[ii];
        @inbounds q_array[ii] = c(ii - 1, x, u);
        @inbounds q_vec_array[ii] = cx(ii - 1, x, u);
        @inbounds Q_array[ii] = cxx(ii - 1, x, u);
        @inbounds r_array[ii] = cu(ii - 1, x, u);
        @inbounds R_array[ii] = cuu(ii - 1, x, u);
        @inbounds P_array[ii] = cux(ii - 1, x, u);
        if isnothing(A_array_input)
            @inbounds A_array[ii] = fx(x, u);
        else
            @inbounds A_array[ii] = A_array_input[ii] # copy(A_array_input[ii])
        end
        if isnothing(B_array_input)
            @inbounds B_array[ii] = fu(x, u);
        else
            @inbounds B_array[ii] = B_array_input[ii] # copy(B_array_input[ii])
        end
        @inbounds W_array[ii] = W(ii - 1);
    end
    @inbounds q_array[N + 1] = h(x_array[N + 1]);
    @inbounds q_vec_array[N + 1] = hx(x_array[N + 1]);
    @inbounds Q_array[N + 1] = hxx(x_array[N + 1]);

    approx_result = ApproximationResult(q_array, q_vec_array, Q_array,
                                        r_array, R_array, P_array,
                                        A_array, B_array, W_array)
    return approx_result
end;


"""
Dynamic Programming Result
"""
struct DynamicProgrammingResult
    s_array::Vector{Float64}
    s_vec_array::Vector{Vector{Float64}}
    S_array::Vector{Matrix{Float64}}
    g_array::Vector{Vector{Float64}}
    G_array::Vector{Matrix{Float64}}
    H_array::Vector{Matrix{Float64}}
end


"""
Compute approximate dynamic programming solution using Riccati(-like) recursion with risk-sensitivity θ and damping μ
"""
function solve_approximate_dp(approx_result::ApproximationResult,
                              l_array::Vector{Vector{Float64}},
                              L_array::Vector{Matrix{Float64}};
                              θ::Float64)# μ::Float64)#::Union{Nothing, DynamicProgrammingResult}
    N = length(approx_result.W_array);
    @assert N == length(l_array) && N == length(L_array)
    s_array = Vector{Float64}(undef, N + 1)
    s_vec_array = Vector{Vector{Float64}}(undef, N + 1)
    S_array = Vector{Matrix{Float64}}(undef, N + 1)
    g_array = Vector{Vector{Float64}}(undef, N)
    G_array = Vector{Matrix{Float64}}(undef, N)
    H_array = Vector{Matrix{Float64}}(undef, N)

    # terminal condition
    s_array[end] = approx_result.q_array[end];
    s_vec_array[end] = approx_result.q_vec_array[end];
    S_array[end] = approx_result.Q_array[end];
    @assert issymmetric(S_array[end]);

    # riccati-like recursion
    for ii in Iterators.reverse(1 : N)
        @inbounds q, q_vec, Q = approx_result.q_array[ii], approx_result.q_vec_array[ii], approx_result.Q_array[ii];
        @inbounds r, R, P = approx_result.r_array[ii], approx_result.R_array[ii], approx_result.P_array[ii];
        @inbounds A, B = approx_result.A_array[ii], approx_result.B_array[ii];
        @inbounds W = approx_result.W_array[ii];
        @inbounds l = l_array[ii];
        @inbounds L = L_array[ii]
        @inbounds M = Symmetric(inv(W) - θ.*S_array[ii + 1]);
        @assert isposdef(M) "M = $(M): (inv(W_{ii}) - θ*S_{ii + 1}) is not PSD at ii = $(ii)"
        @inbounds D = Matrix(1.0I, size(W)) + θ.*S_array[ii + 1]/M
        @inbounds g_array[ii] = r + B'*D*s_vec_array[ii + 1];
        @inbounds G_array[ii] = P + B'*(D*S_array[ii + 1])*A;# + μ.*Matrix(1.0I, size(W)))*A;
        @inbounds H_array[ii] = R + B'*(D*S_array[ii + 1])*B # + μ.*Matrix(1.0I, size(R));# + μ.*Matrix(1.0I, size(W)))*B;
        @inbounds H_array[ii] = Symmetric(H_array[ii])
        #=if require_psd && !isposdef(H_array[ii])
            return nothing
        end=#
        @inbounds s_array[ii] = q + s_array[ii + 1] + 0.5*l'*H_array[ii]*l + l'*g_array[ii];
        if θ == 0.0 # iLQG
            @inbounds s_array[ii] += 0.5*tr(W*S_array[ii + 1])
        else
            @inbounds s_array[ii] += θ/2*s_vec_array[ii + 1]'/M*s_vec_array[ii + 1] - 1/(2*θ)*logdet(W*M);
        end
        @inbounds s_vec_array[ii] = q_vec + A'*D*s_vec_array[ii + 1] + L'*H_array[ii]*l + L'*g_array[ii] + G_array[ii]'*l;
        @inbounds S_array[ii] = Q + A'*D*S_array[ii + 1]*A + L'*H_array[ii]*L + L'*G_array[ii] + G_array[ii]'*L;
        @inbounds S_array[ii] = Symmetric(S_array[ii])
    end
    dp_result = DynamicProgrammingResult(s_array, s_vec_array, S_array,
                                         g_array, G_array, H_array)
    return dp_result
end;


"""
Increase μ and Δ of iLEQG solver
"""
function increase_μ_and_Δ!(ileqg::ILEQGSolver)
    ileqg.Δ = max(ileqg.Δ_0, ileqg.Δ*ileqg.Δ_0);
    ileqg.μ = max(ileqg.μ_min, ileqg.μ*ileqg.Δ);
end;


"""
Decrease μ and Δ of iLEQG solver
"""
function decrease_μ_and_Δ!(ileqg::ILEQGSolver)
    ileqg.Δ = min(1/ileqg.Δ_0, ileqg.Δ/ileqg.Δ_0);
    new_μ_cand = ileqg.μ*ileqg.Δ;
    if new_μ_cand >= ileqg.μ_min;
        ileqg.μ = new_μ_cand;
    else
        ileqg.μ = 0;
    end
end;


"""
Make sure control Hessians are all PSD by regularizing the matrices.
"""
function regularize!(ileqg::ILEQGSolver, dp_result::DynamicProgrammingResult, verbose=true)
    while true
        H_array = copy(dp_result.H_array)
        for ii = 1 : length(H_array)
            H_array[ii] += ileqg.μ*Matrix(1.0I, size(H_array[ii]));
        end
        if all(isposdef.(H_array))
            for ii = 1 : length(H_array)
                dp_result.H_array[ii] = H_array[ii];
            end
            break;
        else
            increase_μ_and_Δ!(ileqg)
            if verbose
                println("------Increasing μ to $(ileqg.μ) and Δ to $(ileqg.Δ)")
            end
        end
    end
    decrease_μ_and_Δ!(ileqg)
    if verbose
        println("------Approximate dynamic programming solved. Decreasing μ to $(ileqg.μ) and Δ to $(ileqg.Δ)")
    end
    return dp_result
end


"""
Perform Backtracking Line Search with Armijo Condition
"""
function line_search!(ileqg::ILEQGSolver, problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                      dp_result::DynamicProgrammingResult, θ::Float64, verbose=true)
    dl_array_new = [-dp_result.H_array[ii]\dp_result.g_array[ii] for ii = 1 : length(dp_result.H_array)]
    L_array_new = [-dp_result.H_array[ii]\dp_result.G_array[ii] for ii = 1 : length(dp_result.H_array)];

    expected_cost_current = dp_result.s_array[1];
    if verbose
        cost_str = @sprintf "%3.3f" expected_cost_current
        println("----Current expected cost: $(cost_str)")
    end
    dir_deriv = 0.0
    for ii = 1 : length(dp_result.H_array)
        dir_deriv += ileqg.l_array[ii]'*dp_result.H_array[ii]*dl_array_new[ii]
                   + dl_array_new[ii]'*dp_result.g_array[ii]
    end
    if verbose
        deriv_str = @sprintf "%3.3f" dir_deriv;
        println("----Directional derivative: $(deriv_str)")
    end
    @assert dir_deriv <= 0.0
    ϵ = ileqg.ϵ_init
    value_new = Inf;
    count = 0;
    while true
        count += 1;
        if verbose
            println("----Performing line search with ϵ == $(ϵ)")
        end
        l_array_new = ileqg.l_array .+ ϵ.*dl_array_new;
        if ileqg.f_returns_jacobian
            x_array_new, u_array_new, A_array_new, B_array_new =
                simulate_dynamics(problem, ileqg.x_array, l_array_new, L_array_new,
                                  f_returns_jacobian=true)
            approx_result_new = approximate_model(problem, u_array_new, x_array_new,
                                                  A_array_new, B_array_new);
        else
            x_array_new, u_array_new = simulate_dynamics(problem, ileqg.x_array,
                                                         l_array_new, L_array_new,
                                                         f_returns_jacobian=false)
            approx_result_new = approximate_model(problem, u_array_new, x_array_new);
        end
        #=dp_result_new = solve_approximate_dp(approx_result_new, u_array_new, L_array_new,
                                             θ=θ, μ=ileqg.μ, require_psd=false);=#
        dp_result_new = solve_approximate_dp(approx_result_new, u_array_new, L_array_new,
                                             θ=θ)#, μ=ileqg.μ)
        expected_cost_new = dp_result_new.s_array[1];
        value_new = expected_cost_new;
        # Armijo condition
        push!(ileqg.ϵ_history, (ϵ, expected_cost_new - (expected_cost_current + ileqg.β*ϵ*dir_deriv)))
        if expected_cost_new <= expected_cost_current + ileqg.β*ϵ*dir_deriv
            ileqg.d_current = maximum(norm.(ileqg.l_array .- u_array_new));
            if verbose
                cost_str = @sprintf "%3.3f" expected_cost_new;
                err_norm_str = @sprintf "%3.3f" ileqg.d_current
                println("------Accepted. New expected cost: $(cost_str). d == $(err_norm_str)")
            end
            ileqg.x_array = x_array_new;
            ileqg.l_array = u_array_new;
            ileqg.L_array = L_array_new;
            if ileqg.f_returns_jacobian
                ileqg.A_array = approx_result_new.A_array
                ileqg.B_array = approx_result_new.B_array
            else
                ileqg.A_array = nothing
                ileqg.B_array = nothing
            end
            break;
        else
            ϵ *= ileqg.λ;
            if ϵ < ileqg.ϵ_min
                ileqg.d_current = maximum(norm.(ileqg.l_array .- u_array_new));
                if verbose
                    cost_str = @sprintf "%3.3f" expected_cost_new;
                    err_norm_str = @sprintf "%3.3f" ileqg.d_current
                    println("------New expected cost: $(cost_str). Terminating as ϵ_min reached. d == $(err_norm_str)")
                end
                ileqg.x_array = x_array_new;
                ileqg.l_array = u_array_new;
                ileqg.L_array = L_array_new;
                if ileqg.f_returns_jacobian
                    ileqg.A_array = approx_result_new.A_array
                    ileqg.B_array = approx_result_new.B_array
                else
                    ileqg.A_array = nothing
                    ileqg.B_array = nothing
                end
                break;
            elseif verbose
                cost_str = @sprintf "%3.3f" expected_cost_new;
                println("------New expected cost: $(cost_str). Decreasing ϵ to $(ϵ)")
            end
        end
    end
    if ileqg.ϵ_init_auto
        if count == 1
            ileqg.ϵ_init = ϵ/ileqg.λ
        else
            while ϵ < ileqg.ϵ_min
                ϵ = ϵ/ileqg.λ;
            end
            ileqg.ϵ_init = ϵ
        end
    end
    return value_new
end


"""
single iteration of iLEQG
"""
function step!(ileqg::ILEQGSolver, problem::FiniteHorizonRiskSensitiveOptimalControlProblem, θ::Float64, verbose=true)
    ileqg.iter_current += 1;
    if verbose
        println("--ILEQG iteration $(ileqg.iter_current)")
        println("----Approximating model around current l_array and x_array")
    end
    approx_result = approximate_model(problem, ileqg.l_array, ileqg.x_array,
                                      ileqg.A_array, ileqg.B_array)

    if verbose
        println("----Solving approximate dynamic programming")
    end
    dp_result = solve_approximate_dp(approx_result, ileqg.l_array, ileqg.L_array,
                                     θ=θ);
    dp_result = regularize!(ileqg, dp_result, verbose);
    value_new = line_search!(ileqg, problem, dp_result, θ, verbose);
    return value_new
end


"""
    solve!(ileqg::ILEQGSolver, problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
    x_0::Vector{Float64}, u_array::Vector{Vector{Float64}}, θ::Float64, verbose=true)

Given `problem`, and `ileqg` solver, solve iLQG (if `θ == 0`) or iLEQG (if `θ > 0`)
with current state `x_0` and nominal control schedule `u_array = [u_0, ..., u_{N-1}]`.

# Return Values (Ordered)
- `x_array::Vector{Vector{Float64}}` -- nominal state trajectory `[x_0,...,x_N]`.
- `l_array::Vector{Vector{Float64}}` -- nominal control schedule `[l_0,...,l_{N-1}]`.
- `L_array::Vector{Matrix{Float64}}` -- feedback gain schedule `[L_0,...,L_{N-1}]`.
- `value::Float64` -- optimal cost-to-go (i.e. value) found by the solver.
- `ϵ_history::Vector{Float64}` -- history of line search step sizes used during
  the iLEQG iteration. Mainly for debugging purposes.

# Notes
- Returns a time-varying affine state-feedback policy `π_k` of the form
  `π_k(x) = L_k(x - x_k) + l_k`.
"""
function solve!(ileqg::ILEQGSolver,
                problem::FiniteHorizonRiskSensitiveOptimalControlProblem,
                x_0::Vector{Float64}, u_array::Vector{Vector{Float64}};
                θ::Float64, verbose=true)
    initialize!(ileqg, problem, x_0, u_array);
    value = Inf;
    while true
        value = step!(ileqg, problem, θ, verbose);
        if ileqg.d > ileqg.d_current && ileqg.μ <= ileqg.μ_min
            if verbose
                err_norm_str = @sprintf "%3.3f" ileqg.d_current
                println("ILEQG Converged. d == $(err_norm_str)")
            end
            break;
        elseif ileqg.iter_current == ileqg.iter_max
            if verbose
                println("Maximum iteration number reached.")
            end
            break;
        end
    end
    x_array, l_array, L_array = copy(ileqg.x_array), copy(ileqg.l_array), copy(ileqg.L_array)
    ϵ_history = copy(ileqg.ϵ_history)
    return x_array, l_array, L_array, value, ϵ_history
end
