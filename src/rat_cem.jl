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
using SparseArrays

"""
    RATCEMSolver(μ_init_array::Vector{Vector{Float64}},
    Σ_init_array::Vector{Matrix{Float64}}; kwargs...)

RAT CEM Solver initialized with `μ_init_array = [μ_0,...,μ_{N-1}]` and
`Σ_init_array = [Σ_0,...,Σ_{N-1}]`, where the initial control distribution at time
`k` is a Gaussian distribution `Distributions.MvNormal(μ_k, Σ_k)`.

# Optional Keyword Arguments
- `num_control_samples::Int64` -- number of Monte Carlo samples for the control
  trajectory. Default: `10`.
- `μ_θ_init::Float64` -- initial mean parameter `μ_θ` for the risk-sensitivity.
  Default: `1.0`.
- `σ_θ_init::Float64` -- initial covariance parameter `σ_θ` for the risk-sensitivity.
  Default: `2.0`.
- `λ_θ::Float64` -- multiplicative modification factor in (0, 1) for ``μ_θ_init` and
  `σ_θ_init`. Default: `0.5`.
- `num_risk_samples::Int64` -- number of Monte Carlo samples for the risk-sensitivity.
  Default: `10`.
- `num_elite::Int64` -- number of elite samples. Default: `10`.
- `iter_max::Int64` -- maximum iteration number. Default: `5`.
- `smoothing_factor::Float64` -- smoothing factor in (0, 1), used to update the
  mean and the variance of the Cross Entropy distribution for the next iteration.
  If `smoothing_factor` is `0.0`, the updated distribution is independent of the
  previous iteration. If it is `1.0`, the updated distribution is the same as the
  previous iteration. Default: `0.1`.
- `u_mean_carry_over::Bool` -- save `μ_array` of the last iteration and use it to
  initialize `μ_array` in the next call to `solve!`. Default: `false`.
- `f_returns_jacobian::Bool` -- if `true`, Jacobian matrices of the dynamics function
  are user-provided. This can reduce computation time since automatic
  differentiation is not used. Default: `false`.
"""
mutable struct RATCEMSolver
    # CE solver parameters
    num_control_samples::Int64
    num_risk_samples::Int64
    num_elite::Int64
    iter_max::Int64
    smoothing_factor::Float64
    u_mean_carry_over::Bool
    f_returns_jacobian::Bool       # Whether the dynamics funcion f also returns jacobians or not

    # action distributions
    μ_init_array::Vector{Vector{Float64}}
    Σ_init_array::Vector{Matrix{Float64}}
    μ_array::Vector{Vector{Float64}}
    Σ_array::Vector{Matrix{Float64}}
    # risk_param distributions
    μ_θ_init::Float64
    σ_θ_init::Float64
    λ_θ::Float64                     # Multiplicative factor for μ_θ and σ_θ in (0, 1)
    μ_θ::Float64
    σ_θ::Float64
    θ_max::Float64                 # Maximum valid θ encountered so far
    θ_min::Float64                 # Minimum valid θ encountered so far
    N::Int64 # control sequence length > 0 (must be the same as N in FiniteHorizonGenerativeProblem)
    iter_current::Int64
end;

function RATCEMSolver(μ_init_array::Vector{Vector{Float64}},
                      Σ_init_array::Vector{Matrix{Float64}};
                      num_control_samples=10,
                      μ_θ_init=1.0,
                      σ_θ_init=2.0,
                      λ_θ=0.5,
                      num_risk_samples=10,
                      num_elite=10,
                      iter_max=5,
                      smoothing_factor=0.1,
                      u_mean_carry_over=false,
                      f_returns_jacobian=false)

    @assert length(μ_init_array) == length(Σ_init_array);
    μ_array, Σ_array = copy(μ_init_array), copy(Σ_init_array);
    μ_θ, σ_θ = μ_θ_init, σ_θ_init;
    θ_max, θ_min = 0.0, Inf
    N = length(μ_init_array);
    iter_current = 0;

    return RATCEMSolver(num_control_samples, num_risk_samples,
                        num_elite, iter_max, smoothing_factor,
                        u_mean_carry_over, f_returns_jacobian,
                        copy(μ_init_array), copy(Σ_init_array),
                        μ_array, Σ_array,
                        μ_θ_init, σ_θ_init, λ_θ, μ_θ, σ_θ, θ_max, θ_min,
                        N, iter_current)
end;

function initialize!(solver::RATCEMSolver)
    solver.iter_current = 0;
    solver.μ_array = copy(solver.μ_init_array);
    solver.Σ_array = copy(solver.Σ_init_array);
    solver.μ_θ = solver.μ_θ_init;
    solver.σ_θ = solver.σ_θ_init;
    solver.θ_max = 0.0;
    solver.θ_min = Inf;
end

struct JointAffineDynamicsModel
    E::AbstractMatrix
    F::AbstractMatrix
    g::Vector{Float64}
end

function get_joint_affine_dynamics(approx_result::ApproximationResult,
                                   x_array::Vector{Vector{Float64}},
                                   u_array::Vector{Vector{Float64}};
                                   return_sparse::Bool=false)
    n, m = length(x_array[1]), length(u_array[1]);
    @assert all(n .== length.(x_array)) && all(m .== length.(u_array));
    N = length(u_array);

    E = zeros(n*(N + 1), m*N);
    F = zeros(n*(N + 1), n*N);
    g = zeros(n*(N + 1));
    v = zeros(n*N);

    g[1 : n] = copy(x_array[1]);
    for row = 2 : N + 1
        g[n*(row - 1) + 1 : n*row] = approx_result.A_array[row - 1]*g[n*(row - 2) + 1 : n*(row - 1)];
    end

    for row = 1 : N
        v[n*(row - 1) + 1 : n*row] = x_array[row + 1] +
                                     -approx_result.A_array[row]*x_array[row] +
                                     -approx_result.B_array[row]*u_array[row];
    end

    for col = 1 : N
        row = col + 1;
        E[n*(row - 1) + 1 : n*row, m*(col - 1) + 1 : m*col] = approx_result.B_array[col];
        F[n*(row - 1) + 1 : n*row, n*(col - 1) + 1 : n*col] = Matrix(1.0I, n, n);
        for row = col + 2 : N + 1
            E[n*(row - 1) + 1 : n*row, m*(col - 1) + 1 : m*col] =
                approx_result.A_array[row - 1]*E[n*(row - 2) + 1 : n*(row - 1), m*(col - 1) + 1 : m*col];
            F[n*(row - 1) + 1 : n*row, n*(col - 1) + 1 : n*col] =
                approx_result.A_array[row - 1]*F[n*(row - 2) + 1 : n*(row - 1), n*(col - 1) + 1 : n*col];
        end
    end

    g += F*v;

    if return_sparse
        E = sparse(E);
        F = sparse(F);
    end

    return JointAffineDynamicsModel(E, F, g)
end

struct JointQuadraticCostModel
    Q::AbstractMatrix
    R::AbstractMatrix
    P::AbstractMatrix
    W::AbstractMatrix
    q_vec::Vector{Float64}
    r::Vector{Float64}
    q::Float64
end

function get_joint_quadratic_cost(problem::FiniteHorizonAdditiveGaussianProblem,
                                  approx_result::ApproximationResult,
                                  x_array::Vector{Vector{Float64}},
                                  u_array::Vector{Vector{Float64}};
                                  return_sparse::Bool=true)
    q = 0.5*sum(dot.(x_array, approx_result.Q_array.*x_array)) +
        0.5*sum(dot.(u_array, approx_result.R_array.*u_array)) +
        sum(dot.(u_array, approx_result.P_array.*x_array[1:end-1])) +
        -sum(dot.(approx_result.q_vec_array, x_array)) +
        -sum(dot.(approx_result.r_array, u_array)) +
        sum(approx_result.q_array);

    n, m = length(x_array[1]), length(u_array[1]);
    N = length(u_array);
    @assert all(n .== length.(x_array)) && all(m .== length.(u_array));

    Q = spzeros(n*(N + 1), n*(N + 1));
    R = spzeros(m*N, m*N);
    P = spzeros(m*N, n*(N + 1));
    W = spzeros(n*N, n*N);
    q_vec = zeros(n*(N + 1));
    r = zeros(m*N);

    for ii = 1 : N
        Q[n*(ii - 1) + 1 : n*ii, n*(ii - 1) + 1 : n*ii] = approx_result.Q_array[ii];
        R[m*(ii - 1) + 1 : m*ii, m*(ii - 1) + 1 : m*ii] = approx_result.R_array[ii];
        P[m*(ii - 1) + 1 : m*ii, n*(ii - 1) + 1 : n*ii] = approx_result.P_array[ii];
        W[n*(ii - 1) + 1 : n*ii, n*(ii - 1) + 1 : n*ii] = problem.W(ii - 1);
        q_vec[n*(ii - 1) + 1 : n*ii] =
            approx_result.q_vec_array[ii] +
            -approx_result.Q_array[ii]*x_array[ii] +
            -approx_result.P_array[ii]'*u_array[ii];
        r[m*(ii - 1) + 1 : m*ii] =
            approx_result.r_array[ii] +
            -approx_result.P_array[ii]*x_array[ii] +
            -approx_result.R_array[ii]*u_array[ii];
    end
    Q[n*N + 1 : n*(N + 1), n*N + 1 : n*(N + 1)] = approx_result.Q_array[N + 1];
    q_vec[n*N + 1 : n*(N + 1)] =
        approx_result.q_vec_array[N + 1] +
        -approx_result.Q_array[N + 1]*x_array[N + 1];

    return JointQuadraticCostModel(Q, R, P, W, q_vec, r, q)
end

function get_open_loop_value(problem::FiniteHorizonAdditiveGaussianProblem,
                             dynamics::JointAffineDynamicsModel,
                             cost::JointQuadraticCostModel,
                             u_array::Vector{Vector{Float64}})

    u = vcat(u_array...);
    E, F, g = dynamics.E, dynamics.F, dynamics.g;
    Q, R, P, W = cost.Q, cost.R, cost.P, cost.W;
    q_vec, r, q = cost.q_vec, cost.r, cost.q;

    value = 0.5*u'*(E'*Q*E + 2*P*E + R)*u + u'*((E'*Q + P)*g + E'q_vec + r) +
            0.5*tr(F'*Q*F*W) + 0.5*g'*Q*g + q_vec'*g + q;

    return value
end

function get_open_loop_value(problem::FiniteHorizonAdditiveGaussianProblem,
                             dynamics::JointAffineDynamicsModel,
                             cost::JointQuadraticCostModel,
                             u_array::Vector{Vector{Float64}},
                             θ::Float64)
    if θ == 0.0
        return get_open_loop_value(problem, dynamics, cost, u_array);
    end

    u = vcat(u_array...);
    E, F, g = dynamics.E, dynamics.F, dynamics.g;
    Q, R, P, W = cost.Q, cost.R, cost.P, cost.W;
    q_vec, r, q = cost.q_vec, cost.r, cost.q;

    N = problem.N;
    n = Int64(size(W, 1) // N);
    Σx_inv = inv((F*W*F')[n + 1 : end, n + 1 : end]);
    A = (F[n + 1 : end, :])'*Σx_inv*F[n + 1 : end, :] - θ*F'*Q*F;
    @assert size(A) == (n*N, n*N)
    try
        @assert isposdef(Symmetric(A)) #"F'*(inv(F*W*F') - θ*Q)*F is not PSD"
    catch
        return Inf
    end

    value = 0.5*u'*(E'*Q*E + 2*P*E + R)*u + u'*((E'*Q + P)*g + E'q_vec + r) +
            0.5*g'*Q*g + q_vec'*g + q;

    value += -0.5/θ*logdet(Matrix(1.0I, n*N, n*N) - θ*(Q*F*W*F')[n + 1 : end, n + 1 : end]);
    a = ((Q*E + P')*u + Q*g + q_vec);
    value += 0.5*θ*a'*F/A*F'*a;

    return value;
end

function get_open_loop_value(problem::FiniteHorizonAdditiveGaussianProblem,
                             x_0::Vector{Float64}, u_array::Vector{Vector{Float64}};
                             f_returns_jacobian=false, θ=0.0)
    if f_returns_jacobian
        x_array, A_array, B_array =
            simulate_dynamics(problem, x_0, u_array, f_returns_jacobian=true);
    else
        x_array =
            simulate_dynamics(problem, x_0, u_array, f_returns_jacobian=false);
        A_array, B_array = nothing, nothing;
    end
    approx_result = approximate_model(problem, u_array, x_array, A_array, B_array);

    joint_dynamics_model = get_joint_affine_dynamics(approx_result, x_array, u_array);
    joint_cost_model = get_joint_quadratic_cost(problem, approx_result, x_array, u_array);

    value = get_open_loop_value(problem, joint_dynamics_model, joint_cost_model, u_array, θ);

    return value;
end

function compute_value_worker(solver::RATCEMSolver,
                              problem::FiniteHorizonAdditiveGaussianProblem,
                              x::Vector{Float64}, #initial state
                              u_array::Vector{Vector{Float64}}, θ::Float64)
    value = get_open_loop_value(problem, x, u_array,
                                f_returns_jacobian=solver.f_returns_jacobian,
                                θ=θ);
     return value
end

function compute_cost(solver::RATCEMSolver,
                      problem::FiniteHorizonAdditiveGaussianProblem,
                      x::Vector{Float64},
                      control_sequence_array::Vector{Vector{Vector{Float64}}},
                      θ_array::Vector{Float64},
                      kl_bound::Float64)
    @assert length(control_sequence_array) == solver.num_control_samples;
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);
    @assert length(θ_array) == solver.num_risk_samples

    num_control_samples = solver.num_control_samples;
    num_risk_samples = solver.num_risk_samples;
    num_samples = num_control_samples*num_risk_samples;
    if nprocs() > 1
        proc_id_array = 2 .+ [mod(ii, nprocs() - 1) for ii = 0 : num_samples - 1]
    else
        proc_id_array = [1 for ii = 0 : num_samples - 1]
    end
    proc_id_matrix = reshape(proc_id_array, num_control_samples, num_risk_samples);
    value_matrix = Matrix{Float64}(undef, num_control_samples, num_risk_samples);
    @sync begin
        for row = 1 : num_control_samples, col = 1 : num_risk_samples
            @inbounds @async value_matrix[row, col] =
                remotecall_fetch(compute_value_worker, proc_id_matrix[row, col],
                                 solver, problem, x, control_sequence_array[row],
                                 θ_array[col])
        end
    end
    if kl_bound > 0.0
        cost_matrix = value_matrix .+ kl_bound./reshape(θ_array, 1, num_risk_samples);
    else
        cost_matrix = value_matrix
    end
    return cost_matrix
end

function compute_cost_serial(solver::RATCEMSolver,
                             problem::FiniteHorizonAdditiveGaussianProblem,
                             x::Vector{Float64}, #initial state
                             control_sequence_array::Vector{Vector{Vector{Float64}}},
                             θ_array::Vector{Float64},
                             kl_bound::Float64)
    @assert length(control_sequence_array) == solver.num_control_samples;
    @assert all([length(sequence) == solver.N for sequence in control_sequence_array]);
    @assert length(θ_array) == solver.num_risk_samples

    value_matrix = Matrix{Float64}(undef, solver.num_control_samples, solver.num_risk_samples);
    for row = 1 : solver.num_control_samples # for-loop over action sequences
        for col = 1 : solver.num_risk_samples # for-loop over risks
            try
                value_matrix[row, col] =
                    get_open_loop_value(problem, x, control_sequence_array[row],
                                        f_returns_jacobian=solver.f_returns_jacobian,
                                        θ=θ_array[col]);
            catch
                value_matrix[row, col] = Inf;
            end
        end
    end
    if kl_bound > 0.0
        cost_matrix = value_matrix .+ kl_bound./reshape(θ_array, 1, solver.num_risk_samples);
    else
        cost_matrix = value_matrix
    end
    return cost_matrix
end

function get_elite_samples(solver::RATCEMSolver,
                           cost_matrix::Matrix{Float64})
    @assert size(cost_matrix, 1) == solver.num_control_samples
    @assert size(cost_matrix, 2) == solver.num_risk_samples

    pq = PriorityQueue{Tuple{Int64, Int64}, Float64}(Base.Order.Reverse);
    u_elite_idx_array = Vector{Int64}();
    θ_elite_idx_array = Vector{Int64}();
    for u_idx = 1 : size(cost_matrix, 1)
        for θ_idx = 1 : size(cost_matrix, 2)
            if length(pq) < solver.num_elite
                enqueue!(pq, (u_idx, θ_idx) => cost_matrix[u_idx, θ_idx])
            else
                if cost_matrix[u_idx, θ_idx] < DataStructures.peek(pq)[2]
                    dequeue!(pq)
                    enqueue!(pq, (u_idx, θ_idx) => cost_matrix[u_idx, θ_idx])
                end
            end
        end
    end
    while !isempty(pq)
        u_idx, θ_idx = DataStructures.peek(pq)[1];
        push!(u_elite_idx_array, u_idx)
        push!(θ_elite_idx_array, θ_idx)
        dequeue!(pq);
    end

    return u_elite_idx_array, θ_elite_idx_array
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
        if solver.num_risk_samples > 1 && all([isapprox(elem, 0.0, atol=1e-6) for elem in var_tt])
            # if all elite samples are the same control sequence despite sampling multiple risks, the variance becomes 0.
            # To prevent the Gaussian from collapsing, add small positive term to var.
            var_tt += 1e-6*ones(size(var_tt))
        end
        cov_array[tt] = Diagonal(var_tt);

        μ_new_array[tt] = (1.0 - solver.smoothing_factor).*mean_array[tt] + solver.smoothing_factor.*solver.μ_array[tt];
        Σ_new_array[tt] = (1.0 - solver.smoothing_factor).*cov_array[tt] + solver.smoothing_factor.*solver.Σ_array[tt];
    end

    μ_θ_new = sum(θ_elite_array)/solver.num_elite;
    σ_θ_new = sqrt(sum((θ_elite_array .- μ_θ_new).^2)/solver.num_elite)
    if solver.num_risk_samples > 1 && isapprox(σ_θ_new, 0.0, atol=1e-3)
        # if all elite samples are the same risk value despite sampling multiple risks, the std becomes 0.
        # To prevent the Gaussian rom collapsing, add small positive term to std.
        σ_θ_new += 1e-3
    end

    return μ_new_array, Σ_new_array, μ_θ_new, σ_θ_new
end

function step!(solver::RATCEMSolver,
               problem::FiniteHorizonAdditiveGaussianProblem,
               x::Vector{Float64},
               kl_bound::Float64,
               rng::AbstractRNG,
               verbose=true, serial=false)
    solver.iter_current += 1;
    if verbose
        println("**CE iteration $(solver.iter_current)")
    end
    control_sequence_array = Vector{Vector{Vector{Float64}}}(undef, solver.num_control_samples);
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
    if kl_bound == 0.0
        θ_array = [0.0];
        cost_matrix = serial ?
                      compute_cost_serial(solver, problem, x, control_sequence_array, θ_array, kl_bound) :
                      compute_cost(solver, problem, x, control_sequence_array, θ_array, kl_bound);
    else
        while true
            if solver.iter_current == 1
                # draw from N(μ_init, σ_init)
                # if too few valid samples, then adjust μ_init, σ_init and redraw.
                # if all samples are valid, then increase μ_init, σ_init for the next iteration.
                if verbose
                    println("****Drawing $(solver.num_risk_samples) positive samples of θ ~ N($(round(solver.μ_θ_init,digits=4)), $(round(solver.σ_θ_init,digits=4)))");
                end
                θ_array = get_positive_samples(solver.μ_θ_init, solver.σ_θ_init, solver.num_risk_samples, rng);
            else
                if verbose
                    println("****Drawing $(solver.num_risk_samples) positive samples of θ ~ N($(round(solver.μ_θ,digits=4)), $(round(solver.σ_θ,digits=4)))");
                end
                θ_array = get_positive_samples(solver.μ_θ, solver.σ_θ, solver.num_risk_samples, rng);
            end
            # objective computation & elite selection
            if verbose
                println("****Evaluating costs of sampled control sequences & risks")
            end
            if !serial
                cost_matrix = compute_cost(solver, problem, x, control_sequence_array, θ_array, kl_bound);
            else
                cost_matrix = compute_cost_serial(solver, problem, x, control_sequence_array, θ_array, kl_bound);
            end
            if verbose
                println(cost_matrix)
            end
            num_inf = sum(isinf.(cost_matrix));
            num_valid = length(cost_matrix) - num_inf;
            if solver.iter_current == 1 && num_valid < max(solver.num_elite, length(cost_matrix)*solver.λ_θ)
                if verbose
                    println("******$(num_inf)/$(length(cost_matrix)) samples are Inf. Redrawing samples")
                end
                solver.μ_θ_init *= solver.λ_θ
                solver.σ_θ_init *= solver.λ_θ
            elseif solver.iter_current == 1 && num_valid == length(cost_matrix)
                solver.μ_θ_init /= solver.λ_θ
                solver.σ_θ_init /= solver.λ_θ
                if verbose
                    println("******Increasing μ_init to $(solver.μ_θ_init) and σ_init to $(solver.σ_θ_init)")
                end
                break
            elseif num_valid >= max(solver.num_elite, length(cost_matrix)*solver.λ_θ)
                if verbose
                    println("******$(num_valid)/$(length(cost_matrix)) samples are valid")
                end
                break
            end
        end
    end

    for ii = 1 : length(θ_array)
        if all(isinf.(cost_matrix[:, ii]))
            continue
        else
            if θ_array[ii] < solver.θ_min
                solver.θ_min = θ_array[ii]
            end
            if θ_array[ii] > solver.θ_max
                solver.θ_max = θ_array[ii]
            end
        end
    end

    control_sequence_elite_idx_array, θ_elite_idx_array =
        get_elite_samples(solver, cost_matrix);
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
                problem::FiniteHorizonAdditiveGaussianProblem,
                x_0::Vector{Float64},
                rng::AbstractRNG;
                kl_bound::Float64, verbose=true, serial=true)
    @assert kl_bound >= 0 "KL Divergence Bound must be non-negative"
    if kl_bound == 0.0
        @assert solver.num_risk_samples == 1 "num_risk_samples must be 1";
    end
    initialize!(solver);
    while solver.iter_current < solver.iter_max
        step!(solver, problem, x_0, kl_bound, rng, verbose, serial)
    end
    if solver.u_mean_carry_over
        # time indices are shifted, assuming that the algorithm runs in a receding-horizon fashion.
        solver.μ_init_array[1:end-1] = copy(solver.μ_array[2:end])
    end
    return copy(solver.μ_array), copy(solver.Σ_array), solver.μ_θ, solver.σ_θ, solver.θ_min, solver.θ_max;
end
