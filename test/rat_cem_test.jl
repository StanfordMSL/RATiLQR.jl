#///////////////////////////////////////
#// File Name: rat_cem_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/03/08
#// Description: Test code for src/rat_cem.jl
#///////////////////////////////////////


@everywhere using LinearAlgebra
using Statistics
using Random

@everywhere mutable struct TestModel
    Q::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    q_vec::Vector{Float64}
    r::Vector{Float64}
    q::Float64
    f::Function
    c::Union{Nothing, Function}
    h::Union{Nothing, Function}
    W::Function

    function f(x, u, f_returns_jacobian=false)
        A = [1.2 0.3 -0.5;
             0.2 0.8 -0.2;
             0.1 0.2  1.1];#Matrix(1.0I, length(x), length(x));
        B = [0.8 -0.3;
             0.5 -0.5;
             1.2  0.3];#zeros(length(x), length(u))
        return f_returns_jacobian ? (A*x + B*u, A, B) : A*x + B*u
    end

    function c(model, k, x, u)
        return 0.5*x'*model.Q*x + 0.5*u'*model.R*u + u'*model.P*x +
               model.q_vec'*x + model.r'*u + model.q
    end

    function h(model, x)
        return 0.5*x'*model.Q*x + model.q_vec'*x;
    end

    function W(k)
        return Matrix(0.05I, 3, 3);
    end

    function TestModel(Q, P, R, q_vec, r, q)
        this = new(Q, P, R, q_vec, r, q, f, nothing, nothing, W);
        this.c = (k, x, u) -> c(this, k, x, u);
        this.h = (x) -> h(this, x);
        return this;
    end
end

@testset "RAT CEM Test" begin
    Q_g = [ 1.0 -0.5 1.0;
         -0.5  3.0 0.2;
          1.0  0.2 9.0];
    P_g = [0.4 0.1 0.3;
           0.2 0.3 0.2];
    R_g = [0.5 0.0;
           0.0 0.5];
    q_vec_g = [-1.0, -3.0, -2.0];

    r_g = [-0.4, -0.2];

    q_g = 2.0;

    # get_joint_affine_dynamics test
    begin
          model = TestModel(Q_g, P_g, R_g, q_vec_g, r_g, q_g);
          N = 4;

          problem = FiniteHorizonAdditiveGaussianProblem(model.f, model.c, model.h, model.W, N);

          x_init = [0.1, 0.9, -3.0];
          u_array = [[2.0, -2.0], [1.0, 0.4], [-0.3, 0.5], [1.0, 1.0]];

          x_array, A_array, B_array =
          simulate_dynamics(problem, x_init, u_array, f_returns_jacobian=true);

          approx_result =
          approximate_model(problem, u_array, x_array, A_array, B_array);

          joint_dynamics_model = get_joint_affine_dynamics(approx_result, x_array, u_array);

          x_vec_test = joint_dynamics_model.E*vcat(u_array...) + joint_dynamics_model.g;
          @test all(x_vec_test .≈ vcat(x_array...));

          w_array = [[3.0, 1.0, -4.0], [0.3, 1.3, 0.5], [-0.5, -0.5, 0.3], [1.0, 1.1, -1.0]]
          x_array_noisy = [x_init];
          for ii = 1 : length(u_array)
              push!(x_array_noisy, model.f(x_array_noisy[end], u_array[ii]) + w_array[ii])
          end

          approx_result =
          approximate_model(problem, u_array, x_array, A_array, B_array);

          joint_dynamics_model = get_joint_affine_dynamics(approx_result, x_array, u_array);

          x_vec_test = joint_dynamics_model.E*vcat(u_array...) +
                 joint_dynamics_model.F*vcat(w_array...) + joint_dynamics_model.g;
          @test all(x_vec_test .≈ vcat(x_array_noisy...))
    end

    # get_joint_quadratic_cost test
    begin
          model = TestModel(Q_g, P_g, R_g, q_vec_g, r_g, q_g);
          N = 1;

          problem = FiniteHorizonAdditiveGaussianProblem(model.f, model.c, model.h, model.W, N);

          x_init = [0.1, 0.9, -3.0];
          u_array = [[2.0, -2.0]];

          x_array, A_array, B_array =
          simulate_dynamics(problem, x_init, u_array, f_returns_jacobian=true);

          approx_result =
          approximate_model(problem, u_array, x_array, A_array, B_array);

          joint_cost_model = get_joint_quadratic_cost(problem, approx_result, x_array, u_array);

          x_joint_test = [3.0, 2.0, 0.1, -0.3, -0.5, 1.0];
          u_joint_test = [-3.0, 2.0];

          cost_true = model.c(0, x_joint_test[1:3], u_joint_test) + model.h(x_joint_test[4:6])

          cost_test_1 = 0.5*(x_joint_test - vcat(x_array...))'*joint_cost_model.Q*(x_joint_test - vcat(x_array...)) +
          0.5*(u_joint_test - vcat(u_array...))'*joint_cost_model.R*(u_joint_test - vcat(u_array...)) +
          (u_joint_test - vcat(u_array...))'*joint_cost_model.P*(x_joint_test - vcat(x_array...)) +
          (x_joint_test - vcat(x_array...))'*vcat(approx_result.q_vec_array...) +
          (u_joint_test - vcat(u_array...))'*vcat(approx_result.r_array...) +
          sum(approx_result.q_array)
          @test cost_test_1 ≈ cost_true
          @test all(Matrix(joint_cost_model.Q) .≈ [model.Q zeros(3, 3);
                                                   zeros(3, 3) model.Q])
          @test all(Matrix(joint_cost_model.R) .≈ model.R)
          @test all(Matrix(joint_cost_model.P) .≈ [model.P zeros(2, 3)])
          @test Matrix(joint_cost_model.W) == Matrix(0.05I, 3, 3)
          @test all(joint_cost_model.q_vec .≈ vcat([q_vec_g; q_vec_g]))
          @test all(joint_cost_model.r .≈ r_g)
          @test joint_cost_model.q .≈ q_g

          cost_test_2 =
          0.5*x_joint_test'*joint_cost_model.Q*x_joint_test +
          0.5*u_joint_test'*joint_cost_model.R*u_joint_test +
          u_joint_test'*joint_cost_model.P*x_joint_test +
          x_joint_test'*joint_cost_model.q_vec +
          u_joint_test'*joint_cost_model.r +
          joint_cost_model.q
          @test cost_test_2 ≈ cost_true
    end

    # get_open_loop_value test
    begin
          model = TestModel(Q_g, P_g, R_g, q_vec_g, r_g, q_g);
          N = 4;

          problem = FiniteHorizonAdditiveGaussianProblem(model.f, model.c, model.h, model.W, N);

          x_init = [0.1, 0.9, -3.0];
          u_array = [[2.0, -2.0], [1.0, 0.4], [-0.3, 0.5], [1.0, 1.0]];

          value = get_open_loop_value(problem, x_init, u_array, f_returns_jacobian=true);
          #@benchmark get_open_loop_value(problem, x_init, u_array, f_returns_jacobian=true)
          value2 = get_open_loop_value(problem, x_init, u_array, f_returns_jacobian=true, θ=1e-9);
          @test isapprox(value, value2, atol=1e-4);

          #=rng = MersenneTwister(1234567);
          M = 5000000;
          sampled_cost_array = zeros(M);

          @showprogress for ii = 1 : M
              sampled_cost_array[ii] =
              integrate_cost(problem,
                             simulate_dynamics(problem, x_init, u_array, rng),
                             u_array);
          end
          println(mean(sampled_cost_array));=#
    end

    # RATCEMSolver test
    model = TestModel(Q_g, P_g, R_g, q_vec_g, r_g, q_g);
    N = 4;

    x_0 = zeros(3)

    problem = FiniteHorizonAdditiveGaussianProblem(model.f, model.c, model.h, model.W, N)

    μ_init_array = [0.1*ones(2) for ii = 1 : N];
    Σ_init_array = [diagm([0.5, 0.5]) for ii = 1 : N];

    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=3, μ_θ_init=1.0, σ_θ_init=2.0, λ_θ=0.5,
                          num_risk_samples=3, num_elite=3, iter_max=3,
                          smoothing_factor=0.1, u_mean_carry_over=false,
                          f_returns_jacobian=false);
    @test solver.N == N
    @test solver.iter_current == 0;
    @test solver.μ_array == μ_init_array
    @test solver.Σ_array == Σ_init_array
    @test solver.μ_θ == solver.μ_θ_init
    @test solver.σ_θ == solver.σ_θ_init

    solver.iter_current = 10;
    solver.μ_array = [ones(2) for ii = 1 : N];
    solver.Σ_array = [Matrix(0.1I, 2, 2) for ii = 1 : N];
    solver.μ_θ = 0.01;
    solver.σ_θ = 0.1;
    initialize!(solver)
    @test solver.iter_current == 0;
    @test solver.μ_array == μ_init_array
    @test solver.Σ_array == Σ_init_array
    @test solver.μ_θ == solver.μ_θ_init
    @test solver.σ_θ == solver.σ_θ_init

    # compute_cost test
    control_sequence_array =
       [[[0.1, -2.0], [1.3, 1.0], [-0.2, 0.4], [0.2, 0.3]],
        [[-1.2, 0.2], [0.5, 0.4], [-0.8, 0.1], [0.1, 0.3]],
        [[0.5,  0.8], [0.1, 0.1], [0.3, -1.2], [0.3, -0.9]]];
    θ_array = [0.05, 0.1, 0.2];
    kl_bound = 1.0;
    cost_matrix = compute_cost(solver, problem, x_0, control_sequence_array, θ_array, kl_bound);
    cost_matrix_test = compute_cost_serial(solver, problem, x_0, control_sequence_array, θ_array, kl_bound)
    @test all(cost_matrix .≈ cost_matrix_test)

    # get_elite_samples test
    u_elite_idx_array, θ_elite_idx_array = get_elite_samples(solver, cost_matrix)
    @test Set(u_elite_idx_array) == Set([1, 3, 3]);
    @test Set(θ_elite_idx_array) == Set([1, 1, 2]);

    # compute_new_distribution test
    μ_new_array, Σ_new_array, μ_θ_new, σ_θ_new =
    compute_new_distribution(solver, control_sequence_array[u_elite_idx_array],
                             θ_array[θ_elite_idx_array]);

    mean_array = mean.([[u[ii] for u in control_sequence_array[u_elite_idx_array]] for ii = 1 : 4])
    cov_array = diagm.(var.([[u[ii] for u in control_sequence_array[u_elite_idx_array]] for ii = 1 : 4]))
    @test all(μ_new_array .≈ 0.9.*mean_array .+ 0.1.*solver.μ_array)
    @test all(Σ_new_array .≈ 0.9.*cov_array .+ 0.1.*solver.Σ_array)
    @test μ_θ_new ≈ mean(θ_array[θ_elite_idx_array])
    @test σ_θ_new ≈ std(θ_array[θ_elite_idx_array], corrected=false)

    μ_new_array_2, Σ_new_array_2, μ_θ_new_2, σ_θ_new_2 =
       compute_new_distribution(solver, control_sequence_array[[1, 1, 1]],
                                θ_array[[1, 1, 1]]);

    @test all(μ_new_array_2 .≈ 0.9.*control_sequence_array[1] .+ 0.1.*solver.μ_array)
    @test all(Σ_new_array_2 .≈ 0.9.*diagm.([1e-6*ones(2) for ii = 1 : 4]) + 0.1.*solver.Σ_array)
    @test μ_θ_new_2 ≈ θ_array[1]
    @test σ_θ_new_2 ≈ 1e-3;

    # step! test
    initialize!(solver);
    rng = MersenneTwister(1234)
    step!(solver, problem, x_0, kl_bound, rng, false, false)
    @test !(solver.μ_θ_init ≈ 1.0)
    @test !(solver.σ_θ_init ≈ 2.0)
    @test solver.μ_θ > 0.0
    @test solver.σ_θ > 0.0
    @test solver.iter_current == 1

    # solve! test
    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=50, μ_θ_init=1.0, σ_θ_init=2.0, λ_θ=0.5,
                          num_risk_samples=10, num_elite=30, iter_max=3,
                          smoothing_factor=0.1, u_mean_carry_over=false,
                          f_returns_jacobian=false);
    rng = MersenneTwister(1234)
    μ_array, Σ_array, μ_θ, σ_θ, θ_min, θ_max =
    solve!(solver, problem, x_0, rng, kl_bound=kl_bound, verbose=false, serial=true);
    @test solver.iter_current == solver.iter_max

    # kl_bound == 0.0 test
    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=50, μ_θ_init=1.0, σ_θ_init=2.0, λ_θ=0.5,
                          num_risk_samples=1, num_elite=30, iter_max=3,
                          smoothing_factor=0.1, u_mean_carry_over=false,
                          f_returns_jacobian=false);
                          rng = MersenneTwister(1234)
    μ_array, Σ_array_2, μ_θ, σ_θ, θ_min, θ_max =
      solve!(solver, problem, x_0, rng, kl_bound=0.0, verbose=false, serial=true);
    @test μ_θ == 0.0
    @test σ_θ == 0.0
    @test θ_min == 0.0
    @test θ_max == 0.0

    # u_mean_carray_over test
    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=50, μ_θ_init=1.0, σ_θ_init=2.0, λ_θ=0.5,
                          num_risk_samples=10, num_elite=30, iter_max=3,
                          smoothing_factor=0.1, u_mean_carry_over=true,
                          f_returns_jacobian=false);

    rng = MersenneTwister(1234)
    μ_array, Σ_array_3, μ_θ, σ_θ, θ_min, θ_max = 
    solve!(solver, problem, x_0, rng, kl_bound=kl_bound, verbose=false, serial=true);
    @test solver.μ_init_array[1:end-1] == μ_array[2:end]
    @test solver.μ_init_array[end] == [0.1, 0.1];
    @test Σ_array_3 == Σ_array
end
