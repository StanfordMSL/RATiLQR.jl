#///////////////////////////////////////
#// File Name: rat_cem_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/03/08
#// Description: Test code for src/rat_cem.jl
#///////////////////////////////////////


using LinearAlgebra
using Statistics
using Random

@testset "RAT CEM Test" begin

    @everywhere f_stochastic(x, u, rng, deterministic=false) = x + u + rand(rng, length(x))
    @everywhere c(k, x, u) = sum(abs.(u));
    @everywhere h(x) = 1.0;
    N = 20;

    problem = FiniteHorizonGenerativeProblem(f_stochastic, c, h, N);

    # RATCEMSolver test
    μ_init_array = [zeros(2) for ii = 1 : N];
    Σ_init_array = [Matrix(1.0I, 2, 2) for ii = 1 : N];
    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=100, deterministic_dynamics=false,
                          num_trajectory_samples=20,
                          μ_θ_init=1.0, σ_θ_init=2.0, num_risk_samples=20,
                          num_elite=50, iter_max=5, smoothing_factor=0.1,
                          mean_carry_over=false);
    @test solver.N == N
    @test solver.iter_current == 0;
    @test solver.μ_array == μ_init_array
    @test solver.Σ_array == Σ_init_array
    @test solver.μ_θ == 1.0
    @test solver.σ_θ == 2.0

    solver.iter_current = 10;
    solver.μ_array = [ones(2) for ii = 1 : N];
    solver.Σ_array = [Matrix(0.1I, 2, 2) for ii = 1 : N];
    solver.μ_θ = 3.0
    solver.σ_θ = 3.0
    initialize!(solver)

    @test solver.iter_current == 0;
    @test solver.μ_array == μ_init_array
    @test solver.Σ_array == Σ_init_array
    @test solver.μ_θ == 1.0
    @test solver.σ_θ == 2.0

    # compute_cost_serial test
    rng = MersenneTwister(1234);
    control_sequence_array = [[rand(rng, 2) for tt = 1 : N] for
                              ii = 1 : solver.num_control_samples];
    x_init = zeros(2);
    sampled_cost_arrays_serial =
        compute_cost_serial(solver, problem, x_init, control_sequence_array, rng);
    @test length(sampled_cost_arrays_serial) == solver.num_control_samples
    @test all([length(cost_array) == solver.num_trajectory_samples for
               cost_array in sampled_cost_arrays_serial])
    # compute_cost test
    sampled_cost_arrays = compute_cost(solver, problem, x_init, control_sequence_array, rng);
    @test all(sampled_cost_arrays .== sampled_cost_arrays_serial) # note that this holds because c is not dependent on x. Otherwise they'd be different.

    rng = MersenneTwister(1234);
    @test length(sampled_cost_arrays) == solver.num_control_samples
    @test all([length(cost_array) == solver.num_trajectory_samples for cost_array in sampled_cost_arrays])
    for ii = 1 : solver.num_control_samples
        x = x_init;
        cost = 0.0;
        for tt = 1 : solver.N
            cost += problem.c(tt - 1, x, control_sequence_array[ii][tt])
            x = problem.f_stochastic(x, control_sequence_array[ii][tt], rng)
        end
        cost += problem.h(x)
        @test all([cost ≈ sampled_cost for sampled_cost in sampled_cost_arrays[ii]])
    end

    # compute_risk test
    risk_test = compute_risk([1.0, 2.0, 3.0, 4.0], 0.5)
    @test risk_test ≈ 2.0*log(mean([exp(0.5*J) for J in [1.0, 2.0, 3.0, 4.0]]))

    # get_elite_samples_test
    @everywhere c(k, x, u) = 0.05*x'*x + 0.001*u'*u;
    @everywhere h(x) = 0.05*x'*x;

    problem = FiniteHorizonGenerativeProblem(f_stochastic, c, h, N);
    sampled_cost_arrays_serial = compute_cost_serial(solver, problem, x_init, control_sequence_array, rng);
    θ_array = [10.0*rand(rng) for ii = 1 : solver.num_risk_samples];
    kl_bound = 2.0;
    control_sequence_elite_idx_array, θ_elite_idx_array, obj_matrix =
    get_elite_samples(solver, control_sequence_array, θ_array, sampled_cost_arrays_serial, kl_bound);
    @test all([obj_matrix[ii, jj] ≈
               compute_risk(sampled_cost_arrays_serial[ii], θ_array[jj]) + kl_bound/θ_array[jj]
               for ii = 1 : solver.num_control_samples, jj = 1 : solver.num_risk_samples])

    begin
        obj_matrix_copied = copy(obj_matrix);
        control_sequence_elite_idx_array_test = Vector{Int64}();
        θ_elite_idx_array_test = Vector{Int64}();
        for ii = 1 : solver.num_elite
            u_idx, θ_idx = Tuple(findmin(obj_matrix_copied)[2])
            push!(control_sequence_elite_idx_array_test, u_idx)
            push!(θ_elite_idx_array_test, θ_idx)
            obj_matrix_copied[u_idx, θ_idx] = Inf;
        end
        @test sort(control_sequence_elite_idx_array_test) == sort(control_sequence_elite_idx_array)
        @test sort(θ_elite_idx_array_test) == sort(θ_elite_idx_array)
    end

    # compute_new_distribution test
    control_sequence_elite_array = control_sequence_array[control_sequence_elite_idx_array];
    θ_elite_array = θ_array[θ_elite_idx_array];
    μ_new_array, Σ_new_array, μ_θ_new, σ_θ_new =
        compute_new_distribution(solver, control_sequence_elite_array, θ_elite_array);

    @test length(μ_new_array) == solver.N;
    @test all([length(μ) == length(x_init) for μ in μ_new_array])
    @test length(Σ_new_array) == solver.N;
    @test all([size(Σ) == (length(x_init), length(x_init)) for Σ in Σ_new_array])
    for tt = 1 : solver.N
        @test μ_new_array[tt] ≈ (1.0 - solver.smoothing_factor).*mean([control_sequence_elite_array[ii][tt] for ii = 1 : solver.num_elite]) +
                                solver.smoothing_factor.*solver.μ_array[tt]
        @test Σ_new_array[tt] ≈ (1.0 - solver.smoothing_factor).*Diagonal(var([control_sequence_elite_array[ii][tt] for ii = 1 : solver.num_elite])) +
                                solver.smoothing_factor.*solver.Σ_array[tt]
    end

    @test μ_θ_new ≈ mean(θ_elite_array)
    @test σ_θ_new ≈ std(θ_elite_array, corrected=false)

    # step! test
    rng = MersenneTwister(1234);
    step!(solver, problem, x_init, kl_bound, rng, false, false);
    @test solver.iter_current == 1;

    # solve! test
    control_array, ~, θ_opt, ~ = solve!(solver, problem, x_init, rng, kl_bound=kl_bound, verbose=false);
    @test solver.iter_current == solver.iter_max
    @test !isnan(θ_opt)

    # mean_carry_over test
    @test solver.μ_init_array == μ_init_array
    @test solver.Σ_init_array == Σ_init_array
    @test solver.μ_θ_init == 1.0
    @test solver.σ_θ_init == 2.0


    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=100, deterministic_dynamics=false,
                          num_trajectory_samples=20,
                          μ_θ_init=1.0, σ_θ_init=2.0, num_risk_samples=20,
                          num_elite=50, iter_max=5, smoothing_factor=0.1,
                          mean_carry_over=true);
    control_array, ~, θ_opt, ~ = solve!(solver, problem, x_init, rng, kl_bound=kl_bound, verbose=false);
    @test solver.μ_init_array[1:end-1] == control_array[2:end]
    @test solver.μ_init_array[end] == zeros(2)
    @test solver.Σ_init_array == Σ_init_array
    @test solver.μ_θ_init == θ_opt
    @test solver.σ_θ_init == 2.0

    # test the case of kl_bound == 0.0
    solver = RATCEMSolver(μ_init_array, Σ_init_array,
                          num_control_samples=100, deterministic_dynamics=false,
                          num_trajectory_samples=20,
                          μ_θ_init=1.0, σ_θ_init=2.0, num_risk_samples=20,
                          num_elite=50, iter_max=5, smoothing_factor=0.1,
                          mean_carry_over=true);
    control_array_1, Σ_array_1, θ_opt, σ_θ =
        solve!(solver, problem, x_init, MersenneTwister(1234), kl_bound=0.0, verbose=false);

    @test θ_opt == 0.0
    @test σ_θ == 0.0
    @test solver.μ_init_array[1:end-1] == control_array_1[2:end]
    @test solver.μ_init_array[end] == zeros(2)
    @test solver.Σ_init_array == Σ_init_array
    @test solver.μ_θ_init == 0.0
    @test solver.σ_θ_init == 2.0

    pets_solver = PETSSolver(μ_init_array, Σ_init_array,
                             num_control_samples=100, deterministic_dynamics=false,
                             num_trajectory_samples=20,
                             num_elite=50, iter_max=5, smoothing_factor=0.1,
                             mean_carry_over=true);
    control_array_2, Σ_array_2 =
        solve!(pets_solver, problem, x_init, MersenneTwister(1234), verbose=false);
    @test control_array_1 == control_array_2
    @test Σ_array_1 == Σ_array_2
end
