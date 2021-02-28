#///////////////////////////////////////
#// File Name: pets_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/06
#// Description: Test code for src/pets.jl
#///////////////////////////////////////


using LinearAlgebra
using Statistics
using Random

@testset "PETS Test" begin

    f_stochastic(x, u, rng, use_true_model=false) = x + u + rand(rng, length(x))
    c(k, x, u) = sum(abs.(u));
    h(x) = 1.0;
    N = 20;

    problem = FiniteHorizonGenerativeProblem(f_stochastic, c, h, N);

    # CrossEntropyDirectOptimizationSolver test
    μ_init_array = [zeros(2) for ii = 1 : N];
    Σ_init_array = [Matrix(1.0I, 2, 2) for ii = 1 : N];
    direct_solver = CrossEntropyDirectOptimizationSolver(μ_init_array, Σ_init_array,
                                                         num_control_samples=20, num_trajectory_samples=100,
                                                         num_elite=5, iter_max=20, smoothing_factor=0.1);

    @test direct_solver.N == N
    @test direct_solver.iter_current == 0;
    @test direct_solver.μ_array == μ_init_array
    @test direct_solver.Σ_array == Σ_init_array

    direct_solver.iter_current = 10;
    direct_solver.μ_array = [ones(2) for ii = 1 : N];
    direct_solver.Σ_array = [Matrix(0.1I, 2, 2) for ii = 1 : N];
    initialize!(direct_solver)

    @test direct_solver.iter_current == 0
    @test direct_solver.μ_array == μ_init_array
    @test direct_solver.Σ_array == Σ_init_array

    # compute_cost_serial test
    rng = MersenneTwister(1234);
    control_sequence_array = [[rand(rng, 2) for tt = 1 : N] for ii = 1 : direct_solver.num_control_samples];
    x_init = zeros(2);
    cost_array_serial = compute_cost_serial(direct_solver, problem, x_init, control_sequence_array, rng);
    # compute_cost test
    cost_array = compute_cost(direct_solver, problem, x_init, control_sequence_array, rng);
    @test all(cost_array .== cost_array_serial) # note that this holds because c is not dependent on x. Otherwise they'd be different.

    rng = MersenneTwister(1234);
    @test length(cost_array) == direct_solver.num_control_samples
    for ii = 1 : direct_solver.num_control_samples
        x = x_init;
        cost = 0.0;
        for tt = 1 : direct_solver.N
            cost += problem.c(tt - 1, x, control_sequence_array[ii][tt])
            x = problem.f_stochastic(x, control_sequence_array[ii][tt], rng)
        end
        cost += problem.h(x)
        @test cost ≈ cost_array[ii]
    end

    # get_elite_samples test
    control_sequence_elite_array = get_elite_samples(direct_solver, control_sequence_array, cost_array);

    @test length(control_sequence_elite_array) == direct_solver.num_elite
    elite_idx_array = [pair[1] for pair in sort(collect(zip(1:1:direct_solver.num_control_samples, cost_array)), by=x->x[2])[1 : direct_solver.num_elite]]
    @test control_sequence_elite_array == control_sequence_array[elite_idx_array]

    # compute_new_distribution test
    μ_new_array, Σ_new_array = compute_new_distribution(direct_solver, control_sequence_elite_array);

    @test length(μ_new_array) == direct_solver.N;
    @test all([length(μ) == length(x_init) for μ in μ_new_array])
    @test length(Σ_new_array) == direct_solver.N;
    @test all([size(Σ) == (length(x_init), length(x_init)) for Σ in Σ_new_array])
    for tt = 1 : direct_solver.N
        @test μ_new_array[tt] ≈ (1.0 - direct_solver.smoothing_factor).*mean([control_sequence_elite_array[ii][tt] for ii = 1 : direct_solver.num_elite]) +
                                direct_solver.smoothing_factor.*direct_solver.μ_array[tt]
        @test Σ_new_array[tt] ≈ (1.0 - direct_solver.smoothing_factor).*Diagonal(var([control_sequence_elite_array[ii][tt] for ii = 1 : direct_solver.num_elite])) +
                                direct_solver.smoothing_factor.*direct_solver.Σ_array[tt]
    end

    # step! test
    rng = MersenneTwister(1234);
    step!(direct_solver, problem, x_init, rng, false, false);
    @test direct_solver.iter_current == 1;

    # solve! test
    control_array, ~ = solve!(direct_solver, problem, x_init, rng, use_true_model=false, verbose=false);

    @test direct_solver.iter_current == direct_solver.iter_max

end
