#///////////////////////////////////////
#// File Name: mppi_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/08/08
#// Description: Test code for src/mppi.jl
#///////////////////////////////////////

using LinearAlgebra
using Random
import StatsFuns: softmax

@testset "MPPI Test" begin

    @everywhere f_stochastic(x, u, rng, deterministic=false) = x + u + rand(rng, length(x))
    @everywhere c(k, x, u) = sum(abs.(u));
    @everywhere h(x) = 1.0;
    N = 20;

    problem = FiniteHorizonGenerativeProblem(f_stochastic, c, h, N);

    # MPPISolver test
    μ_init_array = [zeros(2) for ii = 1 : N];
    Σ = Matrix(0.5I, 2, 2);
    solver = MPPISolver(μ_init_array, Σ,
                        num_control_samples=20, deterministic_dynamics=false,
                        num_trajectory_samples=100, weighting_factor=2.0,
                        iter_max=20, filtering_coefficient=0.5,
                        mean_carry_over=false);

    @test solver.N == N;
    @test solver.iter_current == 0;
    @test solver.μ_array == μ_init_array;
    @test solver.Σ == Σ;

    solver.iter_current = 10;
    solver.μ_array = [ones(2) for ii = 1 : N];
    initialize!(solver);

    @test solver.iter_current == 0;
    @test solver.μ_array == μ_init_array;

    # compute_cost_serial test
    rng = MersenneTwister(1234);
    control_sequence_array = [[rand(rng, 2) for tt = 1 : N] for ii = 1 : solver.num_control_samples];
    x_init = zeros(2);
    cost_array_serial = compute_cost_serial(solver, problem, x_init, control_sequence_array, rng);
    # compute_cost test
    cost_array = compute_cost(solver, problem, x_init, control_sequence_array, rng);
    @test all(cost_array .== cost_array_serial) # note that this holds because c is not dependent on x. Otherwise they'd be different.

    rng = MersenneTwister(1234);
    @test length(cost_array) == solver.num_control_samples
    for ii = 1 : solver.num_control_samples
        x = x_init;
        cost = 0.0;
        for tt = 1 : solver.N
            cost += problem.c(tt - 1, x, control_sequence_array[ii][tt])
            x = problem.f_stochastic(x, control_sequence_array[ii][tt], rng)
        end
        cost += problem.h(x)
        @test cost ≈ cost_array[ii]
    end

    # update_mean! test
    update_mean!(solver, control_sequence_array, cost_array);
    weight_array_test = softmax(-solver.weighting_factor .* cost_array);
    @test all(solver.μ_array .≈
    [sum(weight_array_test .* [control_sequence_array[ii][tt] for ii = 1 : solver.num_control_samples]) for tt = 1 : solver.N])

    # step! test
    rng = MersenneTwister(1234);
    step!(solver, problem, x_init, rng, false, false);

    # solve! test
    control_array = solve!(solver, problem, x_init, rng, verbose=false);

    @test solver.iter_current == solver.iter_max;

    # mean_carry_over test
    @test solver.μ_init_array == μ_init_array;

    solver = MPPISolver(μ_init_array, Σ,
                        num_control_samples=20, deterministic_dynamics=false,
                        num_trajectory_samples=100, weighting_factor=2.0,
                        iter_max=20, filtering_coefficient=0.5,
                        mean_carry_over=true);
    control_array = solve!(solver, problem, x_init, rng, verbose=false);
    @test solver.μ_init_array[1:end-1] == control_array[2:end];
    @test solver.μ_init_array[end] == zeros(2);
    @test solver.Σ == Σ;
end
                      
