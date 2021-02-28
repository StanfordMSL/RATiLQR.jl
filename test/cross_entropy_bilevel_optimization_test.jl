#///////////////////////////////////////
#// File Name: cross_entropy_bilevel_optimization_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/03
#// Description: Test code for src/cross_entropy_bilevel_optimization.jl
#///////////////////////////////////////

using Test
using Random
@everywhere using LinearAlgebra

@testset "Cross Entropy Bilevel Optimization Test" begin
    @everywhere f(x, u) = x.^1.3 + u.^1.5;
    @everywhere c(k, x, u) = sum(x.^2.5 + u.^2.5);
    @everywhere h(x) = 1.0;
    @everywhere W(k) = Matrix(0.01I, 2, 2);
    N = 10;

    x_0 = zeros(2)
    u_array = [0.1*ones(2) for ii = 1 : N];

    problem = FiniteHorizonAdditiveGaussianProblem(f, c, h, W, N)

    solver = CrossEntropyBilevelOptimizationSolver(num_samples=3);
    initialize!(solver)

    θ_array = [0.1, 0.3, 0.43];
    kl_bound = 1.0
    costs = compute_cost(solver, problem, x_0, u_array, θ_array, kl_bound)

    costs_test = compute_cost_serial(solver, problem, x_0, u_array, θ_array, kl_bound)
    @test all(isapprox(costs, costs_test))

    θ_array = get_positive_samples(0.0, 1.0, 10, MersenneTwister(123));
    @test all(θ_array .> 0.0) && length(θ_array) == 10

    rng = MersenneTwister(12344)
    θ_opt, x_array, l_array, L_array, c_opt =
    solve!(solver, problem, x_0, u_array, rng, kl_bound=kl_bound, verbose=false)
    @test !isinf(c_opt)
    @test !isnan(θ_opt)
end
