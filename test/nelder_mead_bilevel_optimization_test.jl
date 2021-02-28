#///////////////////////////////////////
#// File Name: nelder_mead_bilevel_optimization_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/05
#// Description: Test code for src/nelder_mead_bilevel_optimization.jl
#///////////////////////////////////////

using Test
using LinearAlgebra

@testset "Nelder-Mead Simplex Bilevel Optimization Test" begin
    f(x, u) = x.^1.3 + u.^1.5;
    c(k, x, u) = sum(x.^2.5 + u.^2.5);
    h(x) = 1.0;
    W(k) = Matrix(0.01I, 2, 2);
    N = 10;

    x_0 = zeros(2)
    u_array = [0.1*ones(2) for ii = 1 : N];

    problem = FiniteHorizonAdditiveGaussianProblem(f, c, h, W, N);
    kl_bound = 1.0;
    nm_solver = NelderMeadBilevelOptimizationSolver(iter_max=20, ϵ=1e-3, θ_high_init=10.0, θ_low_init=1e-8);

    θ_opt, x_array, l_array, L_array, c_opt =
    solve!(nm_solver, problem, x_0, u_array, kl_bound=kl_bound, verbose=false)
    @test !isinf(c_opt)
    @test !isnan(θ_opt)
    c_low_init = compute_cost_worker(nm_solver, problem, x_0, u_array, nm_solver.θ_low_init, kl_bound)
    c_high_init = compute_cost_worker(nm_solver, problem, x_0, u_array, nm_solver.θ_high_init, kl_bound)
    @test !isinf(c_low_init) && !isinf(c_high_init);
    @test c_opt <= c_low_init && c_opt <= c_high_init
end
