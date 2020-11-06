#///////////////////////////////////////
#// File Name: ileqg_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/03
#// Description: Test code for src/ileqg.jl
#///////////////////////////////////////

using LinearAlgebra
using Test

@testset "iLEQG Test" begin
    f(x, u) = x + u;
    c(k, x, u) = k;
    h(x) = 1.0;
    W(k) = Matrix(1.0I, 2, 2);
    N = 10;

    prob = FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N)

    u_array = [ones(2) for ii = 1 : prob.N];
    x_array = simulate_dynamics(prob, zeros(2), u_array);

    @test x_array[1] == zeros(2)
    @test all([x_array[ii + 1] == f(x_array[ii], u_array[ii]) for ii = 1 : prob.N - 1])

    L_array = [ones(2, 2) for ii = 1:prob.N];
    x_array_new, u_array_new = simulate_dynamics(prob, x_array, u_array, L_array)
    @test all([u_array_new[ii] == u_array[ii] for ii = 1:prob.N])
    @test all([x_array_new[ii] == x_array[ii] for ii = 1:prob.N])

    # integrate_cost test
    cost = integrate_cost(prob, x_array, u_array)
    @test cost ≈ sum([c(ii, x_array[ii], u_array[ii]) for ii = 1 : prob.N]) + h(x_array[end])

    # initialize test
    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array)
    @test solver.l_array == u_array
    @test solver.L_array == [zeros(2, 2) for ii = 1 : prob.N]
    @test solver.x_array == x_array
    @test solver.μ == solver.μ_min
    @test solver.Δ == solver.Δ_0
    @test solver.d_current == Inf
    @test solver.iter_current == 0;
    @test solver.ϵ_history == Tuple{Float64, Float64}[]

    # approximate_model test
    prob.c(k, x, u) = 0.5*x'*x + 1.0*u'*u + x'*u
    prob.h(x) = 0.5*x'*x
    approx_result = approximate_model(prob, u_array, x_array);

    @test all([isapprox(approx_result.q_array[ii], 0.5*(2*(ii - 1)^2) + 1.0*2 + 2*(ii - 1)) for ii = 1 : length(u_array)])
    @test isapprox(approx_result.q_array[end], h(x_array[end]))

    @test all([isapprox(approx_result.q_vec_array[ii], x_array[ii] + ones(2)) for ii = 1:length(u_array)])
    @test isapprox(approx_result.q_vec_array[end], x_array[end])
    @test all([isapprox(approx_result.Q_array[ii], Matrix(1.0I, 2, 2)) for ii = 1:length(x_array)])
    @test all([isapprox(approx_result.r_array[ii], x_array[ii] + 2.0*ones(2)) for ii = 1:length(u_array)])
    @test all([isapprox(approx_result.R_array[ii], 2.0*Matrix(1.0I, 2, 2)) for ii = 1:length(u_array)])
    @test all([isapprox(approx_result.P_array[ii], Matrix(1.0I, 2, 2)) for ii = 1:length(u_array)])
    @test all([approx_result.W_array[ii] == Matrix(1.0I, 2, 2) for ii = 1:length(u_array)])

    # solve_approximate_dp test
    l_array = u_array
    L_array = [zeros(2, 2) for ii = 1:length(u_array)];

    dp_result = solve_approximate_dp(approx_result, l_array, L_array, θ=0.0);
    @test length(dp_result.s_array) == length(x_array)
    @test length(dp_result.s_vec_array) == length(x_array)
    @test all([length(s_vec) == 2 for s_vec in dp_result.s_vec_array])
    @test length(dp_result.S_array) == length(x_array)
    @test all([size(S) == (2, 2) for S in dp_result.S_array])
    @test all([issymmetric(S) for S in dp_result.S_array])
    @test all([isposdef(S) for S in dp_result.S_array])
    @test length(dp_result.g_array) == length(u_array)
    @test all([length(g) == 2 for g in dp_result.g_array])
    @test length(dp_result.G_array) == length(u_array)
    @test all([size(G) == (2, 2) for G in dp_result.G_array])
    @test length(dp_result.H_array) == length(u_array)
    @test all([size(H) == (2, 2) for H in dp_result.H_array])

    dp_result_2 = solve_approximate_dp(approx_result, l_array, L_array, θ=1e-8);
    @test length(dp_result_2.s_array) == length(x_array)
    @test length(dp_result_2.s_vec_array) == length(x_array)
    @test all([length(s_vec) == 2 for s_vec in dp_result_2.s_vec_array])
    @test length(dp_result_2.S_array) == length(x_array)
    @test all([size(S) == (2, 2) for S in dp_result_2.S_array])
    @test all([issymmetric(S) for S in dp_result_2.S_array])
    @test all([isposdef(S) for S in dp_result_2.S_array])
    @test length(dp_result_2.g_array) == length(u_array)
    @test all([length(g) == 2 for g in dp_result_2.g_array])
    @test length(dp_result_2.G_array) == length(u_array)
    @test all([size(G) == (2, 2) for G in dp_result_2.G_array])
    @test length(dp_result_2.H_array) == length(u_array)
    @test all([size(H) == (2, 2) for H in dp_result_2.H_array])
    @test isapprox(dp_result.s_array[1], dp_result_2.s_array[1], rtol=1e-5)

    # increase_μ_and_Δ test
    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array)
    increase_μ_and_Δ!(solver);
    @test solver.Δ == 4.0
    @test solver.μ == 4e-6

    # decrease_μ_and_Δ test
    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array)
    decrease_μ_and_Δ!(solver);
    @test solver.Δ == 0.5
    @test solver.μ == 0.0

    f(x, u) = x.^1.3 + u.^1.5;
    c(k, x, u) = sum(x.^2.5 + u.^2.5);
    h(x) = 1.0;
    W(k) = Matrix(0.01I, 2, 2);
    N = 10;
    θ = 0.5;

    u_array = [0.1*ones(2) for ii = 1 : prob.N];

    prob = FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N)

    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array)
    approx_result = approximate_model(prob, solver.l_array, solver.x_array,
                                      solver.A_array, solver.B_array);
    dp_result = solve_approximate_dp(approx_result, solver.l_array, solver.L_array,
                                 θ=θ);
    # regularize! test
    dp_result = regularize!(solver, dp_result, false);
    @test all([isposdef(H) for H in dp_result.H_array])

    # line_search! test
    value_new = line_search!(solver, prob, dp_result, θ, false)
    @test length(solver.ϵ_history) == 1
    @test solver.ϵ_history[1][1] == 1.0;
    @test solver.ϵ_history[1][2] < 0.0

    x_array, l_array, L_array, value, ϵ_history =
    solve!(solver, prob, zeros(2), u_array, θ=θ, verbose=false);
end
