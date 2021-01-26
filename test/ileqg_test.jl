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
    @test cost ≈ sum([c(ii - 1, x_array[ii], u_array[ii]) for ii = 1 : prob.N]) + h(x_array[end])

    # initialize test
    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array, 0.0);
    @test solver.l_array == u_array
    @test solver.L_array == [zeros(2, 2) for ii = 1 : prob.N]
    @test solver.x_array == x_array
    @test solver.μ == 0.0
    @test solver.Δ == solver.Δ_0
    @test solver.d_current == Inf
    @test solver.iter_current == 0;
    @test solver.ϵ_history == Tuple{Float64, Float64}[]
    dp_result_init = solve_approximate_dp(
                    approximate_model(prob, u_array, x_array),
                    [zeros(2, 2) for ii = 1 : N], θ=0.0, μ=0.0);
    @test solver.value_current ≈ dp_result_init.s_array[1];


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

    prob.c(k, x, u) = 0.5*x'*x + 1.0*u'*u;
    prob.h(x) = 0.5*x'*x
    approx_result = approximate_model(prob, u_array, x_array);
    # solve_approximate_dp test
    dp_result, dl_array_new = solve_approximate_dp!(solver, approx_result, false, θ=0.0);
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
    # # gains should match LQR solution.
    begin
        L_array_lqr = Vector{Matrix{Float64}}(undef, N);
        S_array_lqr = Vector{Matrix{Float64}}(undef, N + 1);
        S_array_lqr[N + 1] = approx_result.Q_array[N + 1];
        for ii = Iterators.reverse(1 : N)
            Q = approx_result.Q_array[ii];
            R = approx_result.R_array[ii];
            A = approx_result.A_array[ii];
            B = approx_result.B_array[ii];
            S_array_lqr[ii] = Q + (A')*S_array_lqr[ii + 1]*A -
                              (A')*S_array_lqr[ii + 1]*B/(R + (B')*S_array_lqr[ii + 1]*B)*(B')*S_array_lqr[ii + 1]*A;
        end
        for ii = 1 : N
            R = approx_result.R_array[ii];
            A = approx_result.A_array[ii];
            B = approx_result.B_array[ii];
            L_array_lqr[ii] = -(R + (B')*S_array_lqr[ii + 1]*B)\B*S_array_lqr[ii + 1]*A
        end
        @test all([all(L_array_lqr[ii] .≈ solver.L_array[ii]) for ii = 1 : N])
    end
    # # Nominal control offsets should be 0 (LQR is linear feedback)
    @test all(isapprox.(norm.(u_array .+ dl_array_new .- solver.L_array.*x_array[1:end-1]), 0.0, atol=1e-8))

    dp_result_2, dl_array_new_2 = solve_approximate_dp!(solver, approx_result, false, θ=1e-8);
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
    @test all([isapprox(dl_array_new[ii], dl_array_new_2[ii]) for ii = 1 : N])

    solve_approximate_dp!(solver, approx_result, false, θ=0.0);

    dp_result_3 = solve_approximate_dp(approx_result, solver.L_array, dl_array_new, θ=0.0, μ=0.0);
    @test dp_result_3.s_array == dp_result.s_array

    # line search test (for linear system)
    line_search!(solver, prob, dl_array_new, 0.0, false)
    @test solver.value_current .≈ dp_result.s_array[1] # line search should find the same optimal solution

    # increase_μ_and_Δ test
    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array, 0.0)
    increase_μ_and_Δ!(solver);
    @test solver.Δ == 4.0
    @test solver.μ == 1e-6

    # decrease_μ_and_Δ test
    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array, 0.0)
    decrease_μ_and_Δ!(solver);
    @test solver.Δ == 0.5
    @test solver.μ == 0.0

    # Testing with nonlinear model
    f(x, u) = x.^1.3 + u.^1.5;
    c(k, x, u) = sum(x.^2.5 + u.^2.5);
    h(x) = 1.0;
    W(k) = Matrix(0.01I, 2, 2);
    N = 10;
    θ = 0.5;

    u_array = [0.1*ones(2) for ii = 1 : prob.N];

    prob = FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N)

    solver = ILEQGSolver(prob);
    initialize!(solver, prob, zeros(2), u_array, θ)
    approx_result = approximate_model(prob, solver.l_array, solver.x_array,
                                      solver.A_array, solver.B_array);
    dp_result, dl_array = solve_approximate_dp!(solver, approx_result, false, θ=θ);
    value_new = line_search!(solver, prob, dl_array, θ, false)
    @test length(solver.ϵ_history) == 1
    @test solver.ϵ_history[1][1] == 1.0;
    @test solver.ϵ_history[1][2] < 0.0

    x_array, l_array, L_array, value, ϵ_history =
    solve!(solver, prob, zeros(2), u_array, θ=0.0, verbose=false);
    @test all([all(isapprox.(x_array[ii], zeros(2), atol=1e-4)) for ii = 1 : N + 1])
end
