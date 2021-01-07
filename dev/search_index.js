var documenterSearchIndex = {"docs":
[{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/#iLQG,-iLEQG","page":"References","title":"iLQG, iLEQG","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"Part of our iLQG and iLEQG implementations are based on the following papers:","category":"page"},{"location":"references/","page":"References","title":"References","text":"E. Todorov and W. Li, \"A generalized iterative lqg method for locally-optimal feedback control of constrained nonlinear stochastic systems,\" in Proceedings of the 2005, American Control Conference, 2005. IEEE, 2005, pp. 300–306.\nY. Tassa, T. Erez, and E. Todorov, \"Synthesis and stabilization of complex behaviors through online trajectory optimization,\" in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012, pp. 4906–4913.\nJ. van den Berg, S. Patil, and R. Alterovitz, \"Motion planning under uncertainty using iterative local optimization in belief space,\" The International Journal of Robotics Research, vol. 31, no. 11, pp. 1263–1278, 2012.\nM. Wang, N. Mehr, A. Gaidon, and M. Schwager, \"Game-theoretic planning for risk-aware interactive agents,\" in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2020, pp.6998–7005.","category":"page"},{"location":"references/#RAT-iLQR,-RAT-iLQR","page":"References","title":"RAT iLQR, RAT iLQR++","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"The RAT iLQR algorithm is originally presented in our paper:","category":"page"},{"location":"references/","page":"References","title":"References","text":"H. Nishimura, N. Mehr, A. Gaidon, and M. Schwager, \"Rat ilqr: a risk auto-tuning controller to optimally account for stochastic model mismatch,\" IEEE Robotics and Automation Letters. IEEE, 2021.","category":"page"},{"location":"references/","page":"References","title":"References","text":"We used the following textbook as a reference to implement the Cross Entropy Method and the Nelder-Mead Simplex Method:","category":"page"},{"location":"references/","page":"References","title":"References","text":"M. J. Kochenderfer and T. A. Wheeler, Algorithms for optimization. The MIT Press, 2019.","category":"page"},{"location":"references/#PETS","page":"References","title":"PETS","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"PETS is originally presented in the first paper. Note that we provide the MPC implementation but not the model-learning part. We referred to the second paper as well during the development of this software.","category":"page"},{"location":"references/","page":"References","title":"References","text":"K. Chua, R. Calandra, R. McAllister, and S. Levine, \"Deep reinforcement learning in a handful of trials using probabilistic dynamics models,\" in Advances in Neural Information Processing Systems, 2018, pp. 4754–4765.\nA. Nagabandi, K. Konoglie, S. Levine, and V. Kumar, \"Deep dynamics models for learning dexterous manipulation,\" in Conference on Robot Learning. PMLR, 2020, pp. 1101-1112.","category":"page"},{"location":"getting-started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting-started/#Installation","page":"Getting Started","title":"Installation","text":"","category":"section"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"This package is developed and tested on Julia v1.5.2, but the code should be combatible with any Julia v1.x. To install the package, run the following commands in the Julia REPL.","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"import Pkg\nPkg.add(url=\"https://github.com/StanfordMSL/RATiLQR.jl.git\")","category":"page"},{"location":"getting-started/#Example","page":"Getting Started","title":"Example","text":"","category":"section"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"In this example, we are going to optimize an LQR objective with  imperfect knowlege of the probability distribution governing the stochastic state transitions of the system. Specifically, we seek to minimize the following objective:","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"max_p in Xi mathbbE_p leftfrac12 sum_k=0^N - 1 left(x_k^mathrmTQx_k + u_k^mathrmTR u_kright) +\nfrac12x_N^mathrmTRx_Nright","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"textsubject to  x_k+1 = x_k + 01 times u_k + w_k in mathbbR^2  (w_0w_N-1) sim p(w_0N-1)","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"where the ambiguity set Xi triangleq p mathbbD_mathrmKL(p Vert q) leq d for the true but unknown distribution p is defined by a reference (Gaussian) distribution q(w_0N-1) triangleq prod_k = 0^N - 1 mathcalN(0 W) and an upper-bound d geq 0 on the KL divergence between p and q. This problem is an instance of a type of problems called Distributionally  Robust Optimal Contol, and this particular problem can be solved by the RATiLQR.jl package.","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"With the RAT iLQR Solver, we can find a locally optimal, affine state-feedback  policy of the form:","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"u_k = pi_k(x) triangleq L_k(x - x_k) + l_k ","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"to approximately minimize the objective.","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"The following Julia code defines the problem using our Problem Definition API and performs the local optimization:","category":"page"},{"location":"getting-started/","page":"Getting Started","title":"Getting Started","text":"# We are going to use 4 worker processes to distribute the Cross Entropy sampling.\n# If you are not doing distributed processing you don't need @everywhere macro.\nusing Distributed\naddprocs(4);\n\n@everywhere using LinearAlgebra, Random\n@everywhere using RATiLQR\n\n\n#=============== PROBLEM DEFINITION ===============#\n# generic 2D single integrator dynamics function with parameter dt.\n@everywhere function f(x, u, dt, f_returns_jacobian::Bool)\n    @assert length(x) == length(u) == 2;\n    x_next = x + dt*u;\n    if f_returns_jacobian\n        A = Matrix(1.0LinearAlgebra.I, 2, 2); # dx_next/dx\n        B = Matrix(dt*LinearAlgebra.I, 2, 2); # dx_next/du\n        return x_next, A, B;\n    else\n        return x_next;\n    end\nend\n\n# generic quadratic cost function with parameters Q and R.\n@everywhere function c(x, u, Q, R)\n    return 1/2*x'*Q*x + 1/2*u'*R*u;\nend\n\n# define your own model.\n@everywhere mutable struct SingleIntegratorLQRModel\n    # Model parameters\n    dt::Float64        # discrete time inverval [s].\n    Q::Matrix{Float64} # 2-by-2 state cost weight matrix.\n    R::Matrix{Float64} # 2-by-2 control cost weight matrix.\n\n    # Member functions needed to define the optimal control problem\n    f::Union{Nothing, Function}  # Noiseless dynamics function x_{k+1} = f(x_k, u_k)\n    c::Union{Nothing, Function}  # stage cost function c(k, x_k, u_k)\n    h::Union{Nothing, Function}  # terminal cost function c(x_N)\n    W::Union{Nothing, Function}  # Gaussian covariance matrix function W(k)\n\n    function SingleIntegratorLQRModel(dt=0.1, Q=Matrix(1.0I, 2, 2), R=Matrix(0.01I, 2, 2))\n        this = new(dt, Q, R, nothing, nothing, nothing, nothing); # leave member functions nothing.\n\n        # Define member functions below so their formats are compatible with the OptimalControlProblem type.\n        this.f = (x, u, f_returns_jacobian=false) -> f(x, u, this.dt, f_returns_jacobian);\n        this.c = (k, x, u) -> c(x, u, this.Q, this.R); # it is ok if c is not dependent on time k.\n        this.h = x -> c(x, zeros(2), this.Q, this.R);\n        this.W = k -> Matrix(0.1*this.dt*I, 2, 2); # it is ok if W is not dependent on time k.\n\n        return this\n    end\nend\n\n\n#=============== OPTIMIZATION ===============#\n# Instantiate a model.\nmodel = SingleIntegratorLQRModel();\n\n# Define an OptimalControlProblem.\nN = 10; # final time index. Note that the initial time is 0.\nproblem = FiniteHorizonRiskSensitiveOptimalControlProblem(model.f, model.c, model.h, model.W, N)\n\n# Instantiate a RAT iLQR solver.\nsolver = CrossEntropyBilevelOptimizationSolver();\n\n# Solve problem to obtain a feedback control policy: π_k(x) = L_k(x - x_k) + l_k.\nrng = MersenneTwister(12345); # pseudo random number generator for the Cross Entropy method\nx_0 = [5.0, 5.0]; # initial state\nu_array = [zeros(2) for ii = 1 : N]; # initial guess for the nominal control schedule\nkl_bound = 0.1; # kl divergence bound between the true unknown noise distribution and the Gaussian model.\n\nθ_opt, x_array, l_array, L_array, value, θ_min, θ_max =\n    solve!(solver, problem, x_0, u_array, rng, kl_bound=kl_bound);","category":"page"},{"location":"optimal-control/#Optimal-Control-Problems","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"","category":"section"},{"location":"optimal-control/#Basics","page":"Optimal Control Problems","title":"Basics","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Suppose that we are given a discrete-time stochastic dynamics model of the form","category":"page"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"x_k+1 = f(x_k u_k w_k)","category":"page"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"where x_k in mathbbR^n is the state, u_k in mathbbR^m is the control input, and  w_k in mathbbR^r is the stochastic noise variable drawn from some probability distribution on mathbbR^r.","category":"page"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Subject to the dynamics constraint, we are interested in minimizing an objective function J over  some finite horizon N:","category":"page"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"J(x_0N u_0N-1) triangleq sum_k=0^N - 1 c(k x_k u_k) + h(x_N)","category":"page"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"where c(k x_k u_k) geq 0 is the stage cost at time k and h(x_N) geq 0 is the terminal cost.","category":"page"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Although not explicitly written above, the actual value of J is dependent on the history of stochastic  noise w_0N-1. This means that the objective J itself is a random variable whose value cannot  be determined without actually observing the outcome. Therefore, Stochastic Opimal Control seeks to minimize a  statistic associated with the objective, such as the expectation mathbbEJ.","category":"page"},{"location":"optimal-control/#Supported-Problem-Types","page":"Optimal Control Problems","title":"Supported Problem Types","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"With RATiLQR.jl, you can formulate various types of stochastic optimal control problems with different objective  statistics and dynamics models.","category":"page"},{"location":"optimal-control/#.-Standard-Problem-with-Gaussian-Noise","page":"Optimal Control Problems","title":"1. Standard Problem with Gaussian Noise","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Objective: mathbbEJ\nDynamics: f(x_k u_k w_k) = f(x_k u_k) + w_k  w_k sim mathcalN(0 W_k)\nDefinition: FiniteHorizonRiskSensitiveOptimalControlProblem\nSolvers: ILEQGSolver","category":"page"},{"location":"optimal-control/#.-Standard-Problem-with-Arbitrary-Noise","page":"Optimal Control Problems","title":"2. Standard Problem with Arbitrary Noise","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Objective: mathbbEJ\nDynamics: f(x_k u_k w_k) where w_k has an arbitrary distribution.\nDefinition: FiniteHorizonGenerativeOptimalControlProblem\nSolvers: CrossEntropyDirectOptimizationSolver","category":"page"},{"location":"optimal-control/#.-Risk-Sensitive-Problem-with-Gaussian-Noise","page":"Optimal Control Problems","title":"3. Risk-Sensitive Problem with Gaussian Noise","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Objective: frac1thetalogleft(mathbbEexp(theta J)right) where theta  0 denotes the risk-sensitivity parameter and is user-specified.\nDynamics: f(x_k u_k w_k) = f(x_k u_k) + w_k  w_k sim mathcalN(0 W_k)\nDefinition: FiniteHorizonRiskSensitiveOptimalControlProblem\nSolvers: ILEQGSolver","category":"page"},{"location":"optimal-control/#.-Distributionally-Robust-Problem","page":"Optimal Control Problems","title":"4. Distributionally-Robust Problem","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"Objective: max_p in Xi mathbbE_p J where p denotes the true, potentially unknown  distribution over the noise variables w_0N-1 and Xi is the ambiguity set that encodes how uncertain we are about the true distribution. Note that p may not be Gaussian. In our formulation, Xi is defined by a KL divergence  bound between the true distribution p(w_0N-1) and the Gaussian model  q(w_0N-1) triangleq prod_k = 0^N - 1 mathcalN(0 W_k) as follows:\nXi triangleq p mathbbD_mathrmKL(p Vert q) leq d\nwhere d  0 is a user-specified constant that defines an upper bound.\nDynamics: f(x_k u_k w_k) = f(x_k u_k) + w_k where w_0N-1 can have an arbitrary distribution as long as it is within the ambiguity set defined by the Gaussian model.\nDefinition: FiniteHorizonRiskSensitiveOptimalControlProblem\nSolvers: CrossEntropyBilevelOptimizationSolver, NelderMeadBilevelOptimizationSolver","category":"page"},{"location":"optimal-control/#Problem-Definition-APIs","page":"Optimal Control Problems","title":"Problem Definition APIs","text":"","category":"section"},{"location":"optimal-control/","page":"Optimal Control Problems","title":"Optimal Control Problems","text":"OptimalControlProblem\nFiniteHorizonRiskSensitiveOptimalControlProblem\nFiniteHorizonGenerativeOptimalControlProblem","category":"page"},{"location":"optimal-control/#RATiLQR.OptimalControlProblem","page":"Optimal Control Problems","title":"RATiLQR.OptimalControlProblem","text":"OptimalControlProblem\n\nAbstract base type for an Optimal Control Problem.\n\n\n\n\n\n","category":"type"},{"location":"optimal-control/#RATiLQR.FiniteHorizonRiskSensitiveOptimalControlProblem","page":"Optimal Control Problems","title":"RATiLQR.FiniteHorizonRiskSensitiveOptimalControlProblem","text":"FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N) <: OptimalControlProblem\n\nA finite horizon, stochastic optimal control problem where the dynamics function is subject to additive Gaussian noise w ~ N(0, W).\n\nArguments\n\nf(x, u, f_returns_jacobian=false) – deterministic dynamics function\nx is a state vector and u is a control input vector.\nThe third positional argument f_returns_jacobian determines whether the user computes and returns the Jacobians, and should default to false. If true, the return value must be augmented with matrices A and B, where A = dx_next/dx and B = dx_next/du. Otherwise the return value is the (noiseless) next state x_next.\nc(k, x, u) – stage cost function\nk::Int >= 0 is a time index where k == 0 is the initial time.\nWe assume that c is non-negative.\nh(x) – terminal cost function\nWe assume that h is non-negative.\nW(k) – covariance matrix function\nReturns a symmetric positive semidefinite matrix that represents the covariance matrix for additive Gaussian noise w ~ N(0, W).\nk::Int >= 0 is a time index where k == 0 is the initial time.\nN::Int64 – final time index\nNote that 0 is the initial time index.\n\nNotes\n\nFunctions f, c, and h should be written generically enough to accept the state x and the input u of type Vector{<:Real}. This is to ensure that ForwardDiff can compute Jacobians and Hessians for iLQG/iLEQG.\n\nExample\n\nimport LinearAlgebra;\n\nfunction f(x, u, f_returns_jacobian=false)\n    x_next = x + u; # 2D single integrator dynamics\n    if f_returns_jacobian\n        A = Matrix(1.0LinearAlgebra.I, 2, 2); # dx_next/dx\n        B = Matrix(1.0LinearAlgebra.I, 2, 2); # dx_next/du\n        return x_next, A, B;\n    else\n        return x_next;\n    end\nend\n\nc(k, x, u) = k/2*x'*x + k/2*u'*u  # time-dependent quadratic stage cost\nN = 10;\nh(x) = N/2*x'*x; # quadratic terminal cost\nW(k) = Matrix(0.1LinearAlgebra.I, 2, 2);\n\nproblem = FiniteHorizonRiskSensitiveOptimalControlProblem(f, c, h, W, N);\n\n\n\n\n\n","category":"type"},{"location":"optimal-control/#RATiLQR.FiniteHorizonGenerativeOptimalControlProblem","page":"Optimal Control Problems","title":"RATiLQR.FiniteHorizonGenerativeOptimalControlProblem","text":"FiniteHorizonGenerativeOptimalControlProblem(f_stochastic, c, h, N) <: OptimalControlProblem\n\nA finite horizon, stochastic optimal control problem where the dynamics function is stochastic and generative.\n\nArguments\n\nf_stochastic(x, u, rng, use_true_model=false) – stochastic dynamics function\nx is a state vector and u is a control input vector.\nThe third positional argument rng is a random seed.\nThe fourth positional argument use_true_model determines whether a solver has access to the true stochastic dynamics and defaults to false.\nThe return value is the (noisy) next state x_next.\nc(k, x, u) – stage cost function\nk::Int >= 0 is a time index where k == 0 is the initial time.\nWe assume that c is non-negative.\nh(x) – terminal cost function\nWe assume that h is non-negative.\nN::Int64 – final time index\nNote that 0 is the initial time index.\n\nExample\n\nimport Distributions;\nimport LinearAlgebra;\n\nfunction f_stochastic(x, u, rng, use_true_model=false)\n    Σ_1 = Matrix(0.5LinearAlgebra.I, 2, 2);\n\n    if use_true_model  # accurate GMM model\n        Σ_2 = Matrix(1.0LinearAlgebra.I, 2, 2)\n        d = Distributions.MixtureModel([Distributions.MvNormal(zeros(2), Σ_1),\n                                        Distributions.MvNormal(ones(2), Σ_2)],\n                                        [0.5, 0.5]);\n    else  # inaccurate Gaussian model\n        d = Distributions.MvNormal(zeros(2), Σ_1);\n    end\n\n    x_next = x + u + Distributions.rand(rng, d); # 2D single integrator dynamics\n    return x_next;\nend\n\nc(k, x, u) = k/2*x'*x + k/2*u'*u  # time-dependent quadratic stage cost\nN = 10;\nh(x) = N/2*x'*x; # quadratic terminal cost\nW(k) = Matrix(0.1LinearAlgebra.I, 2, 2);\n\nproblem = FiniteHorizonGenerativeOptimalControlProblem(f_stochastic, c, h, N);\n\n\n\n\n\n","category":"type"},{"location":"solvers/#Solver-APIs","page":"Solver APIs","title":"Solver APIs","text":"","category":"section"},{"location":"solvers/#iLQG/iLEQG-Solver","page":"Solver APIs","title":"iLQG/iLEQG Solver","text":"","category":"section"},{"location":"solvers/","page":"Solver APIs","title":"Solver APIs","text":"ILEQGSolver","category":"page"},{"location":"solvers/#RATiLQR.ILEQGSolver","page":"Solver APIs","title":"RATiLQR.ILEQGSolver","text":"iLEQGSolver(problem::FiniteHorizonRiskSensitiveOptimalControlProblems, kwargs...)\n\niLQG and iLEQG Solver for problem.\n\nOptional Keyword Arguments\n\nμ_min::Float64 – minimum value for Hessian regularization parameter μ (> 0). Default: 1e-6.\nΔ_0::Float64 – minimum multiplicative modification factor (> 0) for μ. Default: 2.0.\nλ::Float64 – multiplicative modification factor in (0, 1) for line search step size ϵ. Default: 0.5.\nd::Float64 – convergence error norm threshold (> 0). If the maximum l2 norm of the change in nominal control over the horizon is less than d, the solver is considered to be converged. Default: 1e-2.\niter_max::Int64 – maximum iteration number. Default: 100.\nβ::Float64 – Armijo condition number (>= 0) defining a sufficient decrease for backtracking line search. If β == 0, then any cost-to-go improvement is considered a sufficient decrease. Default: 1e-4.\nϵ_init::Float64 – initial step size in (ϵ_min, 1] to start the backtracking line search with. If adaptive_ϵ_init is true, then this value is overridden by the solver's adaptive initialization functionality after the first iLEQG iteration. If adaptive_ϵ_init is false, the specified value of ϵ_init is used across all the iterations as the initial step size. Default:1.0.\nadaptive_ϵ_init::Bool – if true, ϵ_init is adaptively changed based on the last step size ϵ of the previous iLEQG iteration. Default: false.\nIf the first line search iterate ϵ_init_prev in the previous iLEQG iteration is successful, then ϵ_init for the next iLEQG iteration is set to ϵ_init = ϵ_init_prev / λ so that the initial line search step increases.\nOtherwise ϵ_init = ϵ_last where ϵ_last is the line search step accepted in the previous iLEQG iteration.\nϵ_min::Float64 – minimum value of step size ϵ to terminate the line search. When ϵ_min is reached, the last candidate nominal trajectory is accepted regardless of the Armijo condition and the current iLEQG iteration is finished. Default: 1e-6.\nf_returns_jacobian::Bool – if true, Jacobian matrices of the dynamics function are user-provided. This can reduce computation time since automatic differentiation is not used. Default: false.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#RAT-iLQR-Solver","page":"Solver APIs","title":"RAT iLQR Solver","text":"","category":"section"},{"location":"solvers/","page":"Solver APIs","title":"Solver APIs","text":"CrossEntropyBilevelOptimizationSolver","category":"page"},{"location":"solvers/#RATiLQR.CrossEntropyBilevelOptimizationSolver","page":"Solver APIs","title":"RATiLQR.CrossEntropyBilevelOptimizationSolver","text":"CrossEntropyBilevelOptimizationSolver(kwargs...)\n\nRAT iLQR (i.e. Cross Entropy Method + iLEQG) Solver.\n\nOptional Keyword Arguments\n\niLEQG Solver Parameters\n\nμ_min_ileqg::Float64 – minimum value for Hessian regularization parameter μ (> 0). Default: 1e-6.\nΔ_0_ileqg::Float64 – minimum multiplicative modification factor (> 0) for μ. Default: 2.0.\nλ_ileqg::Float64 – multiplicative modification factor in (0, 1) for line search step size ϵ. Default: 0.5.\nd_ileqg::Float64 – convergence error norm threshold (> 0). If the maximum l2 norm of the change in nominal control over the horizon is less than d, the solver is considered to be converged. Default: 1e-2.\niter_max_ileqg::Int64 – maximum iteration number. Default: 100\nβ_ileqg::Float64 – Armijo condition number (>= 0) defining a sufficient decrease for backtracking line search. If β == 0, then any cost-to-go improvement is considered a sufficient decrease. Default: 1e-4.\nϵ_init_ileqg::Float64 – initial step size in (ϵ_min, 1] to start the backtracking line search with. If adaptive_ϵ_init is true, then this value is overridden by the solver's adaptive initialization functionality after the first iLEQG iteration. If adaptive_ϵ_init is false, the specified value of ϵ_init is used across all the iterations as the initial step size. Default:1.0.\nadaptive_ϵ_init_ileqg::Bool – if true, ϵ_init is adaptively changed based on the last step size ϵ of the previous iLEQG iteration. Default: false.\nIf the first line search iterate ϵ_init_prev in the previous iLEQG iteration is successful, then ϵ_init for the next iLEQG iteration is set to ϵ_init = ϵ_init_prev / λ so that the initial line search step increases.\nOtherwise ϵ_init = ϵ_last where ϵ_last is the line search step accepted in the previous iLEQG iteration.\nϵ_min_ileqg::Float64 – minimum value of step size ϵ to terminate the line search. When ϵ_min is reached, the last candidate nominal trajectory is accepted regardless of the Armijo condition and the current iLEQG iteration is finished. Default: 1e-6.\nf_returns_jacobian::Bool – if true, Jacobian matrices of the dynamics function are user-provided. This can reduce computation time since automatic differentiation is not used. Default: false.\n\nCross Entropy Solver Parameters\n\nμ_init::Float64 – initial value of the mean parameter μ used in the first Cross Entropy iteration. Default: 1.0.\nσ_init::Float64 – initial value of the standard deviation parameter σ used in the first Cross Entropy iteration. Default: 2.0.\nnum_samples::Int64 – number of Monte Carlo samples for the risk-sensitivity parameter θ. Default: 10.\nnum_elite::Int64 – number of elite samples. Default: 3.\niter_max::Int64 – maximum iteration number. Default: 5.\nλ::Float64 – multiplicative modification factor in (0, 1) for `μ_init and σ_init. Default: 0.5.\nuse_θ_max::Bool – if true, the maximum feasible θ found is used to perform the final iLEQG optimization instead of the optimal one. Default: false.\n\nNotes\n\nThe values of μ_init and σ_init, which may be modified during optimization, are stored internally in the solver and　carried over to the next call to solve!.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#RAT-iLQR-Solver-2","page":"Solver APIs","title":"RAT iLQR++ Solver","text":"","category":"section"},{"location":"solvers/","page":"Solver APIs","title":"Solver APIs","text":"NelderMeadBilevelOptimizationSolver","category":"page"},{"location":"solvers/#RATiLQR.NelderMeadBilevelOptimizationSolver","page":"Solver APIs","title":"RATiLQR.NelderMeadBilevelOptimizationSolver","text":"NelderMeadBilevelOptimizationSolver(kwargs...)\n\nRAT iLQR++ (i.e. Nelder-Mead Simplex Method + iLEQG) Solver.\n\nOptional Keyword Arguments\n\niLEQG Solver Parameters\n\nμ_min_ileqg::Float64 – minimum value for Hessian regularization parameter μ (> 0). Default: 1e-6.\nΔ_0_ileqg::Float64 – minimum multiplicative modification factor (> 0) for μ. Default: 2.0.\nλ_ileqg::Float64 – multiplicative modification factor in (0, 1) for line search step size ϵ. Default: 0.5.\nd_ileqg::Float64 – convergence error norm threshold (> 0). If the maximum l2 norm of the change in nominal control over the horizon is less than d, the solver is considered to be converged. Default: 1e-2.\niter_max_ileqg::Int64 – maximum iteration number. Default: 100.\nβ_ileqg::Float64 – Armijo condition number (>= 0) defining a sufficient decrease for backtracking line search. If β == 0, then any cost-to-go improvement is considered a sufficient decrease. Default: 1e-4.\nϵ_init_ileqg::Float64 – initial step size in (ϵ_min, 1] to start the backtracking line search with. If adaptive_ϵ_init is true, then this value is overridden by the solver's adaptive initialization functionality after the first iLEQG iteration. If adaptive_ϵ_init is false, the specified value of ϵ_init is used across all the iterations as the initial step size. Default:1.0.\nadaptive_ϵ_init_ileqg::Bool – if true, ϵ_init is adaptively changed based on the last step size ϵ of the previous iLEQG iteration. Default: false.\nIf the first line search iterate ϵ_init_prev in the previous iLEQG iteration is successful, then ϵ_init for the next iLEQG iteration is set to ϵ_init = ϵ_init_prev / λ so that the initial line search step increases.\nOtherwise ϵ_init = ϵ_last where ϵ_last is the line search step accepted in the previous iLEQG iteration.\nϵ_min_ileqg::Float64 – minimum value of step size ϵ to terminate the line search. When ϵ_min is reached, the last candidate nominal trajectory is accepted regardless of the Armijo condition and the current iLEQG iteration is finished. Default: 1e-6.\nf_returns_jacobian::Bool – if true, Jacobian matrices of the dynamics function are user-provided. This can reduce computation time since automatic differentiation is not used. Default: false.\n\nNelder-Mead Simplex Solver Parameters\n\nα::Float64 – reflection parameter. Default: 1.0.\nβ::Float64 – expansion parameter. Default: 2.0.\nγ::Float64 – contraction parameter. Default: 0.5.\nϵ::Float64 – convergence parameter. The algorithm is said to have convergeced  if the standard deviation of the objective values at the vertices of the simplex  is below ϵ. Default: 1e-2.\nλ::Float64 – multiplicative modification factor in (0, 1) for θ_high_init and θ_low_init, which is repeatedly applied in case the objective value is infinity until a feasible region is find. Default: 0.5.\nθ_high_init::Float64 – Initial guess for θ_high. Default: 3.0.\nθ_low_init::Float64 – Initial guess for θ_low. Default: 1e-8.\niter_max::Int64 – maximum iteration number. Default: 100.\n\nNotes\n\nThe Nelder-Mead Simplex method maintains a 1D simplex (i.e. a line segment that consists of 2 points, θ_high and θ_low) to search for the optimal risk-sensitivity parameter θ. θ_high and θ_low refer to the verteces of the simplex with the highest and the lowest objective values, respectively.\nThe initial guesses θ_high_init and θ_low_init, which may be modified during optimization, are stored internally in the solver and carried over to the next call to solve!.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#PETS-Solver","page":"Solver APIs","title":"PETS Solver","text":"","category":"section"},{"location":"solvers/","page":"Solver APIs","title":"Solver APIs","text":"CrossEntropyDirectOptimizationSolver","category":"page"},{"location":"solvers/#RATiLQR.CrossEntropyDirectOptimizationSolver","page":"Solver APIs","title":"RATiLQR.CrossEntropyDirectOptimizationSolver","text":"CrossEntropyDirectOptimizationSolver(μ_init_array::Vector{Vector{Float64}},\nΣ_init_array::Vector{Matrix{Float64}}; kwargs...)\n\nPETS Solver initialized with μ_init_array = [μ_0,...,μ_{N-1}] and Σ_init_array = [Σ_0,...,Σ_{N-1}], where the initial control distribution at time k is a Gaussian distribution Distributions.MvNormal(μ_k, Σ_k).\n\nOptional Keyword Arguments\n\nnum_control_samples::Int64 – number of Monte Carlo samples for the control trajectory. Default: 10.\nnum_trajectory_samples::Int64 – number of Monte Carlo samples for the state trajectory. Default: 10.\nnum_elite::Int64 – number of elite samples. Default: 3.\niter_max::Int64 – maximum iteration number. Default: 5.\nsmoothing_factor::Float64 – smoothing factor in (0, 1), used to update the mean and the variance of the Cross Entropy distribution for the next iteration. If smoothing_factor is 0.0, the updated distribution is independent of the previous iteration. If it is 1.0, the updated distribution is the same as the previous iteration. Default. 0.1.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#The-solve!-Function","page":"Solver APIs","title":"The solve! Function","text":"","category":"section"},{"location":"solvers/","page":"Solver APIs","title":"Solver APIs","text":"Once a problem is defined and a solver is instantiated, you can call solve! with appropriate arguments to perform optimization.","category":"page"},{"location":"solvers/","page":"Solver APIs","title":"Solver APIs","text":"solve!","category":"page"},{"location":"solvers/#RATiLQR.solve!","page":"Solver APIs","title":"RATiLQR.solve!","text":"solve!(ileqg::ILEQGSolver, problem::FiniteHorizonRiskSensitiveOptimalControlProblem,\nx_0::Vector{Float64}, u_array::Vector{Vector{Float64}}, θ::Float64, verbose=true)\n\nGiven problem, and ileqg solver, solve iLQG (if θ == 0) or iLEQG (if θ > 0) with current state x_0 and nominal control schedule u_array = [u_0, ..., u_{N-1}].\n\nReturn Values (Ordered)\n\nx_array::Vector{Vector{Float64}} – nominal state trajectory [x_0,...,x_N].\nl_array::Vector{Vector{Float64}} – nominal control schedule [l_0,...,l_{N-1}].\nL_array::Vector{Matrix{Float64}} – feedback gain schedule [L_0,...,L_{N-1}].\nvalue::Float64 – optimal cost-to-go (i.e. value) found by the solver.\nϵ_history::Vector{Float64} – history of line search step sizes used during the iLEQG iteration. Mainly for debugging purposes.\n\nNotes\n\nReturns a time-varying affine state-feedback policy π_k of the form π_k(x) = L_k(x - x_k) + l_k.\n\n\n\n\n\nsolve!(ce_solver::CrossEntropyBilevelOptimizationSolver,\nproblem::FiniteHorizonRiskSensitiveOptimalControlProblem,\nx_0::Vector{Float64}, u_array::Vector{Vector{Float64}}, rng::AbstractRNG;\nkl_bound::Float64, verbose=true, serial=false)\n\nGiven problem and ce_solver (i.e. a RAT iLQR Solver), solve distributionally robust control with current state x_0 and nominal control schedule u_array = [u_0, ..., u_{N-1}] under the KL divergence bound of kl_bound (>= 0).\n\nReturn Values (Ordered)\n\nθ_opt::Float64 – optimal risk-sensitivity parameter.\nx_array::Vector{Vector{Float64}} – nominal state trajectory [x_0,...,x_N].\nl_array::Vector{Vector{Float64}} – nominal control schedule [l_0,...,l_{N-1}].\nL_array::Vector{Matrix{Float64}} – feedback gain schedule [L_0,...,L_{N-1}].\nvalue::Float64 – optimal cost-to-go (i.e. objective value) found by the solver.\nθ_min::Float64 – minimum feasible risk-sensitivity parameter found.\nθ_max::Float64 – maximum feasible risk-sensitivity parameter found.\n\nNotes\n\nReturns a time-varying affine state-feedback policy π_k of the form π_k(x) = L_k(x - x_k) + l_k.\nIf kl_bound is 0.0, the solver reduces to iLQG.\nIf serial is true, Monte Carlo sampling of the Cross Entropy method is serialized on a single process. If false it is distributed on all the available worker processes.\n\n\n\n\n\nsolve!(nm_solver::NelderMeadBilevelOptimizationSolver,\nproblem::FiniteHorizonRiskSensitiveOptimalControlProblem, x_0::Vector{Float64},\nu_array::Vector{Vector{Float64}}; kl_bound::Float64, verbose=true)\n\nGiven problem and nm_solver (i.e. a RAT iLQR++ Solver), solve distributionally robust control with current state x_0 and nominal control schedule u_array = [u_0, ..., u_{N-1}] under the KL divergence bound of kl_bound (>= 0).\n\nReturn Values (Ordered)\n\nθ_opt::Float64 – optimal risk-sensitivity parameter.\nx_array::Vector{Vector{Float64}} – nominal state trajectory [x_0,...,x_N].\nl_array::Vector{Vector{Float64}} – nominal control schedule [l_0,...,l_{N-1}].\nL_array::Vector{Matrix{Float64}} – feedback gain schedule [L_0,...,L_{N-1}].\nvalue::Float64 – optimal cost-to-go (i.e. objective value) found by the solver.\n\nNotes\n\nReturns a time-varying affine state-feedback policy π_k of the form π_k(x) = L_k(x - x_k) + l_k.\nIf kl_bound is 0.0, the solver reduces to iLQG.\n\n\n\n\n\nsolve!(direct_solver::CrossEntropyDirectOptimizationSolver,\nproblem::FiniteHorizonGenerativeOptimalControlProblem, x_0::Vector{Float64},\nrng::AbstractRNG; use_true_model=false, verbose=true, serial=true)\n\nGiven problem and direct_solver (i.e. a PETS Solver), solve stochastic optimal control with current state x_0.\n\nReturn Values (Ordered)\n\nμ_array::Vector{Vector{Float64}} – array of means [μ_0,...,μ_{N-1}] for the final Cross Entropy distribution for the control schedule.\nΣ_array::Vector{Matrix{Float64}} – array of covariance matrices [Σ_0,...,Σ_{N-1}] for the final Cross Entropy distribution for the control schedule.\n\nNotes\n\nReturns an open-loop control policy.\nIf use_true_model is true, the solver uses the true stochastic dynamics model defined in problem.f_stochastic.\nIf serial is true, Monte Carlo sampling of the Cross Entropy method is serialized on a single process. If false it is distributed on all the available worker processes. We recommend to leave this to true as distributed processing can be slower for this algorithm.\n\n\n\n\n\n","category":"function"},{"location":"#RATiLQR.jl","page":"Home","title":"RATiLQR.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia implementation of the RAT iLQR algorithm and relevant methods for nonlinear (stochastic) optimal control. The following MPC algorithms are currently implemented:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Risk Auto-Tuning iLQR (RAT iLQR)\nRAT iLQR++\niterative Linear-Quadratic-Gaussian (iLQG)\niterative Linear-Exponential-Quadratic-Gaussian (iLEQG)\nPETS","category":"page"},{"location":"","page":"Home","title":"Home","text":"RAT iLQR is a distributionally robust nonlinear MPC via risk-sensitive optimal control. Originally presented in our paper, it locally optimizes a bilevel optimization objective with iLEQG and the Cross Entropy Method. RAT iLQR++ solves the same optimization problem, but with the Nelder-Mead Simplex Method in place of the Cross Entropy Method and generally achieves better performance and faster optimization. iLQG, iLEQG, and PETS are for benchmarking purposes and do not possess the distributional robustness property.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Important\nIf you find this package useful for your research, please cite our publication:@article{nishimura2021ratilqr,\n    author={H. {Nishimura} and N. {Mehr} and A. {Gaidon} and M. {Schwager}},\n    journal={IEEE Robotics and Automation Letters}, \n    title={RAT iLQR: A Risk Auto-Tuning Controller to Optimally Account for Stochastic Model Mismatch}, \n    year={2021},\n    doi={10.1109/LRA.2020.3048660}}","category":"page"},{"location":"#RAT-iLQR-Overview","page":"Home","title":"RAT iLQR Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A brief overview is provided in our Youtube video:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: RAT iLQR Overview)","category":"page"},{"location":"#Support","page":"Home","title":"Support","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Report bugs or request help by opening issues on the GitHub Issues Page.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This software is licensed under the MIT License.","category":"page"}]
}
