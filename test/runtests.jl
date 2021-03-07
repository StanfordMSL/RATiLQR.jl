#///////////////////////////////////////
#// File Name: runtests.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/11/03
#// Description: Test script for RATiLQR
#///////////////////////////////////////

using Test

using Distributed
if nprocs() < 2
    addprocs(1)
end

@everywhere using RATiLQR

@testset "RATiLQR Unit Tests" begin
@info "Executing iLEQG Test"
include("ileqg_test.jl")
@info "Executing RAT iLQR Test"
include("cross_entropy_bilevel_optimization_test.jl")
@info "Executing PETS Test"
include("pets_test.jl")
end
