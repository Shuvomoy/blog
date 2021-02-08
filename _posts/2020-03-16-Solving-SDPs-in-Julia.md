---
layout: post 
title: Solving semidefinite programming problems in Julia
categories: [programming] 
tags: [Julia]
comments: true 
---

In this blog, we discuss how to solve semidefinite programs (SDPs) in ``Julia`` using ``Convex.jl``. We consider optimization problem of the form: <!-- more -->
$$
\begin{align*}
\begin{array}{ll}
\textrm{minimize} & \mathbf{trace}(CX)\\
\textrm{subject to} & \mathbf{trace}(A_{i}X)=b_{i},\\
 & X\succeq0,
\end{array} & i=1,\ldots,m
\end{align*}
$$
 where $X\in\mathbf{S}^{n}$ is the decision variable, and each of the $A_{i}$ matrices and $C$ are also in $\mathbf{S}^{n}$. By the notation $\mathbf{S}^{n}$, we denote the set of all symmetric $n\times n$ matrices.  

First, we load the necessary `Julia` packages.


```julia
using SCS, COSMO, MosekTools, JuMP, LinearAlgebra

using BenchmarkTools
```

Let us create data $A,C,b$ randomly.


```julia
function random_mat_create(n)
    # this function creates a symmetric n×n matrix
    A = randn(n,n)
    A = A'*A
    A = (A+A')/2
    return A
end
```

Here is the data generation process, please change it to your need. 


```julia
n = 10
m = 20
# set of all data matrices A_i
# the data matrix A = [A1 A2 A3 ....]
A = zeros(n, m*n) 
b = zeros(m)
# just ensuring our problem is feasible
X_test = rand(n,n)
X_test = X_test'*X_test
X_test = (X_test+X_test')/2
for i in 1:m
    A[:, (i-1)*n+1:i*n] .= random_mat_create(n)
    b[i] = tr(A[:, (i-1)*n+1:i*n]*X_test)
end
C = abs.(random_mat_create(n))
```

The following function solves the underlying SDP.


```julia

function solve_SDP(A, b, C; solver_name=:COSMO)

# Create variable
    if solver_name == :COSMO
        model = Model(with_optimizer(COSMO.Optimizer))
    elseif solver_name == :Mosek
        model = Model(optimizer_with_attributes(Mosek.Optimizer))
    end

    set_silent(model)

    @variable(model, X[1:n, 1:n], PSD)


    @objective(model, Min, tr(C * X));
    for j in 1:m
        A_j = A[:, (j - 1) * n + 1:j * n]
        @constraint(model, tr(A_j * X) == b[j])
    end

    optimize!(model)

    status = JuMP.termination_status(model)
    X_sol = JuMP.value.(X)
    obj_value = JuMP.objective_value(model)

    return status, X_sol, obj_value

end
```

Time to solve the problem.


```julia
status, X_sol, obj_value = solve_SDP(A, b, C; solver_name=:Mosek)
```


    out: (MathOptInterface.OPTIMAL, [2.907044311952373 1.7130367276575142 … -0.056145513617222656 3.0230926674218024; 1.7130367276575142 3.419039624378557 … 1.0871948703965775 2.0577919984154334; … ; -0.056145513617222656 1.0871948703965775 … 0.7127834057861842 0.5195987956934747; 3.0230926674218024 2.0577919984154334 … 0.5195987956934747 4.234108180247669], 939.2696385581793)

Lets see which solver is faster, `COSMO` or `Mosek`.


```julia
b1 = @benchmark solve_SDP(A, b, C; solver_name=:COSMO)

println("benchmark for COSMO")
println("*************************")
io = IOBuffer()
show(io, "text/plain", b1)
s = String(take!(io))
println(s)
```


    benchmark for COSMO
    *************************
    BenchmarkTools.Trial: 
      memory estimate:  4.33 MiB
      allocs estimate:  35786
      --------------
      minimum time:     36.964 ms (0.00% GC)
      median time:      40.552 ms (0.00% GC)
      mean time:        40.787 ms (1.14% GC)
      maximum time:     46.920 ms (9.97% GC)
      --------------
      samples:          123
      evals/sample:     1


```julia
b2 = @benchmark solve_SDP(A, b, C; solver_name=:Mosek)

println("benchmark for Mosek")
println("***************************")
io = IOBuffer()
show(io, "text/plain", b2)
s = String(take!(io))
println(s)
```


    benchmark for Mosek
    ***************************
    BenchmarkTools.Trial: 
      memory estimate:  3.81 MiB
      allocs estimate:  32015
      --------------
      minimum time:     6.647 ms (0.00% GC)
      median time:      7.524 ms (0.00% GC)
      mean time:        8.374 ms (5.00% GC)
      maximum time:     19.410 ms (25.35% GC)
      --------------
      samples:          597
      evals/sample:     1

So, on average, `Mosek` seems to be 5 times faster than `COSMO`. 

