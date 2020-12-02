---
layout: post 
title: Running parallel Julia code on MIT Supercloud
categories: [programming] 
tags: [Julia]
comments: true 
---

In this blog, we will discuss how to run  parallel `Julia` code on MIT Supercloud. For the basics on how to run `Julia` code in MIT Supercloud, please see my previous blog post [here](https://shuvomoy.github.io/blog/programming/2020/01/24/Running-Julia-code-on-MIT-supercloud.html). <!-- more -->

As an example for running parallel code, we will consider solving sparse regression problem. 

**Sparse regression problem.** The sparse regression problem (also known as regressor selection problem) is concerned with approximating a vector $b\in\mathbf{R}^{m}$ with a linear combination of at most $k$ columns of a matrix $A\in\mathbf{R}^{m\times d}$ with bounded coefficients. The problem can be written as the following optimization problem
$$
\begin{equation}
\begin{array}{ll}
\textrm{minimize} & \|Ax-b\|_{2}^{2}+\frac{\beta}{2}\|x\|^{2}\\
\textrm{subject to} & \mathbf{card}(x)\leq k\\
 & \|x\|_{\infty}\leq M,
\end{array}
\end{equation}
$$

where $x\in\mathbf{R}^{d}$ is the decision variable, and $A\in\mathbf{R}^{m\times d},b\in\mathbf{R}^{m},$ and $M>0$ are problem data.

**Setup.** We have a nonconvex optimization problem (sparse regression problem) and an algorithm [`NExOS`](https://github.com/Shuvomoy/NExOS.jl) that is able to compute a locally optimal solution of this problem under [certain regularity conditions](https://arxiv.org/pdf/2011.04552.pdf). Depending on different initializations, `NExOS` will provide us with different locally optimal solutions. So, naturally we can think of initializing our algorithm with different random points, observe the solutions provided by `NExOS` for different initializations and then pick the locally optimal solution that corresponds to the smallest objective value. We can achieve this task by using parallelization techniques provided in `Julia`. For this blog, we will consider `pmap`.

**Julia Code.** The code for the `julia` file is given below, please save it in a text file and name it `pmap_julia.jl`.

```julia 
using ClusterManagers,Distributed

# Add in the cores allocated by the scheduler as workers
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
print("Added workers: ")
println(nworkers())
    
using Random, NExOS, ProximalOperators, Distributions # load it centrally

@everywhere using Random, NExOS, ProximalOperators, Distributions # load it on each process

# create the array of random initial points

n = 20

M = 1

function random_z(n, M)
    uniDst = Uniform(-M, M)
    x = rand(uniDst,n)
    z = x
    return z
end

## create array of random z s

function create_array_z_randomized(n, N_random_points, M)
    # this array has its first column all zeros and the rest are uniformly distrubted over [-M,M]
    array_z_randomized = zeros(n, N_random_points)
    for i in 2:N_random_points
        array_z_randomized[:,i] = random_z(n, M)
    end
    return array_z_randomized
end

## necessary info to generate the data

bigM = 1e99 # this is the bigM, which is a global variable we are not gonna change

N_random_points = 100

array_z_randomized = create_array_z_randomized(n, N_random_points, M)

array_z_randomized_formatted = [array_z_randomized[:,i] for i in 1:N_random_points]

# we write the distributed part that is loaded everywhere on the processes we created

@everywhere begin

    ## create data
    m = 10
    n = 20
    A = randn(m,n)
    b = randn(m)
    M = 1
    k = convert(Int64, round(m/3))
    beta = 10^-10


    # Let us put everything in a function, which we are going to use later for parallel implementation.

    C = NExOS.SparseSet(M, k) # Create the set
    f = ProximalOperators.LeastSquares(A, b, iterative = true) # Create the function
    settings = NExOS.Settings(μ_max = 2, μ_min = 1e-8, μ_mult_fact = 0.5, verbose = false, freq = 250, γ_updt_rule = :adaptive, β = beta) # settings

    #function sparse_reg_NExOS(z0, C, f, settings) # z0 is the initial point
    function sparse_reg_NExOS(z0)
        
        problem = NExOS.Problem(f, C, settings.β, z0) # problem instance
        state_final = NExOS.solve!(problem, settings)
        return state_final

    end

end # end the begin block

using BenchmarkTools 

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 25 # The number of seconds budgeted for the benchmarking process. The trial will terminate if this time is exceeded (regardless of samples), but at least one sample will always be taken.

# serial implementation
b1 = @benchmark output_map =  map(sparse_reg_NExOS, array_z_randomized_formatted)

println("benchmark for serial code")
println("*************************")
io = IOBuffer()
show(io, "text/plain", b1)
s = String(take!(io))
println(s)

# parallel implementation
b2 = @benchmark output_pmap = pmap(sparse_reg_NExOS, array_z_randomized_formatted)

println("benchmark for parallel code")
println("***************************")
io = IOBuffer()
show(io, "text/plain", b2)
s = String(take!(io))
println(s)

```

## Shell script to submit the job

Now we are going to create a shell script that will be used to submit the job. The code for the shell script is below. Please save it in a text file, and name it `run_pmap_julia.sh`. In the code, `SBATCH -o pmap_julia.log-%j` indicates the name of the file where the output is written, and `SBATCH -n 14` indicates the number of cores or cpus allocated to the job. 

```julia 
#!/bin/bash

# Slurm sbatch options
#SBATCH -o pmap_julia.log-%j
#SBATCH -n 14

# Initialize the module command first source
source /etc/profile

# Load Julia Module
module load julia/1.5.2
 
# Load Gurobi Module
# module load gurobi/gurobi-811            
 
# Call your script as you would from the command line
# Call your script as you would from the command line
julia pmap_julia.jl
```

## Submitting the job

Now log in to MIT supercloud, copy the files created above to your working directory, and run the following command. 

```
LLsub run_pmap_julia.sh
```

That's it! Once the computation is done, we see from the output log file that parallelization has decreased the computation time significantly. 

```julia 
benchmark for serial code
*************************
BenchmarkTools.Trial: 
  memory estimate:  25.59 GiB
  allocs estimate:  205147791
  --------------
  minimum time:     12.589 s (6.61% GC)
  median time:      12.615 s (6.39% GC)
  mean time:        12.615 s (6.39% GC)
  maximum time:     12.641 s (6.17% GC)
  --------------
  samples:          2
  evals/sample:     1
    
benchmark for parallel code
***************************
BenchmarkTools.Trial: 
  memory estimate:  449.75 KiB
  allocs estimate:  10208
  --------------
  minimum time:     1.471 s (0.00% GC)
  median time:      1.504 s (0.00% GC)
  mean time:        1.516 s (0.00% GC)
  maximum time:     1.569 s (0.00% GC)
  --------------
  samples:          17
  evals/sample:     1
```

 

 