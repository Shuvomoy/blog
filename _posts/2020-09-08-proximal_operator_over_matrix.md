---
layout: post 
title: Computing proximal operator of a constrained function in Julia 
categories: [programming] 
tags: [Julia]
comments: true 
---

In this blog, we will show how to compute proximal operator of a constrained function. The [`pluto`](https://github.com/fonsp/Pluto.jl) notebook  for this blog, can be downloaded [here](https://raw.githubusercontent.com/Shuvomoy/blog/gh-pages/codes/proximal_constrained_matrix_pluto.jl).<!-- more -->

As an example we consider the function: 

$$
\begin{eqnarray*}
f(X,D) & : & =\left\Vert \Sigma-X-D\right\Vert _{F}^{2}+I_{\mathcal{P}}(X,D),
\end{eqnarray*}
$$

where $I_{\mathcal{P}}$ denotes the indicator function of the convex
set 

$$
\mathcal{P}=\{(X,D)\in\mathbf{S}^{n}\times\mathbf{S}^{n}\mid X\succeq0,D=\mathbf{diag}(d),d\geq0,d\in \mathbf{R}^{n}\}.
$$

#### Computing the proximal operator of $f$

Proximal operator $\mathbf{prox}_{\gamma f}$ for this function $f$ at $(X,D)$ is *the* optimal solution to the following convex optimization problem:

$$
\begin{equation}
\begin{array}{ll}
\textrm{minimize} & \left\Vert \Sigma-\widetilde{X}-\widetilde{D}\right\Vert _{F}^{2}+\frac{1}{2\gamma}\|\widetilde{X}-X\|_{F}^{2}+\frac{1}{2\gamma}\|\widetilde{D}-D\|_{F}^{2}\\
\textrm{subject to} & \widetilde{X}\succeq0\\
 & \widetilde{D}=\mathbf{diag}(\widetilde{d})\\
 & \widetilde{d}\geq0,
\end{array}
\end{equation}
$$

where $$\widetilde{X}\in\mathbf{S}_{+}^{n},$$ and $$\widetilde{d}\in \mathbf{R}_{+}^{n}$$
(*i.e.*, $\widetilde{D}=\mathbf{diag}(\widetilde{d}$)) are the
optimization variables. 

Now we solve this optimization problem using `Julia`. We will use the package `Convex` and `COSMO`, both open source `Julia` packages.

#### Load the packages
First, we load the packages. If the packages are not installed we can install them by running the following commands in `Julia` REPL.

```julia
using Pkg
Pkg.add("Convex")
Pkg.add("COSMO")
```


```julia
# Load the packages
using Convex
using LinearAlgebra
using COSMO
```

#### Solver function
Let us write the solver function that is going to solve the optimization problem that we described above.


```julia
# put everything in a function
function prox_over_matrix(Σ, γ, X, d)
	
  # This functions takes the input data Σ, γ, X, d and evaluates 
  # the proximal operator of the function f at the point (X,D)
	
  # Data extraction
  # ---------------
  n = length(d) # dimension of the problem
  D = diagm(d) # creates the diagonal matrix D that embed
	
  # Create the variables
  #  --------------------
  X_tl = Convex.Semidefinite(n) # Here Semidefinite(n) encodes that
  # X_tl ≡ ̃X is a positive semidefinite matrix
  d_tl = Convex.Variable(n) # d_tl ≡ ̃d
  D_tl = diagm(d_tl) # Create the diagonal matrix ̃D from ̃d
	
  # Create terms of the objective function, which we write down
  #  in three parts
  #  ----------------------------------------------------------
  t1 = square(norm(Σ - X_tl - D_tl,2))
  t2 = square(norm(X-X_tl,2))
  t3 = square(norm(D-D_tl,2))
	
  # Create objective
  # ----------------
  objective = t1 + (1/(2*γ))*(t2 + t3) # the objective to be minimized
	
  # create the problem instance
  # ---------------------------
  problem = Convex.minimize(objective, [d_tl >= 0])
	
  # set the solver
  # --------------
  solver = () -> COSMO.Optimizer(verbose=false)
	
  # solve the problem
  # -----------------
  Convex.solve!(problem, solver)
	
  # get the optimal solution
  # ------------------------	
  X_sol = X_tl.value
  d_sol = d_tl.value
	
  # return the output	
  return X_sol, d_sol
	
end

```




    prox_over_matrix (generic function with 1 method)



#### Create data to test
We create the data now to test the function we just wrote.


```julia
n = 10
Σ1 = randn(n,n)
Σ = Σ1'*Σ1
X = randn(n,n)
D = randn(n,n)
M = 1
x = [D[i,i] for i in 1:n]
γ = 1
```




    1



#### Test the function
We test the function now to see if the function `prox_over_matrix` works as expected!


```julia
X_o, x_o = prox_over_matrix(Σ, γ, X, x)
```




    ([1.114628361135871 -0.5441043401736868 … -0.30455059149447206 0.2791254733643821; -0.5441043386198742 7.667452388395437 … -1.5039468012639492 1.8816628878704544; … ; -0.3045505866013101 -1.5039467875342283 … 2.3143161293218837 0.3552207776486801; 0.2791254789942864 1.8816628953143448 … 0.3552207559140353 8.947398913460479], [5.02063531600652; 10.46170037768896; … ; 10.764347294894044; 1.5708997917722958])




```julia
X_o
```




    10×10 Array{Float64,2}:
      1.11463   -0.544104   0.112517   …   0.170839  -0.304551    0.279125
     -0.544104   7.66745    0.441084      -0.500284  -1.50395     1.88166
      0.112517   0.441084   1.24836        0.524455  -1.04818    -0.0976843
     -0.182576  -1.30921    1.47056        1.56152   -0.945758    0.298069
     -0.509397   0.365734   0.765196       0.6746     0.0371845   2.41057
      0.493616  -0.220375   1.395      …   3.07552   -1.88673    -4.07402
      0.642302  -3.05051   -0.599022      -0.341396   1.06463    -0.0770057
      0.170839  -0.500284   0.524455       2.07687   -0.809619   -1.4246
     -0.304551  -1.50395   -1.04818       -0.809619   2.31432     0.355221
      0.279125   1.88166   -0.0976842     -1.4246     0.355221    8.9474




```julia
x_o
```




    10×1 Array{Float64,2}:
      5.02063531600652
     10.46170037768896
     10.804522251086823
     10.717881583087392
     10.16775724289986
     13.454876703952081
      9.89391268460101
      8.882205790976348
     10.764347294894044
      1.5708997917722958


