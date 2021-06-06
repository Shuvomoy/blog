### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 43c6f79e-f206-11ea-0082-e3fd10d20eae
begin
	# Load the packages
	using Convex
	using JuMP
	using LinearAlgebra
	using COSMO
	using SCS
end

# ╔═╡ db7ebbc0-f1d7-11ea-0eb0-2d7b749aef2b
md"""
In this blog, we will show how to compute proximal operator of a constrained function. As an example we consider the function: 

$\begin{eqnarray*}
f(X,D) & : & =\left\Vert \Sigma-X-D\right\Vert _{F}^{2}+I_{\mathcal{P}}(X,D),
\end{eqnarray*}$

where $I_{\mathcal{P}}$ denotes the indicator function of the convex
set 

$\mathcal{P}=\{(X,D)\in\mathbf{S}^{n}\times\mathbf{S}^{n}\mid X\succeq0,D=\mathbf{diag}(d),d\geq0,d\in \mathbf{R}^{n}\}.$

#### Computing the proximal operator of $f$

Proximal operator $\mathbf{prox}_{\gamma f}$ for this function $f$ at $(X,D)$ is *the* optimal solution to the following convex optimization problem:

$\begin{equation}
\begin{array}{ll}
\textrm{minimize} & \left\Vert \Sigma-\widetilde{X}-\widetilde{D}\right\Vert _{F}^{2}+\frac{1}{2\gamma}\|\widetilde{X}-X\|_{F}^{2}+\frac{1}{2\gamma}\|\widetilde{D}-D\|_{F}^{2}\\
\textrm{subject to} & \widetilde{X}\succeq0\\
 & \widetilde{D}=\mathbf{diag}(\widetilde{d})\\
 & \widetilde{d}\geq0,
\end{array}
\end{equation}$

where $\widetilde{X}\in\mathbf{S}_{+}^{n},$ and $\widetilde{d}\in \mathbf{R}_{+}^{n}$
(*i.e.*, $\widetilde{D}=\mathbf{diag}(\widetilde{d}$)) are the
optimization variables. 

Now we solve this optimization problem using `Julia`. We will use the package `Convex` and `COSMO`, both open source `Julia` packages.
"""

# ╔═╡ 1669a140-f1d9-11ea-37c4-f5f9e390aff2
md"""
#### Load the packages
First, we load the packages. If the packages are not installed we can install them by running the following commands in `Julia` REPL.

```julia
using Pkg
Pkg.add("Convex")
Pkg.add("COSMO")
```
"""

# ╔═╡ 69814c20-f1d9-11ea-0bde-c9700dd85600
md"""
#### Solver function
Let us write the solver function that is going to solve the optimization problem that we described above.
"""

# ╔═╡ c261d490-f1d9-11ea-1f3f-e93b211d6d44
# put everything in a function
function prox_over_matrix_Convex(Σ, γ, X, d)
	
  # This functions takes the input data Σ, γ, X, d and evaluates the proximal operator of the function f at the point (X,d)
	
  # Data extraction
  # ---------------
  n = length(d) # dimension of the problem
  D = diagm(d) # creates the diagonal matrix D that embed
	
  # Create the variables
  #  --------------------
  X_tl_cvx = Convex.Semidefinite(n) # Here Semidefinite(n) encodes that X_tl ≡ ̃X is a positive semidefinite matrix
  d_tl_cvx = Convex.Variable(n) # d_tl ≡ ̃d
  D_tl_cvx = diagm(d_tl_cvx) # Create the diagonal matrix ̃D from ̃d
	
  # Create terms of the objective function, which we write down in three parts
  #  --------------------------------------------------------------------------
  t1 = square(norm2(Σ - X_tl_cvx - D_tl_cvx)) # norm2 function computes the Frobenius 
  t2 = square(norm2(X-X_tl_cvx))          # norm for a matrix in Convex.jl
  t3 = square(norm2(D-D_tl_cvx))
	
  # Create objective
  # ----------------
  objective = t1 + (1/(2*γ))*(t2 + t3) # the objective to be minimized
	
  # create the problem instance
  # ---------------------------
  problem = Convex.minimize(objective, [d_tl_cvx >= 0])
	
  # set the solver
  # ------------------
  # solver = () -> SCS.Optimizer(verbose=false)
	
  # solve the problem
  # -----------------
  Convex.solve!(problem, SCS.Optimizer(verbose=false))
	
  # get the optimal solution
  # ------------------------	
  X_sol = X_tl_cvx.value
  d_sol = d_tl_cvx.value
	
  # return the output	
  return X_sol, d_sol
	
end


# ╔═╡ d225b1b0-fa1e-11ea-3950-d1011df616b7
md"""
We can write the same function using `JuMP` as well.
"""

# ╔═╡ a69e5a70-f207-11ea-1845-3b2e3616976e
md"""
#### Create data to test
We create the data now to test the function we just wrote.
"""

# ╔═╡ b05f9b4e-f207-11ea-01f0-e561d5a3f0cc
begin
# Create data
# Input
	n = 10
	Σ1 = randn(n,n)
	Σ = Σ1'*Σ1
	X = randn(n,n)
	D = randn(n,n)
	M = 1
	x = [D[i,i] for i in 1:n]
	γ = 1
end

# ╔═╡ ded20490-fa1e-11ea-373d-63c620008af0
# put everything in a function
function prox_over_matrix_JuMP(Σ, γ, X, d)

	# This functions takes the input data Σ, γ, X, d and evaluates the proximal operator of the function f at the point (X,d)

	n = length(d)
	
	prox_model = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false))

	@variables( prox_model,
	begin
		d_tl[1:n] >= 0
		X_tl[1:n, 1:n], PSD
	end
	)

	t_1 = vec(Σ - X_tl - diagm(d_tl))
	t_2 = vec(X_tl-X)
	t_3 = vec(diagm(d_tl)-D)
	obj = t_1'*t_1 + ((1/(2*γ))*(t_2'*t_2 + t_3'*t_3))

	@objective(prox_model, Min, obj)

	optimize!(prox_model)

	obj_val = objective_value(prox_model)
	X_sol = value.(X_tl)
	d_sol = value.(d_tl)

	return X_sol, d_sol

end

# ╔═╡ 40a00df0-f22e-11ea-1383-bfa043625862
md"""
#### Test the function
We test the function now to see if the function `prox_over_matrix` works as expected!
"""

# ╔═╡ 7f811e80-fa1f-11ea-2749-6d6cc64ef668
@time prox_over_matrix_JuMP(Σ, γ, X, d)

# ╔═╡ ded8c5e0-fa1f-11ea-2e2d-85164c12a199
prox_over_matrix_Convex(Σ, γ, X, d)

# ╔═╡ Cell order:
# ╠═db7ebbc0-f1d7-11ea-0eb0-2d7b749aef2b
# ╠═1669a140-f1d9-11ea-37c4-f5f9e390aff2
# ╠═43c6f79e-f206-11ea-0082-e3fd10d20eae
# ╠═69814c20-f1d9-11ea-0bde-c9700dd85600
# ╠═c261d490-f1d9-11ea-1f3f-e93b211d6d44
# ╠═d225b1b0-fa1e-11ea-3950-d1011df616b7
# ╠═ded20490-fa1e-11ea-373d-63c620008af0
# ╠═a69e5a70-f207-11ea-1845-3b2e3616976e
# ╠═b05f9b4e-f207-11ea-01f0-e561d5a3f0cc
# ╠═40a00df0-f22e-11ea-1383-bfa043625862
# ╠═7f811e80-fa1f-11ea-2749-6d6cc64ef668
# ╠═ded8c5e0-fa1f-11ea-2e2d-85164c12a199
