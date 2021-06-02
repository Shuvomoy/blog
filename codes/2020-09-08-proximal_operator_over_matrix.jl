
using Pkg
Pkg.add("Convex")
Pkg.add("COSMO")


## Load the packages
using Convex
using LinearAlgebra
using COSMO
using JuMP
using MosekTools
using SCS


# put everything in a function: first implementatino is using Convex.jl
function prox_PRS_fam_cvxjl(Σ, M, γ, X, d) #(Σ::A, M::R, γ::R, X::A, d::V) where {R <: Real, A <: AbstractMatrix{R}, V <:  AbstractVector{R}} # For now M is not used, may use it in a future version

  # This functions takes the input data Σ, γ, X, d and evaluates
  # the proximal operator of the function f at the point (X,D)

  # Data extraction
  # ---------------
  n = length(d) # dimension of the problem
  # println("*****************************")
  # println(size(d))
  # println("the value of d is = ", d)
  # println("the type of d is", typeof(d))
  D = LinearAlgebra.diagm(d) # creates the diagonal matrix D that embed

  # Create the variables
  #  --------------------
  X_tl = Convex.Semidefinite(n) # Here Semidefinite(n) encodes that
  # X_tl ≡ ̃X is a positive semidefinite matrix
  d_tl = Convex.Variable(n) # d_tl ≡ ̃d
  D_tl = diagm(d_tl) # Create the diagonal matrix ̃D from ̃d

  # Create terms of the objective function, which we write down
  #  in three parts
  #  ----------------------------------------------------------
  t1 = square(norm2(Σ - X_tl - D_tl)) # norm2 computes Frobenius norm in Convex.jl
  t2 = square(norm2(X-X_tl))
  t3 = square(norm2(D-D_tl))

  # Create objective
  # ----------------
  objective = t1 + (1/(2*γ))*(t2 + t3) # the objective to be minimized

  # create the problem instance
  # ---------------------------
  convex_problem = Convex.minimize(objective, [d_tl >= 0, Σ - D_tl  in :SDP])

  # set the solver
  # --------------
  convex_solver = () -> SCS.Optimizer(verbose=false)

  # solve the problem
  # -----------------
  Convex.solve!(convex_problem, convex_solver)

  # get the optimal solution
  # ------------------------
  X_sol = X_tl.value
  d_sol = d_tl.value
  # println("d_sol = ", d_sol)

  # return the output
  return X_sol, vec(d_sol)

end


## put everything in a function (implementation using JuMP)
function prox_PRS_fam_JuMP(Σ, M, γ, X, d; X_tl_sv = nothing, d_tl_sv = nothing, warm_start = false)

	# This functions takes the input data Σ, γ, X, d and evaluates the proximal operator of the function f at the point (X,d)

	n = length(d)

	# prox_model = JuMP.Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => false))

	prox_model = JuMP.Model(optimizer_with_attributes(Mosek.Optimizer))


	@variables( prox_model,
	begin
		d_tl[1:n] >= 0
		X_tl[1:n, 1:n], PSD
	end
	)

	if warm_start == true
		println("warm start enabled")
		# Set warm-start
		set_start_value.(X_tl, X_tl_sv) # Warm start
		set_start_value.(d_tl, d_tl_sv) # Warm start
	    println("norm difference is = ", norm(start_value.(X_tl) - X_tl_sv))
	end



  t_1 = vec(Σ - X_tl - diagm(d_tl))
	t_2 = vec(X_tl-X)
	t_3 = vec(diagm(d_tl)-diagm(d))
	obj = X_tl[n,n] + t_1'*t_1 + ((1/(2*γ))*(t_2'*t_2 + t_3'*t_3))

	@objective(prox_model, Min, obj)

	@constraints(prox_model, begin
		Symmetric(Σ - diagm(d_tl)) in PSDCone()
	end)

	# set_silent(prox_model)

	JuMP.optimize!(prox_model)

	# obj_val = JuMP.objective_value(prox_model)
	X_sol = JuMP.value.(X_tl)
	d_sol = JuMP.value.(d_tl)

	return X_sol, d_sol

end


## All the parameters
n = 10
Σ1 = randn(n,n)
Σ = Σ1'*Σ1
X = randn(n,n)
d = randn(n)
M = 1
γ = 1


## Time to run the code
@time X1, d1 = prox_PRS_fam_cvxjl(Σ, M, γ, X, d)

@time X2, d2 = prox_PRS_fam_JuMP(Σ, M, γ, X, d)

## Test for warmstarting

X_tl_sv = X2
d_tl_sv = d2

@time X2, d2 = prox_PRS_fam_JuMP(Σ, M, γ, X, d; X_tl_sv = X2, d_tl_sv = d2, warm_start = true)


  2.263578 seconds (11.69 k allocations: 1.033 MiB)
  1.613454 seconds (18.60 k allocations: 2.731 MiB)
  warm start enabled
  norm difference is = 0.0
  1.643022 seconds (19.30 k allocations: 2.764 MiB)


using Weave
cd("C:\\Users\\shuvo\\Google Drive\\GitHub\\blog\\codes") # directory that contains the .jmd file
tangle("2020-09-08-proximal_operator_over_matrix.jmd", informat = "markdown") # convert the .jmd file into a .jl file that will contain the code

