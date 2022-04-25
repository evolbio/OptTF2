module OptTF
#using OptTF_settings
using Symbolics, Combinatorics, Parameters, JLD2, Plots, Printf, DifferentialEquations,
	Distributions, DiffEqFlux, GalacticOptim
export generate_tf_activation_f, calc_v, set_r, mma, fit_diffeq

# CODE FOR NODE NOT COMPLETE, USE ODE ONLY UNTIL NODE COMPLETED

####################################################################
# colors, see MMAColors.jl in my private modules

mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

@with_kw struct loss_args
	u0
	prob				# problem to send to solve
	predict				# function that calls correct solve to make prediction
	data				# target data
	tsteps				# time steps for training data
	w					# weights for sequential fitting of time series
end

# When fit only to training data subset, then need full time period info
struct all_time
	prob_all			# problem for full data set	
	tsteps_all			# steps for full data set
end

make_loss_args_all(L::loss_args, A::all_time) =
					loss_args(L; prob=A.prob_all, tsteps=A.tsteps_all,
					w=ones(length(L.data[:,1]),length(L.data[1,:])))

# Generate function to calculate promoter activation following eq S6 of Marbach10_SI.pdf
# Use as f=generate_tf_activation_f(s), in which s is number of input TF binding sites
# in f(v,a,r), lengths v,a,r are s, N, N, augment N-(s+1) for r to N as set_r(r,s)
# call as f(calc_v(y,k,h), a, set_r(r,s))
# s is S.tf_in_num

function generate_tf_activation_f(s; print_def=false)
	N=2^s	# size of powerset
	@variables v[1:s], a[1:N], r[1:N]	# r is rho for cooperativity weighting
	# first s+1 terms are not cross products, no cooperativity
	rr = vcat(ones(s+1),collect(r[s+2:N]))
	vs = collect(powerset(v))		# all possible combinations of binding on
	vs[1] = [1.0]					# empty set => 1.0 weighting
	numer = sum([a[i]*rr[i]*prod(vs[i]) for i in 1:N])
	denom = sum([rr[i]*prod(vs[i]) for i in 1:N])
	to_compute=numer/denom
	if print_def
		println("numer = ", numer)
		println("denom = ", denom)
	end
	# in call, lengths v,a,r are n, N, N-(s+1), with r augmented to N as vcat(ones(s+1),r)
	# expression=Val{false} means compiled function returned instead of symbols
	#   required otherwise cannot use in ode! call
	f_expr = build_function(to_compute, v, a, r; expression=Val{false})
	return eval(f_expr)
end

# y is tf concentration, k is dissociation constant, h is hill coefficient
# need to select y values for input indices for ith gene as getindex(y,S.tf_in[i])
# for full array of concentrations, y, and particular gene i
# so call for ith gene is with S.tf_in_num tf inputs at promoter as
# calc_v(getindex(y,S.tf_in[i]),P.k[i],P.h[i])
calc_v(y, k, h) = [(y[j]/k[j])^h[j] for j in 1:length(h)]
set_r(r,s) = vcat(ones(s+1),r)

# get full array of f values, for f=generate_tf_activation_f(S.tf_in_num) and
# P = ode_parse_p(p,S) and y as full array of TF concentrations, S as settings
calc_f(f,P,y,S) = 
	[f(calc_v(getindex(y,S.tf_in[i]),P.k[i],P.h[i]),P.a[i],set_r(P.r[i],S.tf_in_num))
			for i in 1:S.n]

# m message, _a growth, _d decay, k dissociation, h hill coeff, r cooperativity
function ode_parse_p(p,S)
	n = S.n
	s = S.tf_in_num
	N = 2^s
	ddim = S.opt_dummy_u0 ? 2*n - S.m : 0
	u0_dum = @view p[1:ddim]				# 2n-m or 0
	m_a = @view p[ddim+1:ddim+n]			# n
	m_d = @view p[ddim+n+1:ddim+2n]			# n
	p_a = @view p[ddim+2n+1:ddim+3n]		# n
	p_d = @view p[ddim+3n+1:ddim+4n]		# n
	
	b = ddim+4n
	k = [@view p[b+1+(i-1)*s:b+i*s] for i in 1:n]		# ns
	b = ddim+4n+n*s
	h = [@view p[b+1+(i-1)*s:b+i*s] for i in 1:n]		# ns
	b = ddim+4n+2*n*s
	a = [@view p[b+1+(i-1)*N:b+i*N] for i in 1:n]		# nN
	b = ddim+4n+2*n*s+n*N
	ri = N - (s+1)
	r = [@view p[b+1+(i-1)*ri:b+i*ri] for i in 1:n]		# n(N-(s+1))
	@assert length(p) == b + n*ri
	# return a named tuple
	(;u0_dum, m_a, m_d, p_a, p_d, k, h, a, r)
end

function ode_num_param(S)
	n = S.n
	s = S.tf_in_num
	@assert n >= S.m
	@assert s <= n
	N = 2^s
	ri = N - (s+1)
	ddim = S.opt_dummy_u0 ? 2*n - S.m : 0
	return ddim+4n+2*n*s+n*N+n*ri
end


# modified from FitODE.jl
# m is number of protein dimensions for target pattern
# n is number of protein dimensions
# u_init for proteins taken from target data
# for ode w/mRNA, u_init for all mRNA and for dummy proteins random or optimized
# NODE has only proteins
# TF ODE has n mRNA plus n proteins
function setup_diffeq_func(S)
	ddim_p = S.n - S.m		# dummy protein dimensions
	ddim_all = ddim_p + S.n	# dummy proteins plus dummy mRNA
	# activation function to create nonlinearity, identity is no change
	activate = if (S.activate == 1) identity elseif (S.activate == 2) tanh
						elseif (S.activate == 3) sigmoid else swish end
	# If optimizing initial conditions for dummy dimensions, then for initial condition u0,
	# dummy dimensions are first entries of p
	predict_node_dummy(p, prob, u_init) =
	  		Array(prob(vcat(u_init,p[1:ddim]), p[ddim+1:end]))
	predict_node_nodummy(p, prob, u_init) = Array(prob(u_init, p))
	predict_ode_dummy(p, prob, u_init) =
			solve(prob, S.solver, u0=vcat(p[1:S.n],u_init,p[S.n+1:ddim_all]), p=p[ddim_all+1:end])
	predict_ode_nodummy(p, prob, u_init) = solve(prob, S.solver, p=p)

	# For NODE, many simple options to build alternative network architecture, see SciML docs
	if S.use_node
		dudt = FastChain(FastDense(S.n, S.layer_size, activate), FastDense(S.layer_size, S.n))
		ode! = nothing
		predict = S.opt_dummy_u0 ?
			predict_node_dummy :
			predict_node_nodummy
	else
		dudt = nothing
		function ode!(du, u, p, t, S, f)
			n = S.n
			u_tf = @view u[1:n]
			u_pr = @view u[n+1:2n]
			P = ode_parse_p(p,S)
			f_val = calc_f(f,P,u_pr,S)
			du[1:n] .= P.m_a .* f_val .- P.m_d .* u_tf			# mRNA level
			du[n+1:2n] .= P.p_a .* u_tf .- P.p_d .* u_pr		# protein level
		end
		predict = S.opt_dummy_u0 ?
			predict_ode_dummy :
			predict_ode_nodummy
	end
	return dudt, ode!, predict
end

function callback(p, loss_val, S, L, pred; doplot = true, show_lines = true)
	# printing gradient takes calculation time, turn off may yield speedup
	if (S.print_grad)
		grad = gradient(p->loss(p,S,L)[1], p)[1]
		gnorm = sqrt(sum(abs2, grad))
		println(@sprintf("%5.3e; %5.3e", loss_val, gnorm))
	else
		display(loss_val)
	end
	if doplot
		# plot current prediction against data
		len = length(pred[1,:])
		dim = length(pred[:,1])
		ts = L.tsteps[1:len]
		plt = plot(size=(600,400*dim), layout=(dim,1))
		plot_type! = if show_lines plot! else scatter! end
		for i in 1:dim
			plot_type!(ts, L.data[i,1:len], label = "", color=mma[1], subplot=i)
			plot_type!(plt, ts, pred[i,:], label = "", color=mma[2], subplot=i)
		end
		display(plot(plt))
  	end
  	return false
end

function loss(p, S, L)
	pred_all = L.predict(p, L.prob, L.u0)
	pred = S.use_node ? pred_all[1:S.m,:] : pred_all[S.n+1:S.n+S.m]	# proteins
	pred_length = length(pred[1,:])
	if pred_length != length(L.w[1,:]) println("Mismatch") end
	loss = sum(abs2, L.w[:,1:pred_length] .* (L.data[:,1:pred_length] .- pred))
	return loss, S, L, pred
end

calc_gradient(p,S,L) = gradient(p->loss(p,S,L)[1], p)[1]

# For iterative fitting of times series
function weights(a, tsteps, S; b=10.0, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vv = copy(v)
	for i in 2:S.m
		vv = vcat(vv,v)
	end
	return vv
end

function fit_diffeq(S)
	data, u0, tspan, tsteps = S.f_data(S);
	dudt, ode!, predict = setup_diffeq_func(S);
	
	# If using subset of data for training then keep original and truncate tsteps
	tsteps_all = copy(tsteps)
	tspan_all = tspan
	if (S.train_frac < 1)
		tsteps = tsteps[tsteps .<= S.train_frac*tsteps[end]]
		tspan = (tsteps[begin], tsteps[end])
	end
	
	beta_a = 1:S.wt_incr:S.wt_steps
	if !S.use_node p_init = 0.1*rand(ode_num_param(S)) end;
	f = generate_tf_activation_f(S.tf_in_num)

	local result
	for i in 1:length(beta_a)
		println("Iterate ", i, " of ", length(beta_a))
		w = weights(S.wt_base^beta_a[i], tsteps, S)
		last_time = tsteps[length(w[1,:])]
		ts = tsteps[tsteps .<= last_time]
		# for ODE and opt_dummy, may redefine u0 and p, here just need right sizes for ode!
		prob = S.use_node ?
					NeuralODE(dudt, (0.0,last_time), S.solver, saveat = ts, 
						reltol = S.rtol, abstol = S.atol) :
					ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S, f), u0,
						(0.0,last_time), p_init, saveat = ts, reltol = S.rtol, abstol = S.atol)
		L = loss_args(u0,prob,predict,data,tsteps,w)
		# On first time through loop, set up params p for optimization. Following loop
		# turns use the parameters returned from sciml_train(), which are in result.u
		if (i == 1)
			p = S.use_node ? prob.p : p_init
			if S.opt_dummy_u0
				p = S.use_node ? vcat(rand(S.n-S.m),p) : vcat(rand(2S.n-S.m),p)
			end
		else
			p = result.u
		end
		print(loss(p,S,L)[1])
		# alternative: GalacticOptim.AutoForwardDiff() or GalacticOptim.AutoZygote()
# 		result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
# 						 p, ADAM(S.adm_learn), GalacticOptim.AutoForwardDiff();
# 						 cb = callback, maxiters=S.max_it)
	end
end

#############################################################

function tmp()
	
	# To prepare for final fitting and calculations, must set prob to full training
	# period with tspan and tsteps and then redefine loss_args values in L
	prob = S.use_node ?
			NeuralODE(dudt, tspan, S.solver, saveat = tsteps, 
					reltol = S.rtolR, abstol = S.atolR) :
			ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0,
					tspan, p_init, saveat = tsteps, reltol = S.rtolR, abstol = S.atolR)
	if (S.train_frac == 1.0)
		prob_all = prob
	else
		prob_all = S.use_node ?
			NeuralODE(dudt, tspan_all, S.solver, saveat = tsteps_all, 
					reltol = S.rtolR, abstol = S.atolR) :
			ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S.n, S.nsqr), u0, tspan_all, 
					p_init, saveat = tsteps_all, reltol = S.rtolR, abstol = S.atolR)
		
	end
	w = ones(2,length(tsteps))
	L = loss_args(u0,prob,predict,ode_data,ode_data_orig,tsteps,w)
	A = all_time(prob_all, tsteps_all)
	p_opt = refine_fit(result.u, S, L)
	return p_opt, L, A
end


end # module
