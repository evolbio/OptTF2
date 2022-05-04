module OptTF
#using OptTF_settings
using Symbolics, Combinatorics, Parameters, JLD2, Plots, Printf, DifferentialEquations,
	Distributions, DiffEqFlux, GalacticOptim, StatsPlots.PlotMeasures
export generate_tf_activation_f, calc_v, set_r, mma, fit_diffeq

# CODE FOR NODE NOT COMPLETE, USE ODE ONLY UNTIL NODE COMPLETED

# Variables may go negative, which throws error. Could add bounds
# to constrain optimization. But for now sufficient just to rerun
# with lower noise for parameters, or lower tolerances for solver
# in settings (atol and rtol) or perhaps a different solver. Might
# be that negative values only arise through numerical error of 
# diffeq solver, so more precision can possibly prevent this issue

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
	data_diff			# first time differences for slopes at each point
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

# calc_vv(y, k, h) = begin println(length(y), " ", length(k), " ", length(h)); [(y[j]/k[j])^h[j] for j in 1:length(h)] end

trunc_zero(x) = x < 0 ? 0. : x
set_r(r,s) = vcat(ones(s+1),r)
function calc_v(y, k, h)
	# println("y, k, h sizes = ", length(y), " ", length(k), " ", length(h))
	y .= trunc_zero.(y)
	[(y[j]/k[j])^h[j] for j in 1:length(h)]
end

# get full array of f values, for f=generate_tf_activation_f(S.tf_in_num) and
# P = ode_parse_p(p,S) and y as full array of TF concentrations, S as settings
calc_f(f,P,y,S) = 
	[f(calc_v(getindex(y,S.tf_in[i]),P.k[i],P.h[i]),P.a[i],set_r(P.r[i],S.tf_in_num))
			for i in 1:S.n]

# m message, p protein, _a growth, _d decay, k dissociation, h hill coeff, r cooperativity
function ode_parse_p(p,S)
	n = S.n
	s = S.tf_in_num
	N = 2^s
	# transform parameter values for gradients into values used for ode
	# min val for rates is 1e-2, 0 for all others, max values vary, see p_max
	# see linear_sigmoid() for transform of gradient params to ode params
	d = 1e-2	# cutoff from boundaries at which change from linear to sigmoid
	k1 = d*(1.0+exp(-10.0*d))
	k2 = 10.0*(1.0-d) + log(d/(1.0-d))
	pp = linear_sigmoid.(p, d, k1, k2)	# this normalizes on [0,1] with linear_sigmoid pattern
	# length and max vals for rates, k, h, a, r
	p_dim = [4n,n*s,n*s,n*N,n*(N-(s+1))]
	p_max = [1e2,1e4,5e0,1e0,1e1]
	p_min = zeros(length(p))
	p_min[1:4n] = 1e-2 .* ones(4n)
	p_mult = []
	for i in 1:length(p_dim)
		append!(p_mult, p_max[i] .* ones(p_dim[i]))
	end
	# set min on rates m_a, m_d, p_a, p_d, causes top to be 1e2 + 1e-2
	ppp = (pp .* p_mult) .+ p_min
	m_a = @view ppp[1:n]			# n
	m_d = @view ppp[n+1:2n]			# n
	p_a = @view ppp[2n+1:3n]		# n
	p_d = @view ppp[3n+1:4n]		# n
	
	b = 4n
	k = [@view ppp[b+1+(i-1)*s:b+i*s] for i in 1:n]		# ns
	b = 4n+n*s
	h = [@view ppp[b+1+(i-1)*s:b+i*s] for i in 1:n]		# ns
	b = 4n+2*n*s
	a = [@view ppp[b+1+(i-1)*N:b+i*N] for i in 1:n]		# nN, [0,1] for TF activation
	b = 4n+2*n*s+n*N
	ri = N - (s+1)
	r = [@view ppp[b+1+(i-1)*ri:b+i*ri] for i in 1:n]	# n(N-(s+1))
	@assert length(ppp) == b + n*ri
	# return a named tuple
	(;m_a, m_d, p_a, p_d, k, h, a, r)			# return named tuple
end

test_range(x, top) = @assert minimum(x) >= 0 && maximum(x) <= top
test_range(x, top, offset) = @assert minimum(x) >= offset && maximum(x) <= top+offset

function test_param(p,S)
	P = ode_parse_p(p,S)
	test_range(P.m_a,1e2,1e-2)
	test_range(P.m_d,1e2,1e-2)
	test_range(P.p_a,1e2,1e-2)
	test_range(P.p_d,1e2,1e-2)
	test_range(minimum(P.k),1e4)	# need min of min for array of arrays
	test_range(minimum(P.h),5e0)
	test_range(minimum(P.a),1e0)
	if S.tf_in_num > 1 test_range(minimum(P.r),1e1) end
end

# precalculate and pass k1 = d(1+exp(-10d)) and k2 = 10(max-d) + log(d/(max-d))
# goes from encoded value for gradients -> param value for ode
# see mma file linear_sigmoid.nb
function linear_sigmoid(p, d, k1, k2)
	if p < d
		k1 / (1+exp(-10.0*p))
	elseif p > 1.0 - d
		1.0 / (1 + exp(-10.0*p + k2))
	else
		p
	end
end

# inverts from param value for ode -> encoded value for gradients 
function inverse_lin_sigmoid(p, d, k1, k2)
	if p < d
		0.1*log(p/(k1-p))
	elseif p > 1.0 - d
		0.1*(k2 + log(p/(1-p)))
	else
		p
	end
end

# Goal is to setup near equilibrium matching u0 for tracked proteins, dummy proteins set at
# equilibrium value of first tracked protein, u0[1], and all mRNAs at 0.1 times protein level
# use a = 1 for all a values, so that activation f is 0.5
# Yields p_a=10, p_d=m_d=1, m_a=0.2 u, for which u is target initial value of protein
# Alternatively, start with one protein type present and all other protein and mRNA conc at 0

# All params have min value at 0 except rates which are min at 1e-2, all params have max vals
# see ode_parse_p()

function init_ode_param(u0,S; noise=2e-3, start_equil=false)
	@assert length(u0) == (S.opt_dummy_u0 ? S.m : 2S.n)
	num_p = ode_num_param(S)
	p = zeros(num_p)
	n = S.n
	m = S.m
	s = S.tf_in_num
	N = 2^s
	ddim = S.opt_dummy_u0 ? 2*n - m : 0
	
	d = 1e-2	# cutoff from boundaries at which change from linear to sigmoid
	k1 = d*(1.0+exp(-10.0*d))
	k2 = 10.0*(1.0-d) + log(d/(1.0-d))

	if start_equil == true || S.opt_dummy_u0
		# dummies packed as n mRNA and n-m proteins
		# not transformed between linear_sigmoid and inverse_lin_sigmoid, use raw values
		if S.opt_dummy_u0
			# m mRNA for tracked proteins, 0.1*u0
			p[1:m] .= [inverse_lin_sigmoid(0.1*u0[i]/1e4,d,k1,k2) for i in 1:m]					
			if n > m
				# n-m dummy mRNA, 0.1*u0[1]
				p[m+1:n] .= [inverse_lin_sigmoid(0.1*u0[1]/1e4,d,k1,k2) for i in 1:m+1:n]
				# n-m dummy proteins set to u0[1]
				p[n+1:2n-m] .= [inverse_lin_sigmoid(u0[1]/1e4,d,k1,k2) for i in 1:n+1:2n-m]
			end
		end
		base = S.opt_dummy_u0 ? 0 : n				# if false, u0 is 2S.n, if true, u0 is S.m
# 		p[ddim+1:ddim+m] .= 0.2 .* u0[base+1:base+m] # m_a
# 		if (n>m) p[ddim+m+1:ddim+n] .= (0.2 * u0[base+1]) .* ones(n-m) end
		# start with parameters the same for all loci
		p[ddim+1:ddim+n] .= 0.2 .* u0[base+1] .* ones(n) # m_a
		p[ddim+n+1:ddim+2n] .= ones(n)				# m_d
		p[ddim+2n+1:ddim+3n] .= 10.0 .* ones(n)		# p_a
		p[ddim+3n+1:ddim+4n] .= ones(n)				# p_d
	else
		u0 .= vcat(zeros(n),[20.],zeros(n-1))
		p[ddim+1:ddim+n] .= 10.0 .* ones(n)			# m_a
		p[ddim+n+1:ddim+2n] .= ones(n)				# m_d
		p[ddim+2n+1:ddim+3n] .= 10.0 .* ones(n)		# p_a
		p[ddim+3n+1:ddim+4n] .= ones(n)				# p_d		
	end
	
	# ode_parse adds 1e-2 to rate parameters, so subtract here
	# multiply by 0.1 to slow down rate processes, otherwise so fast
	# that equil achieved and maintained too strongly, so cannot fit fluctuations
	p[ddim+1:ddim+4n] .= 0.1 .* p[ddim+1:ddim+4n] .- (1e-2 .* ones(4n))
	
	p[ddim+1:ddim+4n] .= [inverse_lin_sigmoid(p[i]/1e2,d,k1,k2) for i in ddim+1:ddim+4n]
	
	b = ddim+4n
	p[b+1:b+n*s] .= 5e2 .* ones(n*s)			# k
	p[b+1:b+n*s] .= [inverse_lin_sigmoid(p[i]/1e4,d,k1,k2) for i in b+1:b+n*s]
	
	b = ddim+4n+n*s
	p[b+1:b+n*s] .= 2.0 .* ones(n*s)			# h
	p[b+1:b+n*s] .= [inverse_lin_sigmoid(p[i]/5e0,d,k1,k2) for i in b+1:b+n*s]
	
	b = ddim+4n+2n*s
	# p[b+1:b+n*N] .= 0.5 .* ones(n*N)			# a
	# use rand to provide more initial variation
	p[b+1:b+n*N] .= rand(n*N)					# a
	p[b+1:b+n*N] .= [inverse_lin_sigmoid(p[i],d,k1,k2) for i in b+1:b+n*N]
	
	b = ddim+4n+2*n*s+n*N
	n_r = n*(N-(s+1))
	p[b+1:b+n_r] .= ones(n_r)					# r
	p[b+1:b+n_r] .= [inverse_lin_sigmoid(p[i]/1e1,d,k1,k2) for i in b+1:b+n_r]
	
	@assert (b+n_r) == num_p
	# Add small amount of noise, note that will be transformed by sigmoid, so nonlinear
	p .= p .* (1.0 .+ noise.*randn(num_p))
	return p
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
	d = 1e-2	# cutoff from boundaries at which change from linear to sigmoid
	k1 = d*(1.0+exp(-10.0*d))
	k2 = 10.0*(1.0-d) + log(d/(1.0-d))
	predict_node_dummy(p, prob, u_init) =
	  		Array(prob(vcat(u_init,(1e4.*linear_sigmoid.(p[1:ddim],d,k1,d2))), p[ddim+1:end]))
	predict_node_nodummy(p, prob, u_init) = Array(prob(u_init, p))
	predict_ode_dummy(p, prob, u_init) =
			solve(prob, S.solver, u0=vcat((1e4.*linear_sigmoid.(p[1:S.n],d,k1,k2)),
					u_init,(1e4.*linear_sigmoid.(p[S.n+1:ddim_all],d,k1,k2))),
					p=p[ddim_all+1:end])
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
			u_m = @view u[1:n]			# mRNA
			u_p = @view u[n+1:2n]		# protein
			P = ode_parse_p(p,S)
			f_val = calc_f(f,P,u_p,S)
			du[1:n] .= P.m_a .* f_val .- P.m_d .* u_m			# mRNA level
			du[n+1:2n] .= P.p_a .* u_m .- P.p_d .* u_p		# protein level
		end
		predict = S.opt_dummy_u0 ?
			predict_ode_dummy :
			predict_ode_nodummy
	end
	return dudt, ode!, predict
end

function callback(p, loss_val, S, L, pred_all;
						doplot = true, show_all = true, show_lines = true)
	# printing gradient takes calculation time, turn off may yield speedup
	if (S.print_grad)
		grad = gradient(p->loss(p,S,L)[1], p)[1]
		gnorm = sqrt(sum(abs2, grad))
		println(@sprintf("%5.3e; %5.3e", loss_val, gnorm))
	else
		println(@sprintf("%5.3e", loss_val))
		b = S.opt_dummy_u0 ? 2S.n-S.m : 0
		#P = ode_parse_p(p[b+1:end],S)
		#display(P.a)
	end
	if doplot
		len = length(pred_all[1,:])
		ts = L.tsteps[1:len]
		plot_type! = if show_lines plot! else scatter! end
		if !show_all
			# plot current prediction against data, show only target proteins
			pred = if S.use_node @view pred_all[1:S.m,:] else @view pred_all[S.n+1:S.n+S.m,:] end
			dim = length(pred[:,1])
			plt = plot(size=(600,400*dim), layout=(dim,1),left_margin=12px)
			for i in 1:dim
				plot_type!(ts, L.data[i,1:len], label = "", color=mma[1], subplot=i)
				plot_type!(plt, ts, pred[i,:], label = "", color=mma[2], subplot=i)
			end
		elseif S.use_node	# NODE and show_all, NOT TESTED YET
			dim = length(pred_all[:,1])
			plt = plot(size=(600,250*dim), layout=(dim,1),left_margin=12px)
			for i in 1:dim
				# proteins only, no mRNA in NODE
				if (i <= S.m) # target data
					plot_type!(ts, L.data[i:len], label = "", color=mma[1], subplot=i)
				end
				plot_type!(plt, ts, pred_all[i,:], label = "", color=mma[2], subplot=i)
			end			
		else				# ODE and show_all
			dim = length(pred_all[:,1])
			plt = plot(size=(1200,250*(dim ÷ 2)), layout=((dim ÷ 2),2),left_margin=12px)
			for i in 1:dim
				# proteins [S.n+1:2S.n] on left, mRNA [1:S.n] on right
				idx = vcat(2:2:dim,1:2:dim) 	# subplot index [2,4,6,...,1,3,5,...]
				if (i > S.n && i <= (S.n + S.m)) # target data
					plot_type!(ts, L.data[i-S.n,1:len], label = "", color=mma[1], subplot=idx[i])
				end
				plot_type!(plt, ts, pred_all[i,:], label = "", color=mma[2], subplot=idx[i])
			end			
		end
		display(plot(plt))
  	end
  	return false
end

function loss(p, S, L)
	pred_all = L.predict(p, L.prob, L.u0)
	# pick out tracked proteins
	pred = if S.use_node @view pred_all[1:S.m,:] else @view pred_all[S.n+1:S.n+S.m,:] end
	pred_length = length(pred[1,:])
	if pred_length != length(L.w[1,:]) println("Mismatch") end
	# add cost for concentrations over 1e3
	loss = sum(abs2, L.w[:,1:pred_length] .* (L.data[:,1:pred_length] .- pred))
	
	# add loss amount for differences in slopes by first diffs
	# probably not useful because if difficulty fitting with MSE loss, using
	# slopes will typically not fix the difficulty
	
	# pred_diff = pred[:,2:end] - pred[:,1:end-1]
	# loss_diff = sum(abs2, L.w[:,1:pred_length-1] .*
	#					(L.data_diff[:,1:pred_length-1] .- pred_diff[:,1:pred_length-1]))
	# loss = loss + 0e3*loss_diff

	return loss, S, L, pred_all
end

# this uses zygote, which seems to be very slow, consider ForwardDiff
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

# new_rseed true uses new seed, false reuses preset_seed
function fit_diffeq(S; noise = 0.05, new_rseed = S.generate_rand_seed)
	S.set_rseed(new_rseed, S.preset_seed)
	data, data_diff, u0, tspan, tsteps = S.f_data(S);
	dudt, ode!, predict = setup_diffeq_func(S);
	
	# If using subset of data for training then keep original and truncate tsteps
	tsteps_all = copy(tsteps)
	tspan_all = tspan
	if (S.train_frac < 1)
		tsteps = tsteps[tsteps .<= S.train_frac*tsteps[end]]
		tspan = (tsteps[begin], tsteps[end])
	end
	
	beta_a = 1:S.wt_incr:S.wt_steps
	if !S.use_node p_init = init_ode_param(u0,S; noise=noise) end;
	f = generate_tf_activation_f(S.tf_in_num)
	num_var = S.use_node ? S.n : 2S.n

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
						(-00.0,last_time), p_init, saveat = ts,
						reltol = S.rtol, abstol = S.atol)
		L = loss_args(u0,prob,predict,data,data_diff,tsteps,w)
		# On first time through loop, set up params p for optimization. Following loop
		# turns use the parameters returned from sciml_train(), which are in result.u
		if (i == 1)
			p = S.use_node ? prob.p : p_init
			if S.opt_dummy_u0 && S.use_node p = vcat(randn(S.n-S.m),p) end
		else
			p = result.u
			if !S.use_node
				ddim = S.opt_dummy_u0 ? 2S.n - S.m : 0
				test_param(p[ddim+1:end],S)		# check that params are within bounds
			end
		end
		
		# use to look at plot of initial conditions, set to false for normal use
		if false
			loss_v, _, _, pred_all = loss(p,S,L)
			callback(p, loss_v, S, L, pred_all)
			@assert false
		end
		
		# see https://galacticoptim.sciml.ai/stable/API/optimization_function for
		# alternative optimization functions, use GalacticOptim. prefix
		# common choices AutoZygote() and AutoForwardDiff()
		# Zygote may fail depending on variety of issues that might be fixed
		# ForwardDiff more reliable but may be slower
		
		# For constraints on variables, must use AutoForwardDiff() and add
		# lb=zeros(num_var), ub=1e3 .* ones(num_var),
		# However, using constraints on parameters instead, which allows Zygote
		result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
						 p, ADAM(S.adm_learn), GalacticOptim.AutoForwardDiff();
						 #lb=zeros(num_var), ub=1e3 .* ones(num_var),
						 cb = callback, maxiters=S.max_it)
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
