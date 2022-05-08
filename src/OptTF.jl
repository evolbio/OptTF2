module OptTF
#using OptTF_settings
using Symbolics, Combinatorics, Parameters, JLD2, Plots, Printf, DifferentialEquations,
	Distributions, DiffEqFlux, GalacticOptim, StatsPlots.PlotMeasures
include("OptTF_param.jl")
export generate_tf_activation_f, calc_v, set_r, mma, fit_diffeq, make_loss_args_all,
			refine_fit_bfgs, refine_fit, loss, save_data, load_data, ode_parse_p

# Variables may go negative, which throws error. Could add bounds
# to constrain optimization. But for now sufficient just to rerun
# with lower noise for parameters, or lower tolerances for solver
# in settings (atol and rtol) or perhaps a different solver. Might
# be that negative values only arise through numerical error of 
# diffeq solver, so more precision can possibly prevent this issue

####################################################################
# colors, see MMAColors.jl in my private modules

const mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

load_data_warning = true

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

function ode!(du, u, p, t, S, f)
	n = S.n
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	P = ode_parse_p(p,S)
	f_val = calc_f(f,P,u_p,S)
	du[1:n] .= P.m_a .* f_val .- P.m_d .* u_m			# mRNA level
	du[n+1:2n] .= P.p_a .* u_m .- P.p_d .* u_p			# protein level
end

# modified from FitODE.jl
# m is number of protein dimensions for target pattern
# n is number of protein dimensions
# u_init for proteins taken from target data
# for ode w/mRNA, u_init for all mRNA and for dummy proteins random or optimized
# TF ODE has n mRNA plus n proteins
function setup_diffeq_func(S)
	ddim_p = S.n - S.m		# dummy protein dimensions
	ddim_all = ddim_p + S.n	# dummy proteins plus dummy mRNA
	# If optimizing initial conditions for dummy dimensions, then for initial condition u0,
	# dummy dimensions are first entries of p
	d = 1e-2	# cutoff from boundaries at which change from linear to sigmoid
	k1 = d*(1.0+exp(-10.0*d))
	k2 = 10.0*(1.0-d) + log(d/(1.0-d))
	max_u = S.rate / S.low_rate
	predict_ode_dummy(p, prob, u_init) =
			solve(prob, S.solver, u0=vcat((max_u.*linear_sigmoid.(p[1:S.n],d,k1,k2)),
					u_init,(max_u.*linear_sigmoid.(p[S.n+1:ddim_all],d,k1,k2))),
					p=p[ddim_all+1:end])
	predict_ode_nodummy(p, prob, u_init) = solve(prob, S.solver, p=p)

	predict = S.opt_dummy_u0 ?
		predict_ode_dummy :
		predict_ode_nodummy
	return predict
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
			pred = @view pred_all[S.n+1:S.n+S.m,:]
			dim = length(pred[:,1])
			plt = plot(size=(600,400*dim), layout=(dim,1),left_margin=12px)
			for i in 1:dim
				plot_type!(ts, L.data[i,1:len], label = "", color=mma[1], subplot=i)
				plot_type!(plt, ts, pred[i,:], label = "", color=mma[2], subplot=i)
			end
		else
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
	pred = @view pred_all[S.n+1:S.n+S.m,:]
	pred_length = length(pred[1,:])
	if pred_length != length(L.w[1,:]) println("Mismatch") end
	# add cost for concentrations over 1e3
	loss = sum(abs2, L.w[:,1:pred_length] .* (L.data[:,1:pred_length] .- pred))
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
function fit_diffeq(S; noise = 0.1, new_rseed = S.generate_rand_seed)
	S.set_rseed(new_rseed, S.preset_seed)
	data, data_diff, u0, tspan, tsteps = S.f_data(S);
	predict = setup_diffeq_func(S);
	
	# If using subset of data for training then keep original and truncate tsteps
	tsteps_all = copy(tsteps)
	tspan_all = tspan
	if (S.train_frac < 1)
		tsteps = tsteps[tsteps .<= S.train_frac*tsteps[end]]
		tspan = (tsteps[begin], tsteps[end])
	end
	
	beta_a = 1:S.wt_incr:S.wt_steps
	p = init_ode_param(u0,S; noise=noise)
	f = generate_tf_activation_f(S.tf_in_num)

	local result
	for i in 1:length(beta_a)
		println("Iterate ", i, " of ", length(beta_a))
		w = weights(S.wt_base^beta_a[i], tsteps, S)
		last_time = tsteps[length(w[1,:])]
		ts = tsteps[tsteps .<= last_time]
		# for ODE and opt_dummy, may redefine u0 and p, here just need right sizes for ode!
		prob = ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S, f), u0,
						(-00.0,last_time), p, saveat = ts,
						reltol = S.rtol, abstol = S.atol)
		L = loss_args(u0,prob,predict,data,data_diff,tsteps,w)
		# On first time through loop, set up params p for optimization. Following loop
		# turns use the parameters returned from sciml_train(), which are in result.u
		if (i > 1)
			p = result.u
			ddim = S.opt_dummy_u0 ? 2S.n - S.m : 0
			test_param(p[ddim+1:end],S)		# check that params are within bounds
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
		# lb=zeros(2S.n), ub=1e3 .* ones(2S.n),
		# However, using constraints on parameters instead, which allows Zygote
		result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
						 p, ADAM(S.adm_learn), GalacticOptim.AutoZygote();
						 cb = callback, maxiters=S.max_it)
	end
	# To prepare for final fitting and calculations, must set prob to full training
	# period with tspan and tsteps and then redefine loss_args values in L
	prob = ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S, f), u0,
					tspan, p, saveat = tsteps, reltol = S.rtolR, abstol = S.atolR)
	if (S.train_frac == 1.0)
		prob_all = prob
	else
		prob_all = ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S, f), u0, tspan_all, 
					p, saveat = tsteps_all, reltol = S.rtolR, abstol = S.atolR)
		
	end
	w = ones(S.m,length(tsteps))
	L = loss_args(u0,prob,predict,data,data_diff,tsteps,w)
	A = all_time(prob_all, tsteps_all)
	p_opt = refine_fit(result.u, S, L)
	return p_opt, L, A
end

#############################################################

function refine_fit(p, S, L; rate_div=5.0, iter_mult=2.0)
	println("\nFinal round of fitting, using full time series in given data")
	println("Last step of previous fit did not fully weight final pts in series")
	println("Reducing ADAM learning rate by ", rate_div,
				" and increasing iterates by ", iter_mult, "\n")
	rate = S.adm_learn / rate_div
	iter = S.max_it * iter_mult
	result = DiffEqFlux.sciml_train(p -> loss(p,S,L), p, ADAM(rate),
						GalacticOptim.AutoZygote(); cb = callback, maxiters=iter)
	return result.u
end

function refine_fit_bfgs(p, S, L) 
	println("\nBFGS sometimes suffers instability or gives other warnings")
	println("If so, then abort and do not use result\n")
	result = DiffEqFlux.sciml_train(p -> loss(p,S,L),
						 p, BFGS(); cb = callback, maxiters=S.max_it)
	return result.u
end

# particular data saved changes with versions of program
# could send tuple rather than naming variables but helps to have
# named variables here to check that currently required variables are
# saved
save_data(p, S, L, L_all, loss_v, pred; file=S.out_file) =
			jldsave(file; p, S, L, L_all, loss_v, pred)

# jld2 stores data as Dict with keys as strings
# to return named tuple, must first transform to dict with keys as symbols
#	see https://www.reddit.com/r/Julia/comments/8aw93w
# then use ... operator to transform dict w/symbol keys to named tuple
#	see https://discourse.julialang.org/t/10899/8?page=2
# Disadvantage of loading whatever is saved is that no checking for particular
# variables is made, so caller must check that required variables present
function load_data(file)
	global load_data_warning
	dt_string_keys = load(file)
	dt_symbol_keys = Dict()
	for (k,v) in dt_string_keys
    	dt_symbol_keys[Symbol(k)] = v
	end
	if load_data_warning
		println("\nWarning may occur if loaded data struct not defined or")
		println("differs from current definition of that structure. Check keys")
		println("in returned named tuple by using keys() on returned value. Check")
		println("if a required key is missing and adjust accordingly.\n")
		load_data_warning = false
	end
	(; dt_symbol_keys...)
end

# Can check for specific keys in load by this function, if required key
# is missing, should throw an error
function load_data_old(file)
	dt = load(file)
	(p = dt["p"], S = dt["S"], L = dt["L"], L_all = dt["L_all"],
						loss_v = dt["loss_v"], pred = dt["pred"])
end

end # module
