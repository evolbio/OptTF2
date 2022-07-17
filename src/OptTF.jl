module OptTF
using Symbolics, Combinatorics, Parameters, JLD2, Printf, DifferentialEquations,
	Distributed, Optimization, OptimizationOptimJL, ForwardDiff, 	
	Statistics, Plots, Distributions, Optimisers, Random, NNlib,
	OptimizationOptimisers, Lux, DiffEqSensitivity
include("OptTF_settings.jl")
include("OptTF_param.jl")
include("OptTF_plots.jl")
include("OptTF_data.jl")
using .OptTF_settings
using .OptTF_param
using .OptTF_plots
using .OptTF_data
export generate_tf_activation_f, calc_v, set_r, mma, fit_diffeq, make_loss_args_all,
			refine_fit_bfgs, refine_fit, setup_refine_fit, loss, save_data, 
			load_data, remake_days_train, loss_args, callback, mma, hill
export Settings, default_ode, reset_rseed					# from OptTF_settings
export generate_circadian, circadian_val					# from OptTF_data
export plot_callback, plot_stoch, plot_temp, plot_stoch_dev_dur, save_summary_plots,
			plot_tf, plot_percentiles, plot_w_range, plot_tf_4_onepage
export ode_parse_p

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

load_data_warning = true

@with_kw struct loss_args
	u0
	prob				# problem to send to solve
	predict				# function that calls correct solve to make prediction
	tsteps				# time steps for training data
	hill_k				# hill coeff for loss calculation
	w					# weights for sequential fitting of time series
	f					# tf_activation function
	init_on				# boolean, light signal initially on or off
	rand_offset			# boolean, offset day/night input signal
	noise_wait			# ave waiting time to random on/off switch for day/night input
	tf					# function for NODE
	re					# function to reconstruct parameters for NODE
	state				# state setup for NODE parameters
end

# When fit only to training data subset, then need full time period info
struct all_time
	prob_all			# problem for full data set	
	tsteps_all			# steps for full data set
end

make_loss_args_all(L::loss_args, A::all_time) =
					loss_args(L; prob=A.prob_all, tsteps=A.tsteps_all,
					w=ones(length(A.tsteps_all)))

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
	#   required otherwise cannot use in ode call
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
set_r(r,s) = vcat(ones(s+1),r)	# pasting r onto ones for cases of single binding
f_range(base, length, i) = base+1+(i-1)*length:base+i*length # much faster than direct indexing
calc_v(y, k, h) = (y ./ k).^h

# get full array of f values, for f=generate_tf_activation_f(S.tf_in_num) and
# p as parameters and y as full array of TF concentrations, S as settings
# see OptTF_param.ode_parse_p(p,S) for parameter extraction
# no longer using ode_parse_p because it is slow, see git version 5ca1483 for original
# of calc_f and ode
get_y(y,S,i) = (S.n == S.tf_in_num) ? y : getindex(y,S.tf_in[i])

function calc_f(f,p,y,S)
	yy = trunc_zero.(y)
	@views [f(
	 calc_v(get_y(yy,S,i),p[f_range(S.bk,S.s,i)],p[f_range(S.bh,S.s,i)]),
	 p[f_range(S.ba,S.N,i)],
	 set_r(p[f_range(S.br,S.ri,i)],S.tf_in_num)) for i in 1:S.n]
end

# assumes x on [0,1], so intensity on [10^-6,1], 10^-3 arbitrary day/night transition
intensity(x) = 10.0^(6.0*(x-1.0))

# f is TF activation, g is external input
# protein 1 is output, protein 2 influenced by input, m=2, n>=2
function ode(u, p, t, S, f, G)
	n = S.n
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	
	pp = linear_sigmoid.(p, S.d, S.k1, S.k2)	# normalizes on [0,1] w/linear_sigmoid 
	# set min on rates m_a, m_d, p_a, p_d, causes top to be p_max + p_min
	ppp = (pp .* S.p_mult) .+ S.p_min

	m_a = @view ppp[1:n]
	m_d = @view ppp[n+1:2n]
	p_a = @view ppp[2n+1:3n]
	p_d = @view ppp[3n+1:4n]
	
	# for testing, set rate parameters to constants, all optimizing via f_val
	# m_a = 1e-2 * ones(n) * S.s_per_d
	# m_d = 1e-4 * ones(n) * S.s_per_d
	# p_a = 1e-1 * ones(n) * S.s_per_d
	# p_d = 1e-3 * ones(n) * S.s_per_d

	f_val = calc_f(f,ppp,u_p,S)

	du_m = m_a .* f_val .- m_d .* u_m		# mRNA level
	du_p = p_a .* u_m .- p_d .* u_p		# protein level
	light = G.circadian_val(G,t)				# noisy circadian input
	# fast extra production rate by post-translation modification or allostery
	du_p_2 = du_p[2] + S.light_prod_rate * intensity(light)
	return [du_m; du_p[1]; du_p_2; du_p[3:end]]
end

function node(u, p, t, S, tf, re, state, G)
	n = S.n
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	p_rates = @view p[1:4n]
	p_nn = @view p[4n+1:end]
	
	# first 4n parameters are rates
	pp = linear_sigmoid.(p_rates, S.d, S.k1, S.k2)	# normalizes on [0,1] w/linear_sigmoid 
	# set min on rates m_a, m_d, p_a, p_d, causes top to be p_max + p_min
	ppp = (pp .* S.p_mult) .+ S.p_min

	m_a = @view ppp[1:n]
	m_d = @view ppp[n+1:2n]
	p_a = @view ppp[2n+1:3n]
	p_d = @view ppp[3n+1:4n]
	
	# remainder of p is for NN
	f_val = tf(u_p,re(p_nn),state)[1]
	du_m = m_a .* f_val .- m_d .* u_m		# mRNA level
	du_p = p_a .* u_m .- p_d .* u_p		  	# protein level
	light = G.circadian_val(G,t)			# noisy circadian input
	# fast extra production rate by post-translation modification or allostery
	du_p_2 = du_p[2] + S.light_prod_rate * intensity(light)
	return [du_m; du_p[1]; du_p_2; du_p[3:end]]
end

########################
# noise-related functions
# if using noise, must call solve only via loss()

# for SDE diffusion
# When x < 16, set noise so that at 4sd below normal, noise equals size of x
# should help to avoid very small or negative values resulting from noise
sqrt_abs(x) = (x > 16.) ? sqrt(x) : abs(x) / 4.
# sqrt_abs(x,k) = (x < k) ? sqrt_abs(x) : sqrt(k)
# sqrt_abs(x,k) = (x > k) ? sqrt(x) : abs(x) / sqrt(k)
# sqrt_abs(x) = (x > 1.0) ? sqrt(x) : 0.0
# sqrt_abs(x) = sqrt(abs(x))
ode_noise(u, p, t) = sqrt_abs.(u)

# callback setup to handle with negative values
affect!(integrator) = integrator.u .= abs.(integrator.u) 
cb_cond(u, t, integrator) = true
cb_zero = DiscreteCallback(cb_cond, affect!,save_positions=(false,false))

# for stochastic jumps
jump_rate(u,p,t,rate) = rate
function jump_affect!(integrator)
	i = rand(1:length(integrator.u))	# randomly select variable
	val = integrator.u[i]				# if < 1, add 1.0, else +/- sqrt
	val += (val < 1.0) ? 1.0 : rand(-1:2:1) * sqrt(val)
	integrator.u[i] = val
end
function jump_prob(prob,S)
	jump = ConstantRateJump((u,p,t) -> jump_rate(u,p,t,S.jump_rate),jump_affect!)
	# do not save soln data at jumps
	JumpProblem(prob,Direct(),jump,save_positions=(false,false))
end

# end noise-related items
#########################

# modified from FitODE.jl
# m is number of protein dimensions for target pattern
# n is number of protein dimensions
# u_init for proteins taken from target data
# for ode w/mRNA, u_init for all mRNA and for dummy proteins random or optimized
# TF ODE has n mRNA plus n proteins
function setup_diffeq_func(S)
	# If optimizing initial conditions for dummy dimensions, then for initial condition u0,
	# dummy dimensions are first entries of p
	d = 1e-2	# cutoff from boundaries at which change from linear to sigmoid
	k1 = d*(1.0+exp(-10.0*d))
	k2 = 10.0*(1.0-d) + log(d/(1.0-d))
	# 
	function predict_ode_dummy(p, prob)
		u_init = vcat(S.max_m.*linear_sigmoid.(p[1:S.n],d,k1,k2),
					S.max_p.*linear_sigmoid.(p[S.n+1:2S.n],d,k1,k2))
		if S.jump
			prob = remake(prob, u0=u_init, p=p[S.ddim+1:end])
			prob = jump_prob(prob,S)
			solve(prob, S.solver)
		else
			solve(prob, S.solver, u0=u_init, p=p[S.ddim+1:end])
		end
	end
	# predict_ode_nodummy(p, prob, u_init) = solve(prob, S.solver, p=p)
	function predict_ode_nodummy(p, prob)
		if S.jump
			prob = remake(prob, p=p)
			prob = jump_prob(prob,S)
			solve(prob, S.solver)
		else
			solve(prob, S.solver, p=p)
		end
	end

	predict = S.opt_dummy_u0 ?
		predict_ode_dummy :
		predict_ode_nodummy
	return predict
end

hill(m,k,x) = x^k/(m^k+x^k)

function callback(p, loss_val, S, L, G, pred_all; doplot = true, show_all = true)
	# printing gradient takes calculation time, turn off may yield speedup
	if (S.print_grad)
		grad = calc_gradient(p,S,L)
		gnorm = sqrt(sum(abs2, grad))
		println(@sprintf("%5.3e; %5.3e", loss_val, gnorm))
	else
		println(@sprintf("%5.3e", loss_val))
	end
	if doplot && myid() == 1 && haskey(ENV,"DISPLAY")
		plot_callback(loss_val, S, L, G, pred_all, show_all)
  	end
  	return false
end

# loss_pow is exponent on loss values, value of 2 to square values
# and thus emphasize the largest deviant. Goal is to reduce variation
# by penalizing larger deviations more heavily. Not clear if that works.
function loss_batch(p, S, L; loss_pow=1)
	lossv = 0.0
	pred_all = Vector{Vector{Float64}}(undef,0)
	local G_ret
	lk = ReentrantLock()
	Threads.@threads for i in 1:S.batch
		if i < S.batch
			lval = loss(p,S,L)[1]
			lock(lk) do
				lossv += lval^loss_pow
			end
		else	# last iterate, pick up data to plot for this call to loss
			lval, _, _, G, pred = loss(p,S,L)
			lock(lk) do
				lossv += lval^loss_pow
				pred_all = pred
				G_ret = G
			end
		end
	end
	return lossv^(1/loss_pow), S, L, G_ret, pred_all
end

function loss(p, S, L)
	G = S.f_data(S; init_on=L.init_on, rand_offset=L.rand_offset, noise_wait=L.noise_wait)
	diffeq = S.use_node ?
				(u, p, t) -> node(u, p, t, S, L.tf, L.re, L.state, G) :
				(u, p, t) -> ode(u, p, t, S, L.f, G)
	# for SDE, cannot remake problem, so restate it
	# if using diffusion or noise, must call solve only via loss()
	# consider callback=cb_zero to handle issues with negative values
	if S.diffusion
		prob = SDEProblem(diffeq, ode_noise, L.u0, L.prob.tspan, p,
						reltol = S.rtol, abstol = S.atol,
						callback=nothing, saveat = values(L.prob.kwargs).saveat,
						maxiters=1e6)
	else
		prob = remake(L.prob, f=diffeq)
	end
	pred_all = L.predict(p, prob)

	# pick out tracked protein, first protein in system output
	pred = @view pred_all[S.n+1,:]
	pred_length = length(pred)
	input = @view G.input_true[1:pred_length]
	hill_input = hill.(0.5,L.hill_k,input)
	hill_pred  = hill.(S.switch_level,L.hill_k,pred)
	@assert pred_length == length(L.w)
	loss = sum(abs2, L.w .* (hill_input .- hill_pred))
	return loss, S, L, G, pred_all
end

calc_gradient(p,S,L) = S.use_node ?
							Zygote.gradient(p->loss(p,S,L)[1], p)[1] :
							ForwardDiff.gradient(p->loss(p,S,L)[1], p)

# For iterative fitting of times series
function weights(a, tsteps, S; b=10.0, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	w[w .> trunc]
end

# For constraints on variables, must use AutoForwardDiff() and add
# lb=zeros(2S.n), ub=1e3 .* ones(2S.n),
# However, using constraints on parameters instead, which allows AutoZygote()

# Zygote may fail depending on variety of issues that might be fixed
# ForwardDiff more reliable but may be slower
# Note: my "p" is documentation's "u" and vice versa
opt_func(S,L) = OptimizationFunction(
			(p,u) -> (S.batch == 1) ? loss(p,S,L) : loss_batch(p,S,L),
			S.use_node ? Optimization.AutoZygote() : Optimization.AutoForwardDiff())
opt_prob(p,S,L) = OptimizationProblem(opt_func(S,L), p, L.u0)

# new_rseed true uses new seed, false reuses preset_seed
# noise for stochasticity in initial parameters
# noise_wait for average time in days for loss or gain of external light signal
function fit_diffeq(S; noise = 0.1, new_rseed = S.generate_rand_seed,
						init_on = false, offset = false, noise_wait = 0.0,
						hill_k_init=2.0)
	S.set_rseed(new_rseed, S.preset_seed)
	
	# Need to extract data, tspan, tsteps
	
	# random init on [1e2,1e4] for protein, [1e0,1e2] for mRNA if !S.opt_dummy_u0
	u0 = (1e4-1e2) * rand(2S.n) .+ (1e2 * ones(2S.n))
	u0[1:S.n] .= 1e-2 * u0[S.n+1:2S.n]	# set mRNAs to 1e-2 of protein levels
	G = S.f_data(S; init_on=init_on, rand_offset=offset, noise_wait=noise_wait);
	predict = setup_diffeq_func(S);
	
	# for NODE
	if S.use_node
		ns = 7		# inner nodes per layer * num inputs
		layers = 2	# number inner layers
		in_layer = Dense(S.n => ns*S.n, identity)
		out_layer = Dense(ns*S.n => S.n, x -> hill(1,2,abs(x)))
		mid_layer = Dense(ns*S.n => ns*S.n, x -> hill(1,2,abs(x)))

		tf = Chain([x -> (log ∘ abs).(x), in_layer,
					[mid_layer for l in 1:layers]..., out_layer])

		ps, state = Lux.setup(Random.default_rng(), tf)
		p_node, re = destructure(ps)
		p_node = Lux.glorot_normal(Random.default_rng(), length(p_node); gain = 3)
	else
		tf = re = state = nothing
	end

	# If using subset of data for training then keep original and truncate tsteps
	tsteps = tsteps_all = G.tsteps
	tspan = tspan_all = G.tspan
	if (S.train_frac < 1)
		tsteps = tsteps[tsteps .<= S.train_frac*tsteps[end]]
		tspan = (tsteps[begin], tsteps[end])
	end
	
	beta_a = 1:S.wt_incr:S.wt_steps
	p = init_ode_param(u0,S; noise=noise)
	if S.use_node p = vcat(p, p_node) end
	f = generate_tf_activation_f(S.tf_in_num)

	local result
	for i in 1:length(beta_a)
		println("Iterate ", i, " of ", length(beta_a))
		w = weights(S.wt_base^beta_a[i], tsteps, S)
		# consider alternative way of increasing Hill coeff with iterates
		hill_k = hill_k_init# + i/5
		last_time = tsteps[length(w)]
		ts = tsteps[tsteps .<= last_time]
		# for ODE and opt_dummy, may redefine u0 and p, here just need right sizes for ode
		diffeq = S.use_node ?
					(u, p, t) -> node(u, p, t, S, tf, re, state, G) :
					(u, p, t) -> ode(u, p, t, S, f, G)
		prob = ODEProblem(diffeq, u0, (0.0,last_time), p, saveat = ts,
						reltol = S.rtol, abstol = S.atol)
		# if S.jump prob = jump_prob(prob,S) end
		L = loss_args(u0,prob,predict,tsteps,hill_k,w,f,init_on,offset,
						noise_wait,tf,re,state)
		# On first time through loop, set up params p for optimization. Following loop
		# turns use the parameters returned from sciml_train(), which are in result.u
		if (i > 1)
			p = result.u
			test_param(p[S.ddim+1:end],S)		# check that params are within bounds
		end
		
		# use to look at plot of initial conditions, set to false for normal use
		if false
			loss_v, _, _, G, pred_all = loss(p,S,L)
			plot_callback(loss_v, S, L, G, pred_all, true)
			#println("gradient = ", gradient(p->loss(p,S,L)[1], p)[1])
			@assert false
		end
				
		result = solve(opt_prob(p,S,L), ADAM(S.adm_learn), callback = callback,
						maxiters=S.max_it)
		
		iter = @sprintf "_%02d" i
		tmp_file = S.proj_dir * "/tmp/" * S.start_time * iter * ".jld2"
		p = result.u
		jldsave(tmp_file; p, S, L)
		if !haskey(ENV, "DISPLAY")		# if running on remote w/no display
			plot_temp(p, S, L; all_time=false)
		end
		if i > 1
			iter = @sprintf "_%02d" i-1
			tmp_file = S.proj_dir * "/tmp/" * S.start_time * iter * ".jld2"
			rm(tmp_file; force=true)
		end
	end
	# To prepare for final fitting and calculations, must set prob to full training
	# period with tspan and tsteps and then redefine loss_args values in L
	w, L, A = setup_refine_fit(result.u, S, L)
	p_opt = refine_fit(result.u, S, L)
	return p_opt, L, A
end

function setup_refine_fit(p, S, L)
	f = generate_tf_activation_f(S.tf_in_num)
	predict = setup_diffeq_func(S);
	G = S.f_data(S);
	tspan = (L.tsteps[begin], L.tsteps[end])
	diffeq = S.use_node ?
				(u, p, t) -> node(u, p, t, S, L.tf, L.re, L.state, G) :
				(u, p, t) -> ode(u, p, t, S, f, G)
	prob = ODEProblem(diffeq, L.u0, tspan, p, saveat = L.tsteps,
					reltol = S.rtolR, abstol = S.atolR)
	if (S.train_frac == 1.0)
		prob_all = prob
	else
		prob_all = ODEProblem(diffeq, L.u0, G.tspan, p, saveat = G.tsteps,
					reltol = S.rtolR, abstol = S.atolR)
		if S.jump prob_all = jump_prob(prob_all,S) end		
	end
	w = ones(length(L.tsteps))
	L = loss_args(L.u0,prob,predict,L.tsteps,L.hill_k,w,L.f,L.init_on,L.rand_offset,
			L.noise_wait,L.tf,L.re,L.state)
	A = all_time(prob_all, G.tsteps)
	return w, L, A
end

#############################################################

function refine_fit(p, S, L; rate_div=5.0, iter_mult=2.0)
	println("\nFinal round of fitting, using full time series in given data")
	println("Last step of previous fit did not fully weight final pts in series")
	println("Reducing ADAM learning rate by ", rate_div,
				" and increasing iterates by ", iter_mult, "\n")
	rate = S.adm_learn / rate_div
	iter = S.max_it * iter_mult
	result = solve(opt_prob(p,S,L), ADAM(rate), callback = callback, maxiters=iter)
	return result.u
end

function refine_fit_bfgs(p, S, L) 
	println("\nBFGS sometimes suffers instability or gives other warnings")
	println("If so, then abort and do not use result\n")
	println("In this case, BFGS may fail as do most optimizers in Optim.jl,\n\
					\tif so try using NelderMead\n")
	result = solve(opt_prob(p,S,L), BFGS(), callback = callback, maxiters=S.max_it)
	return result.u
end

# particular data saved changes with versions of program
# could send tuple rather than naming variables but helps to have
# named variables here to check that currently required variables are
# saved
save_data(p, S, L, G, L_all, loss_v, pred; file=S.out_file) =
			jldsave(file; p, S, L, G, L_all, loss_v, pred)

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

#############################################################
# utilities for manipulating output and analyzing runs

# Reset total days and training days
function remake_days_train(p, S, L; days=12, train_frac=0.5)
	S = Settings(S; days=(days|>Float64), train_frac=(train_frac|>Float64))
	ts_all = 0.0:S.save_incr:S.days
	ts = (S.train_frac < 1) ? ts_all[ts_all .<= S.train_frac*ts_all[end]] : ts_all
	tmp_L = loss_args(L; tsteps=ts)
	_, L, A = setup_refine_fit(p, S, tmp_L)
	L_all = (S.train_frac < 1) ? make_loss_args_all(L, A) : L;
	G = S.f_data(S)
	return S, L, L_all, G
end



end # module
