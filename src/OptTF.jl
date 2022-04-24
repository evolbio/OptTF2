module OptTF
#using OptTF_settings
using Symbolics, Combinatorics, Catalyst, Parameters, JLD2, Plots, Printf,
	DifferentialEquations
export generate_tf_activation_f, calc_v, set_r, mma

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
	f_expr = build_function(to_compute, v, a, r)
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

# from https://catalyst.sciml.ai/dev/tutorials/using_catalyst/#Mass-Action-ODE-Models
function generate_repressilator()
	repressilator = @reaction_network Repressilator begin
		hillr(P₃,α,K,n), ∅ --> m₁
		hillr(P₁,α,K,n), ∅ --> m₂
		hillr(P₂,α,K,n), ∅ --> m₃
		(δ,γ), m₁ <--> ∅
		(δ,γ), m₂ <--> ∅
		(δ,γ), m₃ <--> ∅
		β, m₁ --> m₁ + P₁
		β, m₂ --> m₂ + P₂
		β, m₃ --> m₃ + P₃
		μ, P₁ --> ∅
		μ, P₂ --> ∅
		μ, P₃ --> ∅
	end α K n δ γ β μ
	@parameters  α K n δ γ β μ
	@variables t m₁(t) m₂(t) m₃(t) P₁(t) P₂(t) P₃(t)
	pmap  = (α => .5, K => 40, n => 2, δ => log(2)/120, 
		      γ => 5e-3, β => 20*log(2)/120, μ => log(2)/60)
	u₀map = [m₁ => 0., m₂ => 0., m₃ => 0., P₁ => 20., P₂ => 0., P₃ => 0.]
	
	odesys = convert(ODESystem, repressilator)
	save_incr = 10.
	total_steps = 2000
	tspan_total = (0., total_steps*save_incr)
	save_steps = 1000
	tspan_save = (0., save_steps*save_incr)
	tsteps_save = 0.:save_incr:save_steps*save_incr
	oprob = ODEProblem(repressilator, u₀map, tspan_total, pmap)
	# rows 4:6 are proteins
	# first 10000 time intervals as warmup, return second period
	# size is (3,1001) for three proteins at 1001 timepoints
	# and 1000 save_incr steps
	# for fitting, probably sufficient to use subset of about 350 pts
	data = solve(oprob, Tsit5(), saveat=10.)[4:6,total_steps-save_steps:total_steps]
	u0 = data[:,1]
	return data, u0, tspan_save, tsteps_save
end

function plot_repressilator_time(sol; show_all=false)
	stop = (show_all) ? length(sol[1,:]) : 300
	display(plot([sol[i,1:stop] for i in 1:3],yscale=:log10))
end

function plot_repressilator_total(sol)
	display(plot([sum(sol[:,j]) for j in 1:length(sol[1,:])],yscale=:identity))
end

plot_repressilator_phase(sol) =
		display(plot([log2.(Tuple([sol[i,j] for i in 1:3])) for j in 1:length(sol[1,:])],
			camera=(50,60), linewidth=2, color=mma[1], limits=(5.2,8.8),
			ticks=(5.644:8.644,string.([50,100,200,400]))))

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
			solve(prob, S.solver, u0=vcat(u_init,p[1:ddim_all]), p=p[ddim_all+1:end])
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
			P = ode_parse_p(p,S)
			f_val = calc_f(f,P,u,S)
			du[1:n] .= P.m_a .* f_val .- P.m_d .* u[1:n]			# mRNA level
			du[n+1:2n] .= P.p_a .* u[1:n] .- P.p_d .* u[n+1:2n]		# protein level
		end
		predict = S.opt_dummy_u0 ?
			predict_ode_dummy :
			predict_ode_nodummy
	end
	return dudt, ode!, predict
end

#############################################################

function callback(p, loss_val, S, L, pred; doplot = true, show_lines = false)
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
		ts = L.tsteps[1:len]
		plt = plot(size=(600,800), layout=(2,1))
		plot_type! = if show_lines plot! else scatter! end
		plot_type!(ts, L.ode_data[1,1:len], label = "hare", subplot=1)
		plot_type!(plt, ts, pred[1,:], label = "pred", subplot=1)
		plot_type!(ts, L.ode_data[2,1:len], label = "lynx", subplot=2)
		plot_type!(plt, ts, pred[2,:], label = "pred", subplot=2)
		display(plot(plt))
  	end
  	return false
end

function loss(p, S, L)
	pred_all = L.predict(p, L.prob, L.u0)
	pred = pred_all[1:2,:]	# First rows are hare & lynx, others dummies
	pred_length = length(pred[1,:])
	if pred_length != length(L.w[1,:]) println("Mismatch") end
	loss = sum(abs2, L.w[:,1:pred_length] .* (L.ode_data[:,1:pred_length] .- pred))
	return loss, S, L, pred_all
end

calc_gradient(p,S,L) = gradient(p->loss(p,S,L)[1], p)[1]

# For iterative fitting of times series
function weights(a, tsteps; b=10.0, trunc=S.wt_trunc) 
	w = [1 - cdf(Beta(a,b),x/tsteps[end]) for x = tsteps]
	v = w[w .> trunc]'
	vcat(v,v)
end



end # module
