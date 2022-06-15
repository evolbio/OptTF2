module OptTF_param
using OptTF
export ode_parse_p, test_param, linear_sigmoid, init_ode_param
# Functions for managing parameters
# Primary goal here is to set bounds on parameters
# Also, bounds on parameters set bounds on concentration variables.
# All parameters and variables are non-negative. Each parameter has an upper bound
# set in OptTF_settings. The rate parameters also have a lower bound, S.[mp]_low_rate

# To set bounds, the method is to allow the parameter vector used for optimization
# to vary without bounds and to transform raw parameter values into parameters used
# in ode! by parsing the parameters. Typically, each raw parameter maps directly 
# to the used parameter over the range [low + d', hi - d'], where low and high are
# the parameter bounds and d' = d(hi-low) is a zone near the boundaries that is
# transformed by a sigmoid function. For example, a raw parameter p < low+d' is
# transformed by a sigmoid function to map p into the range [low, low+d']. There
# is also an inverse mapping from and ode parameter in [low,low+d'] to a raw
# parameter, where, for example, as the used parameter approaches low, the raw
# parameter becomes a large negative value. Similarly for the top of the range.

# m message, p protein, _a growth, _d decay, k dissociation, h hill coeff, r cooperativity
function ode_parse_p(p,S)
	n = S.n
	s = S.s
	N = S.N
	# transform parameter values for gradients into values used for ode
	# min val for rates is S.d=1e-2, 0 for all others, max values vary, see p_max
	# see linear_sigmoid() for transform of gradient params to ode params
	pp = linear_sigmoid.(p, S.d, S.k1, S.k2)	# normalizes on [0,1] with linear_sigmoid pattern
	# set min on rates m_a, m_d, p_a, p_d, causes top to be p_max + p_min
	ppp = (pp .* S.p_mult) .+ S.p_min
	m_a = @view ppp[1:n]			# n
	m_d = @view ppp[n+1:2n]			# n
	p_a = @view ppp[2n+1:3n]		# n
	p_d = @view ppp[3n+1:4n]		# n
	
	k = [@view ppp[S.bk+1+(i-1)*s:S.bk+i*s] for i in 1:n]		# ns
	h = [@view ppp[S.bh+1+(i-1)*s:S.bh+i*s] for i in 1:n]		# ns
	a = [@view ppp[S.ba+1+(i-1)*N:S.ba+i*N] for i in 1:n]		# nN, [0,1] for TF activation
	r = [@view ppp[S.br+1+(i-1)*S.ri:S.br+i*S.ri] for i in 1:n]	# n(N-(s+1))
	
	# preallocating is faster but fails with AutoDiff. For example
	# k = Vector{SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}}(undef,n)
	# for i in 1:n  k[i] = @view ppp[S.bk+1+(i-1)*s:S.bk+i*s]  end

	@assert length(ppp) == S.br + n*S.ri
	# return a named tuple
	(;m_a, m_d, p_a, p_d, k, h, a, r)			# return named tuple
end

test_range(x,top,offset) = @assert minimum(x) >= offset && maximum(x) <= top+offset
test_range(xmin,xmax,top,offset) =
					@assert minimum(xmin) >= offset && maximum(xmax) <= top+offset

function test_param(p,S)
	P = ode_parse_p(p,S)
	no_offset = 0.0
	test_range(P.m_a,S.m_rate,S.m_low_rate)
	test_range(P.m_d,S.m_rate,S.m_low_rate)
	test_range(P.p_a,S.p_rate,S.p_low_rate)
	test_range(P.p_d,S.p_rate,S.p_low_rate)
	test_range(minimum(P.k),maximum(P.k),S.k,S.k_min)	# need min of min for array of arrays
	test_range(minimum(P.h),maximum(P.h),S.h,no_offset)
	test_range(minimum(P.a),maximum(P.a),S.a,no_offset)
	if S.tf_in_num > 1 test_range(minimum(P.r),maximum(P.r),S.r,no_offset) end
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

# All params have min value at 0 except rates which are min at low_rate, all params have max vals
# see ode_parse_p()

function init_ode_param(u0,S; noise=1e-1)
	num_p = S.num_param
	p = zeros(num_p)
	n = S.n
	m = S.m
	s = S.tf_in_num
	N = 2^s
	ddim = S.ddim
	
	d = 1e-2	# cutoff from boundaries at which change from linear to sigmoid
	k1 = d*(1.0+exp(-10.0*d))
	k2 = 10.0*(1.0-d) + log(d/(1.0-d))

	if S.opt_dummy_u0
		p[1:n] .= [inverse_lin_sigmoid(u0[i]/S.max_m,d,k1,k2) for i in 1:n]					
		p[n+1:2n] .= [inverse_lin_sigmoid(u0[i]/S.max_p,d,k1,k2) for i in n+1:2n]					
	end
 	# mRNA equil = u0 for protein * 1e-2; protein equil = u0 for protein
 	# should make constants here in relation to settings, S, otherwise fragile
 	# and will break for changes in m_rate, p_rate
	p[ddim+1:ddim+n] .= 1.01e-6 .* u0[n+1:2n] .* ones(n) 	# m_a
	p[ddim+n+1:ddim+2n] .= 1.01e-4 * ones(n)			# m_d
	p[ddim+2n+1:ddim+3n] .= 1.01e-1 * ones(n)			# p_a
	p[ddim+3n+1:ddim+4n] .= 1.01e-3 * ones(n)			# p_d
	
	# ode_parse adds S.low_rate to rates, so subtract here, change rates to 1/d
	p[ddim+1:ddim+2n] .= S.s_per_d .* p[ddim+1:ddim+2n] .- (S.m_low_rate .* ones(2n))
	p[ddim+2n+1:ddim+4n] .= S.s_per_d .* p[ddim+2n+1:ddim+4n] .- (S.p_low_rate .* ones(2n))
	@assert minimum(p[ddim+1:ddim+4n]) > 0
	
	p[ddim+1:ddim+2n] .= [inverse_lin_sigmoid(p[i]/S.m_rate,d,k1,k2) for i in ddim+1:ddim+2n]
	p[ddim+2n+1:ddim+4n] .= [inverse_lin_sigmoid(p[i]/S.p_rate,d,k1,k2)
									for i in ddim+2n+1:ddim+4n]
	
	b = ddim+4n
	# ode_parse adds S.k_min, so subtract here
	p[b+1:b+n*s] .= 5e2 .* ones(n*s) .- (S.k_min .* ones(n*s))			# k
	p[b+1:b+n*s] .= [inverse_lin_sigmoid(p[i]/S.k,d,k1,k2) for i in b+1:b+n*s]
	
	b = ddim+4n+n*s
	p[b+1:b+n*s] .= 2.0 .* ones(n*s)			# h
	p[b+1:b+n*s] .= [inverse_lin_sigmoid(p[i]/S.h,d,k1,k2) for i in b+1:b+n*s]
	
	b = ddim+4n+2n*s
	# p[b+1:b+n*N] .= 0.5 .* ones(n*N)			# a
	p[b+1:b+n*N] .= rand(n*N)					# use rand to provide more initial variation
	p[b+1:b+n*N] .= [inverse_lin_sigmoid(p[i]/S.a,d,k1,k2) for i in b+1:b+n*N]
	
	b = ddim+4n+2*n*s+n*N
	n_r = n*(N-(s+1))
	p[b+1:b+n_r] .= ones(n_r)					# r
	p[b+1:b+n_r] .= [inverse_lin_sigmoid(p[i]/S.r,d,k1,k2) for i in b+1:b+n_r]
	
	@assert (b+n_r) == num_p
	# Add small amount of noise, note that will be transformed by sigmoid, so nonlinear
	p .= p .* (1.0 .+ noise.*randn(num_p))
	return p
end

end	#module