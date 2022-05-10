# Functions for managing parameters
# Primary goal here is to set bounds on parameters
# Also, bounds on parameters set bounds on concentration variables.
# All parameters and variables are non-negative. Each parameter has an upper bound
# set in OptTF_settings. The rate parameters also have a lower bound, S.low_rate

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
	# set min on rates m_a, m_d, p_a, p_d, causes top to be 1e2 + 1e-2
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

test_range(x, top) = @assert minimum(x) >= 0 && maximum(x) <= top
test_range(x, top, offset) = @assert minimum(x) >= offset && maximum(x) <= top+offset

function test_param(p,S)
	P = ode_parse_p(p,S)
	test_range(P.m_a,S.rate,S.low_rate)
	test_range(P.m_d,S.rate,S.low_rate)
	test_range(P.p_a,S.rate,S.low_rate)
	test_range(P.p_d,S.rate,S.low_rate)
	test_range(minimum(P.k),S.k)	# need min of min for array of arrays
	test_range(minimum(P.h),S.h)
	test_range(minimum(P.a),S.a)
	if S.tf_in_num > 1 test_range(minimum(P.r),S.r) end
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

function init_ode_param(u0,S; noise=1e-1, start_equil=false)
	@assert length(u0) == (S.opt_dummy_u0 ? S.m : 2S.n)
	num_p = S.num_param
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
				p[m+1:n] .= [inverse_lin_sigmoid(0.1*u0[1]/1e4,d,k1,k2) for i in m+1:n]
				# n-m dummy proteins set to u0[1]
				p[n+1:2n-m] .= [inverse_lin_sigmoid(u0[1]/1e4,d,k1,k2) for i in n+1:2n-m]
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
	
	# ode_parse adds S.low_rate to rate parameters, so subtract here
	# multiply by 0.1 to slow down rate processes, otherwise so fast
	# that equil achieved and maintained too strongly, so cannot fit fluctuations
	p[ddim+1:ddim+4n] .= 0.1 .* p[ddim+1:ddim+4n] .- (S.low_rate .* ones(4n))
	
	p[ddim+1:ddim+4n] .= [inverse_lin_sigmoid(p[i]/S.rate,d,k1,k2) for i in ddim+1:ddim+4n]
	
	b = ddim+4n
	p[b+1:b+n*s] .= 5e2 .* ones(n*s)			# k
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
