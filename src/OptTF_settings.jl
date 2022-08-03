module OptTF_settings
include("OptTF_data.jl")
using .OptTF_data
using Parameters, DifferentialEquations, Dates, Random, StatsBase, NNlib
export Settings, default_ode, reset_rseed

# One can initialize and then modify settings as follows
# S = Settings(S; layer_size=50, activate=3, [ADD OTHER OPTIONS AS NEEDED])
# S = default_ode()		# default settings for ODE
# S = Settings(S; opt_dummy_u0 = true, [ADD OTHER OPTIONS AS NEEDED])
# However, many variables are calculated from others, so often best to start over
# See docs for Parameters.jl package

default_ode() = Settings(
	n			= 4,
	rtol		= 1e-4,
	atol		= 1e-6,
	adm_learn	= 0.002,
	days		= 6.0,
	train_frac	= 2/3,
	max_it		= 150,
	jump 		= false,
	diffusion	= true,
	batch 		= 12
)

reset_rseed(S, rseed) = Settings(S; generate_rand_seed=false, preset_seed=rseed,
							actual_seed=set_rand_seed(false,rseed))

function set_rand_seed(gen_seed, preset_seed)
	rseed = gen_seed ? rand(UInt) : preset_seed
	Random.seed!(rseed)
	return rseed
end

@with_kw struct Settings

# use NODE for TF network input-output
use_node = true

# function to generate or load data for fitting
f_data = generate_circadian

# rates are per second, transform to per day by multiplying by 86400.0 s/d
s_per_d = 86400.0

# stochastic jump
jump = false
jump_rate = 5e-4 * s_per_d
diffusion = true
batch = 6						# parallel loss calculation with batching
@assert (!jump || !diffusion)	# can't have both jump and diffusion in current code
# @assert jump || diffusion || (batch == 1)		# allow batch > 1 for entrainment

# fraction of time series to use for training, rest can be used to test prediction
# truncates training data as train_data[train_data .<= train_frac*train_data[end]]
train_frac = 1.0		# 1.0 means use all data for training

# n => number of proteins w/n matching mRNA, for 2n
# m=2 is min number of variables with n-m number of dummy dimensions
# first protein is output, second protein responds to light input
n = 3
m = 2
@assert n >= m
@assert m >= 2

# NN structure for NODE runs
ns = 5		# inner nodes per layer * num inputs
layers = 0	# number inner layers
# mid-layer activation: try mish, swish, x -> hill(1,2,abs(x))
act_in = mish
act_mid = mish
act_out = sigmoid		# x -> hill(1,2,abs(x)) or sigmoid
parallel = true

# no self connections by this algorithm
allow_self = true		# allow self connections
gr_type = 1				# 1 => random, 2 => cycle for use in matching repressilator 
tf_in_num = n			# should be <= n-1 if self avoided, <= if w/self
@assert tf_in_num <= n "allow_self and tf_in_num ($tf_in_num) greater than n ($n)"
@assert (allow_self || tf_in_num < n) "!allow_self and tf_in_num ($tf_in_num) greater than n-1"
# if (allow_self && tf_in_num > n)
# 	exit("allow_self and tf_in_num ($tf_in_num) greater than n ($n)")
# end
# if (!allow_self && tf_in_num >= n)
# 	exit("!allow_self and tf_in_num ($tf_in_num) greater than n-1 ($n)")
# end

# array of arrays, each entry array is the list of incoming TF connections for a gene
tf_in = 
  if gr_type == 2
	tf_in = circshift([[i] for i in 1:n],1)		# cycle_digraph, as in repressilator
  elseif allow_self
	[sort(sample(1:n, tf_in_num, replace=false)) for i in 1:n] # digraph w/tf_in_num in degree
  else
	[sort(sample(vcat(1:i-1,i+1:n) , tf_in_num, replace=false)) for i in 1:n] # no self
  end

opt_dummy_u0 = true	# optimize dummy init values instead of using rand values

# Larger tolerances are faster but errors make gradient descent more challenging
# However, fit is sensitive to tolerances
# with larger tolerances whereas ODE needs smoother gradient from smaller tolerances??
rtol = 1e-10		# relative tolerance for solver, ODE -> ~1e-10 or a bit less
atol = 1e-12		# absolute tolerance for solver, ODE -> ~1e-12 or a bit less
rtolR = 1e-10		# relative tolerance for solver for refine_fit stages
atolR = 1e-12		# absolute tolerance for solver for refine_fit stages
adm_learn = 0.0005	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 200		# max iterates for each incremental learning step
					# try 200 with small tolerances, and Rodas4P solver
					# and 500 for larger tolerances and TRBDF2
print_grad = false	# show gradient on terminal, requires significant overhead

# parameter bounds: lower bound is zero for all parameters except
# rates m_a, m_d, p_a, and p_d, which have lower bound of
days	= 1.0		# number of circadian cycles
steps_per_day = 50
save_incr = 1.0 / steps_per_day 

m_low_rate = 1e-4 * s_per_d
p_low_rate = 1e-3 * s_per_d
# upper bounds
m_rate 	= 1e-2 * s_per_d
p_rate 	= 1e0 * s_per_d
k		= 1e4
k_min	= 1e2
h		= 5e0
a		= 1e0
r		= 1e1
p_max	= [m_rate,p_rate,k,h,a,r]
max_m	= m_rate / m_low_rate
# protein production rate in response to light, via fast post-translation
# modification or allostery, base max rate via mRNA is p_rate * max_m
# where max_m is max mRNA concentration
light_mult = 1e0
light_prod_rate	= light_mult * p_rate * max_m
# protein produced at max rate p_prate + from light stimulation at
# p_prate * light_mult * max_m
max_p	= p_rate * (1+light_mult) * max_m / p_low_rate
switch_level = 1e-2 * p_rate * max_m / p_low_rate

# values needed in ode_parse_p()
s = tf_in_num
N = 2^s
d = 1e-2
k1 = d*(1.0+exp(-10.0*d))
k2 = 10.0*(1.0-d) + log(d/(1.0-d))
ddim = opt_dummy_u0 ? 2*n : 0
num_param = use_node ? ddim+4n : ddim+4n + 2*n*s+n*N+n*(N - (s+1))
p_min = calc_pmin(n,s,num_param,ddim,m_low_rate,p_low_rate,k_min,use_node)
p_mult = calc_pmult(n,s,N,p_max,use_node)
bk = 4n
bh = bk+n*s
ba = bh+n*s
br = ba+n*N
ri = N - (s+1)

start_time = Dates.format(now(),"yyyymmdd_HHMMSS")
proj_dir = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF"
out_file = "/Users/steve/Desktop/" * start_time * ".jld2"

git_vers = chomp(read(`git -C $proj_dir rev-parse --short HEAD`,String))

# if true, program generates new random seed, otherwise uses rand_seed
generate_rand_seed = true
preset_seed = 0x0861a3ea66cd3e9a	# use if generate_rand_seed = false
actual_seed = set_rand_seed(generate_rand_seed, preset_seed)
set_rseed = set_rand_seed

# Training done iteratively, starting with first part of time series,
# then adding additional time series steps and continuing the fit
# process. For each step, the time points are weighted according
# to a 1 - cdf(Beta distribution). To move the weights to the right
# (later times), the first parameter of the beta distribution is
# increased repeatedly over i = 1:wt_incr:wt_steps, with the parameter
# equal to wt_base^i, with wt_incr determining the rate at which the
# incremental fits move to the "right" to include later times in the
# data series, larger wt_incr moves faster to the right

# Smaller values of wt_base move the weighting increments at a 
# slower pace and require more increments and longer run time, 
# but may gain by avoiding the common local minimum as 
# a simple regression line through the fluctuating time series.

# wt_steps is smallest integer such that wt_base^wt_steps >=500.
wt_base = 1.1		# good default is 1.1
wt_trunc = 1e-2		# truncation for weights
wt_steps = Int(ceil(log(500)/log(wt_base)))
wt_incr = 1			# increment for i = 1:wt_incr:wt_steps, see above

# would be worthwhile to experiment with various solvers
# see https://diffeq.sciml.ai/stable/solvers/ode_solve/
# 
# ODE solver, Tsit5() for nonstiff and fastest, but may be unstable.
# Alternatively use stiff solver TRBDF2(), slower but more stable.
# For smaller tolerances, if unstable try Rodas4P().
# Likely it is oscillatory dynamics that cause the difficulty.

# For SDE, ISSEM is only solver that seems to work, probably because of
# large size of noise terms
# 
# Might need higer adm_learn parameter with stiff solvers, which are
# more likely to get trapped in local minimum. 
# Maybe gradient through solvers differs ??
# Or maybe the stiff solvers provide less error fluctuation
# and so need greater learning momentum to shake out of local minima ??

solver = diffusion ? ISSEM() : Tsit5()	# Rodas4P() or Tsit5()

end # struct

function calc_pmin(n,s,pnum,ddim,m_low_rate,p_low_rate,k_min,use_node)
	p_min = zeros(pnum-ddim)
	p_min[1:2n] .= m_low_rate .* ones(2n)
	p_min[2n+1:4n] .= p_low_rate .* ones(2n)
	if !use_node p_min[4n+1:4n+n*s] .= k_min .* ones(n*s) end
	p_min
end

function calc_pmult(n,s,N,p_max,use_node)::Vector{Float64}
	p_dim = use_node ? [2n,2n] : [2n,2n,n*s,n*s,n*N,n*(N-(s+1))]
	p_mult = []
	for i in 1:length(p_dim)
		append!(p_mult, p_max[i] .* ones(p_dim[i]))
	end
	p_mult
end

end	# module