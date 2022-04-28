module OptTF_settings
using OptTF_data, Parameters, DifferentialEquations, Dates, Random, Graphs, SparseArrays
export Settings, default_ode, default_node, reset_rseed, recalc_settings

default_ode() = Settings(n=5, tf_in_num=4, rtol=1e-4, atol=1e-6, adm_learn=0.05, train_frac=0.5)
default_node() = Settings(use_node=true, rtol=1e-3, atol=1e-4, rtolR=1e-6, atolR=1e-8,
						max_it=500, solver = TRBDF2())
reset_rseed(S, rseed) = Settings(S; generate_rand_seed=false, preset_seed=rseed,
							actual_seed=set_rand_seed(false,rseed))

function set_rand_seed(gen_seed, preset_seed)
	rseed = gen_seed ? rand(UInt) : preset_seed
	Random.seed!(rseed)
	return rseed
end

# fix calculated settings, in case one setting changes must propagate to others
function recalc_settings(S)
	tmp = (S.tf_in_num == S.n-1) ? 1 : 2	# first row used only when tf_in_num == n-1
	graph = S.gr_type == 2 ?
		SparseArrays.sparse(cycle_digraph(n)) :
		SparseArrays.sparse(random_regular_digraph(S.n+1,S.tf_in_num,dir=:in))[tmp:end,tmp:end]
	tf_in = [findall(>(0),graph[:,i]) for i in 1:length(graph[1,:])]
	
	S = Settings(S; start_time = Dates.format(now(),"yyyymmdd_HHMMSS"),
			git_vers = chomp(read(`git -C $(S.proj_dir) rev-parse --short HEAD`,String)),
			actual_seed = set_rand_seed(S.generate_rand_seed, S.preset_seed),
			wt_steps = Int(ceil(log(500)/log(S.wt_base))))
	S = Settings(S; graph=graph, tf_in=tf_in)
	S = Settings(S; out_file = "/Users/steve/Desktop/" * S.start_time * ".jld2")
	return S
end

# One can initialize and then modify settings as follows
# S = default_node()	# default settings for NODE
# S = Settings(S; layer_size=50, activate=3, [ADD OTHER OPTIONS AS NEEDED])
# S = default_ode()		# default settings for ODE
# S = Settings(S; opt_dummy_u0 = true, [ADD OTHER OPTIONS AS NEEDED])
# See docs for Parameters.jl package

@with_kw struct Settings

use_node = false	# switch between NODE and ODE
layer_size = 20		# size of layers for NODE

# function to generate or load data for fitting
f_data = generate_repressilator

# fraction of time series to use for training, rest can be used to test prediction
# truncates training data as train_data[train_data .<= train_frac*train_data[end]]
train_frac = 1.0		# 1.0 means use all data for training

# n => number of variables in NODE, for ODE n is # proteins w/n matching mRNA, for 2n
# m is number of variables in target data, n>=m, with n-m number of dummy dimensions
n = 5
m = 3					# repressilator is 3, vary as needed

# with random_regular_digraph, row 1 has no out connections, so initially make
# matrix of size n+1 then delete first row and column
# no self connections by this algorithm
gr_type = 1				# 1 => random, 2 => cycle for use in matching repressilator 
tf_in_num = 4			# should be <= n-1 if self avoided, <= if w/self
@assert tf_in_num < n "tf_in_num ($tf_in_num) should be less than n ($n)"
tmp = (tf_in_num == n-1) ? 1 : 2	# first row used only when tf_in_num == n-1
graph = gr_type == 2 ?
	SparseArrays.sparse(cycle_digraph(n)) :
	SparseArrays.sparse(random_regular_digraph(n+1,tf_in_num,dir=:in))[tmp:end,tmp:end]
tf_in = [findall(>(0),graph[:,i]) for i in 1:length(graph[1,:])]

opt_dummy_u0 = false	# optimize dummy init values instead of using rand values

# Larger tolerances are faster but errors make gradient descent more challenging
# However, fit is sensitive to tolerances, seems NODE may benefit from fluctuations
# with larger tolerances whereas ODE needs smoother gradient from smaller tolerances??
rtol = 1e-10		# relative tolerance for solver, ODE -> ~1e-10, NODE -> ~1e-2 or -3
atol = 1e-12		# absolute tolerance for solver, ODE -> ~1e-12, NODE -> ~1e-3 or -4
rtolR = 1e-10		# relative tolerance for solver for refine_fit stages
atolR = 1e-12		# absolute tolerance for solver for refine_fit stages
adm_learn = 0.0005	# Adam rate, >=0.0002 for Tsit5, >=0.0005 for TRBDF2, change as needed
max_it = 200		# max iterates for each incremental learning step
					# try 200 for ODE, small tolerances, and Rodas4P solver
					# and 500 for NODE, with larger tolerances and TRBDF2
print_grad = false	# show gradient on terminal, requires significant overhead

start_time = Dates.format(now(),"yyyymmdd_HHMMSS")
proj_dir = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF"
out_file = "/Users/steve/Desktop/" * start_time * ".jld2"

git_vers = chomp(read(`git -C $proj_dir rev-parse --short HEAD`,String))

# if true, program generates new random seed, otherwise uses rand_seed
generate_rand_seed = true
preset_seed = 0x0861a3ea66cd3e9a	# use if generate_rand_seed = false
actual_seed = set_rand_seed(generate_rand_seed, preset_seed)

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
# 
# Might need higer adm_learn parameter with stiff solvers, which are
# more likely to get trapped in local minimum. 
# Maybe gradient through solvers differs ??
# Or maybe the stiff solvers provide less error fluctuation
# and so need greater learning momentum to shake out of local minima ??

solver = Rodas4P()

# Activation function:
# tanh seems to give good fit, perhaps best fit, and maybe overfit
# however, gradient does not decline near best fit, may limit methods that
# require small gradient near fixed point; identity fit not as good but
# gradient declines properly near local optimum with BFGS()

activate = 2	 # use one of 1 => identity, 2 => tanh, 3 => sigmoid, 4 => swish

end # struct

end	# module