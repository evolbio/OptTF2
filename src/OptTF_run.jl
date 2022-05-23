using OptTF, OptTF_settings

# see FitODE for additional notes
S = default_ode();

# L is struct that includes u0, ode_data, tsteps, see struct loss_args for other parts
# A is struct that includes tsteps_all and prob_all, used if S.train_frac < 1 that
# splits data into initial training period and later period used to compare w/prediction
p_opt1,L,A  = fit_diffeq(S;noise=0.5, noise_wait=1000.0);

# If using a subset of data for training, then need L_all with full time period for all data
# L always refers to training period, which may or may not be all time steps
L_all = (S.train_frac < 1) ? make_loss_args_all(L, A) : L;

p_opt2 = refine_fit_bfgs(p_opt1,S,L)

# run bfgs a second time if desired
# p_opt2 = refine_fit_bfgs(p_opt2,S,L)

# bfgs sometimes fails, if so then use p_opt1 or try repeating refine_fit
# p_opt2 = refine_fit(p_opt1,S,L)
# and again if needed
# p_opt2 = refine_fit(p_opt2,S,L)

# see definition of refine_fit() for other options to refine fit
# alternatively, may be options for ode solver and tolerances that would allow bfgs

loss1, _, _, pred1 = loss(p_opt1,S,L);		# use if p_opt2 fails or p_opt1 of interest
loss2, _, _, pred2 = loss(p_opt2,S,L);

use_2 = true;	# set to false if p_opt2 fails, true if p_opt2 is good

p, loss_v, pred = use_2 ? (p_opt2, loss2, pred2) : (p_opt1, loss1, pred1);

# if gradient is of interest
grad = calc_gradient(p,S,L)
gnorm = sqrt(sum(abs2, grad))

# save results
save_data(p, S, L, L_all, loss_v, pred; file=S.out_file)

# test loading
dt_test = load_data(S.out_file);
keys(dt_test)

# If OK, then move out_file to standard location and naming for runs
f_name = "repress-5-4_1.jld2"
mv(S.out_file, S.proj_dir * "/output/" * f_name)
# then delete temporary files
tmp_list = readdir(S.proj_dir * "/tmp/",join=true);
rm.(tmp_list[occursin.(S.start_time,tmp_list)]);

# To use following steps, move saved out_file to proj_output using 
# example in following steps for naming convention

###################################################################
# Look at optimized parameters

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/";
file = "repress-5-4_1.jld2"; 						# fill this in with desired file name
dt = load_data(proj_output * file);					# may be warnings for loaded functions
idx = dt.S.opt_dummy_u0 ? S.ddim+1 : 1
PP=ode_parse_p(dt.p[idx:end],dt.S);

# plot final result
OptTF.callback(dt.p, dt.loss_v, dt.S, dt.L, dt.pred)

###################################################################
# Load intermediate results

proj_tmp = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/tmp/";
file = "20220513_045714_66.jld2"; 					# fill this in with desired file name
dtt = load_data(proj_tmp * file);					# may be warnings for loaded functions
S = dtt.S;
idx = S.opt_dummy_u0 ? S.ddim+1 : 1
PP=ode_parse_p(dtt.p[idx:end],S);

w, L, A = setup_refine_fit(dtt.p,S,dtt.L);
p_opt2 = p_opt1 = refine_fit(dtt.p,S,L);
L_all = (S.train_frac < 1) ? make_loss_args_all(L, A) : L;

# now can use commands from first section above

###################################################################
# Benchmark main calculations

using OptTF, OptTF_settings, DifferentialEquations, BenchmarkTools, DiffEqFlux,
	ForwardDiff, Profile
S=default_ode();

u0 = (1e4-1e3) * rand(2S.n) .+ (1e3 * ones(2S.n));
u0[1:S.n] .= 1e-2 * u0[S.n+1:2S.n]	# set mRNAs to 1e-2 of protein levels;
G = S.f_data(S; rand_offset=false, noise_wait=0.0);
predict = OptTF.setup_diffeq_func(S);
tsteps = tsteps_all = G.tsteps;
tspan = tspan_all = G.tspan;
if (S.train_frac < 1)
	tsteps = tsteps[tsteps .<= S.train_frac*tsteps[end]];
	tspan = (tsteps[begin], tsteps[end]);
end
p = OptTF.init_ode_param(u0,S; noise=0.1);
f = OptTF.generate_tf_activation_f(S.tf_in_num);
hill_k = 2.0;
w = ones(length(tsteps));
last_time = tsteps[length(w)];
ts = tsteps;
prob = ODEProblem((du, u, p, t) -> ode!(du, u, p, t, S, f, G), u0,
				(0.0,last_time), p, saveat = ts,
				reltol = S.rtol, abstol = S.atol);
L = OptTF.loss_args(u0,prob,predict,tsteps,hill_k,w,f,false,false,0.0);

@btime loss(p,S,L)[1];
@btime ForwardDiff.gradient(p->loss(p,S,L)[1], p)[1];

# uses Zygote, fails sometimes, slower than ForwardDiff for smaller length(p)
@btime gradient(p->loss(p,S,L)[1], p)[1];	




