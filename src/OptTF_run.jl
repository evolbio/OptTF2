using OptTF

# see OptTF for additional notes
S = default_ode();

# L is struct that includes u0, ode_data, tsteps, see struct loss_args for other parts
# A is struct that includes tsteps_all and prob_all, used if S.train_frac < 1 that
# splits data into initial training period and later period used to compare w/prediction
# use noise_wait=2 to test switching on and off of light signal, 1000 to keep signal off
p_opt1,L,A  = fit_diffeq(S; noise=0.5, noise_wait=2.0, hill_k_init=2.0);
p_opt1,L,A  = fit_diffeq(S; noise=0.5, noise_wait=1000.0, hill_k_init=2.0);

# If using a subset of data for training, then need L_all with full time period for all data
# L always refers to training period, which may or may not be all time steps

tf, re, state, _ = generate_tf_node(S);			# must regenerate tf function
L = loss_args(L; tf=tf, re=re, state=state);
L_all = (S.train_frac < 1) ? make_loss_args_all(L, A) : L;
loss1, _, _, GG1, pred1 = loss(p_opt1,S,L);		# use if p_opt2 fails or p_opt1 of interest
use_2 = false;	# set to false if p_opt2 fails, true if p_opt2 is good

p_opt2 = refine_fit_bfgs(p_opt1,S,L)			# or p_opt2=p_opt1;
loss2, _, _, GG2, pred2 = loss(p_opt2,S,L);
use_2 = true;	# set to false if p_opt2 fails, true if p_opt2 is good

# run bfgs a second time if desired
# p_opt2 = refine_fit_bfgs(p_opt2,S,L)

# bfgs sometimes fails, if so then use p_opt1 or try repeating refine_fit
# p_opt2 = refine_fit(p_opt1,S,L)
# and again if needed
# p_opt2 = refine_fit(p_opt2,S,L)

# see definition of refine_fit() for other options to refine fit
# alternatively, may be options for ode solver and tolerances that would allow bfgs

p, loss_v, GG, pred = use_2 ? (p_opt2, loss2, GG2, pred2) : (p_opt1, loss1, GG1, pred1);

# save results
save_data(p, S, L, GG, L_all, loss_v, pred; file=S.out_file)

# test loading
dt_test = load_data(S.out_file);
keys(dt_test)

# If OK, then move out_file to standard location and naming for runs
f_name = "sde-4_8_t4.jld2"
mv(S.out_file, S.proj_dir * "/output_node/" * f_name)
# then delete temporary files
tmp_list = readdir(S.proj_dir * "/tmp/",join=true);
rm.(tmp_list[occursin.(S.start_time,tmp_list)]);

# if gradient is of interest
grad = calc_gradient(p,S,L)
gnorm = sqrt(sum(abs2, grad))

# change time period or training period, retrain
S, L, L_all, G = remake_days_train(p_opt1, S, L; days=12, train_frac=1/2);
p_opt2 = refine_fit(p_opt1,S,L)
# save as above with different file name

# increase hill_k and retrain
L = OptTF.loss_args(L; hill_k=5.0);
p_opt2 = refine_fit(p_opt2,S,L)
# save as above with different file name

###################################################################
# some plots from saved file

using OptTF, DifferentialEquations

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/";
basefile = proj_output * "output_node/SDE_tol3_3/sde-4_1_t4";
dt = load_data(basefile * ".jld2");				# may be warnings for loaded functions
S = dt.S;
ff = generate_tf_activation_f(dt.S.tf_in_num);
tf, re, state, _ = generate_tf_node(S);			# must regenerate tf function
L = loss_args(dt.L; tf=tf, re=re, state=state, f=ff);

# choose one for deterministic or stochastic plots
S = Settings(dt.S; diffusion=false, batch=1, solver=Tsit5());
S = Settings(dt.S; diffusion=true, batch=5, solver=ISSEM());

# set time period
S, L, L_all, G = remake_days_train(dt.p, S, L; days=2*S.days, train_frac=S.train_frac/2);
# alternatively, set fixed period
new_days = 36;
new_train_frac = dt.S.train_frac / (new_days / dt.S.days);
S, L, L_all, G = remake_days_train(dt.p, S, L; days=new_days, 
										train_frac=new_train_frac);

# plots
plot_stoch(dt.p, S, L, G, L_all; samples=10, display_plot=true)	# increase samples as needed

loss_all, _, _, G_all, pred_all = loss(dt.p,S,L_all);
plot_callback(loss_all, S, L_all, G_all, pred_all, true; no_display=false)


###################################################################
# Load results and complete optimization using OptTF, DifferentialEquations

using OptTF, DifferentialEquations
proj_dir = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/";
basefile = proj_dir * "output_node/node-8_1_t4";
dt = load_data(basefile * ".jld2");				# may be warnings for loaded functions
S = dt.S;
ff = generate_tf_activation_f(dt.S.tf_in_num);
tf, re, state, _ = generate_tf_node(S);			# must regenerate tf function
L = loss_args(dt.L; tf=tf, re=re, state=state, f=ff);

S = Settings(dt.S; diffusion=false, batch=1, solver=Tsit5());

# look at some plots, not necessary, change days and samples as needed
S, L, L_all, G = remake_days_train(dt.p, S, L; days=2*S.days, train_frac=S.train_frac/2);
plot_stoch(dt.p, S, L, G, L_all; samples=1, display_plot=true) # deterministic or stoch

loss_all, _, _, G_all, pred_all = loss(dt.p,S,L_all);
plot_callback(loss_all, S, L_all, G_all, pred_all, true; no_display=false)

# if stochastic then use following
S = Settings(dt.S; diffusion=true, batch=12, solver=ISSEM());

# for fitting to 6 days
S, L, L_all, G = remake_days_train(dt.p, S, L; days=12, train_frac=1/2);

# optimize and save
p_opt2 = refine_fit(dt.p,S,L; rate_div=2.0, iter_mult=1.0)

loss_v, _, _, GG, pred = loss(p_opt2,S,L);
save_data(p_opt2, S, L, GG, L_all, loss_v, pred; file=basefile * ".jld2")

# alter hill coefficient, optimize and save
L = OptTF.loss_args(L; hill_k=5.0);
p_opt2 = refine_fit(dt.p,S,L)

loss_v, _, _, GG, pred = loss(p_opt2,S,L);
save_data(p_opt2, S, L, GG, L_all, loss_v, pred; file=basefile * "_t6_h5.jld2")

###################################################################
# Load tmp file and then save with full jld2 data for plotting

using OptTF, DifferentialEquations

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/tmp/";
basef = "20220720_084507_17";
basefile = proj_output * basef;
dt = load_data(basefile * ".jld2");				# may be warnings for loaded functions
S = dt.S;
ff = generate_tf_activation_f(dt.S.tf_in_num);
tf, re, state, _ = generate_tf_node(S);			# must regenerate tf function
L = loss_args(dt.L; tf=tf, re=re, state=state, f=ff);

loss_v, _, _, GG, pred = loss(dt.p,S,L);
S, L, L_all, G = remake_days_train(dt.p, S, L; days=S.days, train_frac=S.train_frac);
save_data(dt.p, S, L, GG, L_all, loss_v, pred; file= basefile * "_test.jld2")

# plot directory must have "tmp" subdirectory otherwise will fail
save_summary_plots("../tmp/" * basef * "_test"; samples=1000,
					plot_dir="/Users/steve/Desktop/plots/tmp/");

plot_percentiles([basef * "_test"]; data_dir="/Users/steve/Desktop/plots/tmp/",
				use_duration=false, show_days=[10,20,30])

# change noise_wait and redo
L = loss_args(L; noise_wait=1000.0);
loss_v, _, _, GG, pred = loss(dt.p,S,L);
S, L, L_all, G = remake_days_train(dt.p, S, L; days=S.days, train_frac=S.train_frac);
save_data(dt.p, S, L, GG, L_all, loss_v, pred; file= basefile * "_test_w1000.jld2")
save_summary_plots("../tmp/" * basef * "_test_w1000"; samples=1000,
					plot_dir="/Users/steve/Desktop/plots/tmp/");

plot_percentiles([basef * "_test_w1000"]; data_dir="/Users/steve/Desktop/plots/tmp/",
				use_duration=false, show_days=[10,20,30])

###################################################################
# Look at optimized parameters

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/";
file = "stoch-4-4_1.jld2"; 			# fill this in with desired file name
dt = load_data(proj_output * file);					# may be warnings for loaded functions
idx = dt.S.opt_dummy_u0 ? dt.S.ddim+1 : 1
PP=ode_parse_p(dt.p[idx:end],dt.S);

###################################################################
# refine fit with jumps [not tested recently]

proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/";
file = "circad-4_2.jld2"; 							# fill this in with desired file name
dt = load_data(proj_output * file);					# may be warnings for loaded functions
S = Settings(dt.S; jump=true, batch=6);
L = OptTF.loss_args(dt.L; hill_k=50.0);

S = Settings(dt.S; jump=true, batch=6, jump_rate=5e-5 * S.s_per_d, adm_learn=1e-3);
w, L, A = setup_refine_fit(dt.p,S,L);
p_opt1 = refine_fit(dt.p,S,L);

###################################################################
# Benchmark main calculations

using OptTF, DifferentialEquations, BenchmarkTools, ForwardDiff, Profile

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

# uses Zygote, often fails, slower than ForwardDiff for smaller length(p)
using Zygote,  SciMLSensitivity
@btime Zygote.gradient(p->loss(p,S,L)[1], p)[1];


########################### Stochastic runs evaluation ###########################

# examples using save_summary_plots from OptTF_plots.jl

save_summary_plots("SDE_tol3_3/sde-4_1_t4"; samples=1000,
						plot_dir="/Users/steve/Desktop/plots/");

files = ["SDE_tol3_3/sde-4_$(i)_t4" for i in 1:4];

save_summary_plots.(files; samples=1000, plot_dir="/Users/steve/Desktop/plots/");

files = ["sde-4_$(i)_t4" for i in 1:8];
plot_percentiles(files;
			data_dir="/Users/steve/Desktop/plots/SDE_tol3_3/", use_duration=false,
			show_days=[10,20,30]);

using StatsPlots.PlotMeasures

in_dir = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output_node/SDE_tol3_3/";
plt=plot_w_range(["sde-4_1_t4"]; file_labels=[""], samples=1000, in_dir=in_dir,
				ylim=(-4.4,4.4), show_days=[10,20,30], bottom_trim=70px)

###################

save_summary_plots.(["circad-5-5_1_t6", "circad-6-6_2_t6"]);

save_summary_plots.(["circad-3-2_1_t6", "circad-3-2_2_t6", "circad-3-2_3_t6", "circad-3-3_1_t6", "circad-3-3_2_t6", "circad-3-3_3_t6", "circad-3-3_4_t6", "circad-3-3_5_t6", "circad-4-4_1_t6", "circad-4-4_2_t6", "circad-4-4_3_t6", "circad-5-5_1_t6", "circad-5-5_2_t6", "circad-5-5_3_t6", "circad-5-5_4_t6", "circad-5-5_w2_stoch_1_t6", "circad-6-6_1_t6", "circad-6-6_2_t6", "circad-4-4_2_stoch", "circad-4-4_2_stoch_t6_h5", "stoch-3-3_1_t6", "stoch-3-3_2_t6", "stoch-4-4_2_t6_h5", "stoch-4-4_3_t6"], samples=1000);

# some of the best
plot_percentiles(["circad-3-2_1_t6", "circad-3-3_5_t6", "circad-4-4_2_t6", "circad-4-4_3_t6", "circad-5-5_3_t6", "circad-6-6_1_t6", "circad-4-4_2_stoch", "circad-4-4_2_stoch_t6_h5", "stoch-3-3_1_t6", "stoch-4-4_2_t6_h5", "stoch-4-4_3_t6"], show_days=[5,10,15])

plt = plot_percentiles(["stoch-4-4_1_w2_34_test", "stoch-4-4_1_w2_w4_t8",
			"stoch-4-4_1_w2_w6_t12"];
			data_dir="/Users/steve/Desktop/plots/tmp/", use_duration=false,
			show_days=[10,20,30]);

plt=plot_w_range(["stoch-4-4_1_w2", "stoch-4-4_1_w2_w4_t8", "stoch-4-4_1_w2_w6_t12"];
		file_labels=["w2", "w4_t8", "w6_t12"], samples=1000, show_days=[10,20,30])

using StatsPlots.PlotMeasures

plt=plot_w_range(["stoch-4-4_1_w2"]; file_labels=[""], samples=1000,
                       ylim=(-4.4,4.4), show_days=[10,20,30], bottom_trim=70px)

plt=plot_tf_4_onepage("stoch-4-4_1_w2", p_focal=1)

################### Approx Bayes, split training and prediction ##################

using OptTF, OptTF_bayes, OptTF_settings, Plots, StatsPlots

# If reloading data needed
proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/";
#train_time = "60";						# e.g., "all", "60", "75", etc
#train = "train_" * train_time * "/"; 	# directory for training period
train = "";
file = "circad-4-4_3.jld2"; 				# fill this in with desired file name
bfile = proj_output * train * "bayes-" * file;
dfile = proj_output * train * file;

dt = load_data(dfile);					# check loaded data vars with keys(dt)
ff = generate_tf_activation_f(dt.S.tf_in_num);
LL = OptTF.loss_args(dt.L; f=ff);

# Train pSGLD on train_frac*days period, with plotting over days
# For use dt.S.days and dt.S.train_frac for saved values in data file
S, L, L_all, G = remake_days_train(dt.p, dt.S, LL; days=12, train_frac=1/2);

# If calculating approx bayes posterior, start here
# If loading previous calculations, skip to load_bayes()

# If first call to psgld_sample gives large loss, may be that gradient
# is small causing large stochastic term, try using pre_λ=1e-1 or other values
# Good values for a vary, try a=5e-5
B = pSGLD(warmup=5000, sample=10000, a=5e-5, pre_λ=1e-8);

losses, parameters, ks, ks_times = psgld_sample(dt.p, S, L, B);

save_bayes(B, losses, parameters, ks, ks_times; file=bfile);

# If loading previous results from psgld_sample(), skip previous three steps
bt = load_bayes(bfile);					# check loaded data vars with keys(bt)

# trajectories sampled from posterior parameter distn
plot_traj_bayes(bt.parameters, S, L, L_all, G; samples=20)

# plot full dynamics from original parameters
loss_all, _, _, G_all, pred_all = loss(dt.p,S,L_all);
callback(p, loss_all, S, L_all, G_all, pred_all)

# plot full dynamics from random parameter combination
p = bt.parameters[rand(1:length(bt.parameters))];
loss_all, _, _, G_all, pred_all = loss(p,S,L_all);
callback(p, loss_all, S, L_all, G_all, pred_all)

############

# look at decay of epsilon over time
plot_sgld_epsilon(15000; a=bt.B.a, b=bt.B.a, g=bt.B.g)

# plot loss values over time to look for convergence
plot_moving_ave(bt.losses, 300)
plot_autocorr(bt.losses, 1:20)		# autocorrelation over given range

# compare density of losses to examine convergence of loss posterior distn
plot_loss_bayes(bt.losses; skip_frac=0.0, ks_intervals=10)

# parameters
pr = p_matrix(bt.parameters);		# matrix rows for time and cols for parameter values
pts = p_ts(bt.parameters,8);		# time series for 8th parameter, change index as needed
density(pts)						# approx posterior density plot

# autocorr
autoc = auto_matrix(bt.parameters, 1:30);	# row for parameter and col for autocorr vals
plot_autocorr(pts, 1:50)			# autocorrelation plot for ts in in pts, range of lags
plot(autoc[8,:])					# another way to get autocorr plot for 8th parameter
plot_autocorr_hist(bt.parameters,10)	# distn for 10th lag over all parameters


######################## Distributed processing ###############################
# in general
# ret = @spawnat PROC_# COMMAND
# val = fetch(ret)

using Distributed

# tunnel needed for fisher, not alice, not sure why
addprocs([("fisher",1)]; exename=`/usr/local/bin/julia`,tunnel=true,
			env=["JULIA_DEPOT_PATH"=>"/opt/julia","JULIA_NUM_THREADS"=>"6"],
			exeflags=`--project=/Users/steve/sim/zzOtherLang/julia/projects/OptTF`)
addprocs([("alice2",1)]; exename=`/usr/local/bin/julia`,
			env=["JULIA_DEPOT_PATH"=>"/opt/julia","JULIA_NUM_THREADS"=>"6"],
			exeflags=`--project=/Users/steve/sim/zzOtherLang/julia/projects/OptTF`)

procs()		# rmprocs(NUMBER) or rmprocs([vector of numbers])

@everywhere push!(LOAD_PATH, "src/")
@everywhere using OptTF, OptTF_settings

ret = @spawnat 2 default_ode();
S = fetch(ret);

ret = @spawnat 2 fit_diffeq(S;noise=0.5, noise_wait=1000.0, hill_k_init=2.0);

