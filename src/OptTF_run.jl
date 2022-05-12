using OptTF, OptTF_settings

# see FitODE for additional notes
S = default_ode();

# L is struct that includes u0, ode_data, tsteps, see struct loss_args for other parts
# A is struct that includes tsteps_all and prob_all, used if S.train_frac < 1 that
# splits data into initial training period and later period used to compare w/prediction
p_opt1,L,A  = fit_diffeq(S;noise=0.1);

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

use_2 = true;	# set to false if p_opt2 fails

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
f_name = "repress-3-2_3.jld2"
mv(S.out_file, S.proj_dir * "/output/" * f_name)
# then delete temporary files
tmp_list = readdir(S.proj_dir * "/tmp/",join=true);
rm.(tmp_list[occursin.(S.start_time,tmp_list)]);

# To use following steps, move saved out_file to proj_output using 
# example in following steps for naming convention

###################################################################
# Look at optimized parameters

proj_output = S.proj_dir * "/output/";
file = "repress-3-1_1.jld2"; 				# fill this in with desired file name
dt = load_data(proj_output * file);
PP=ode_parse_p(dt.p[2dt.S.n-S.m+1:end],dt.S);		# assuming opt_dummy_u0 is true

# plot final result
OptTF.callback(dt.p, dt.loss_v, dt.S, dt.L, dt.pred)