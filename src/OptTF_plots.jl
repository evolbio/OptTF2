using Interpolations, Roots, Logging, Distributions, Plots, StatsPlots,
		StatsPlots.PlotMeasures

function plot_callback(loss_val, S, L, G, pred_all, show_all; no_display=false)
	lw = 2
	len = length(pred_all[1,:])
	ts = @view L.tsteps[1:len]
	day = 0.5:1:ts[end]
	night = 1:1:ts[end]
	pred = @view pred_all[S.n+1:S.n+S.m,:]
	
	log10_yrange = 4
	log10_switch = log10(S.switch_level)
	log10_bottom = log10_switch - (log10_yrange / 2)
	# for example, 1og10 range of 4 and switch at 1e3 yields range (1e1,1e5)
	yrange = (10^(log10_switch - log10_yrange/2), 10^(log10_switch + log10_yrange/2))
	if !show_all
		# plot current prediction against data, show only target proteins
		dim = length(pred[:,1])
		plt = plot(size=(600,400*dim), layout=(dim,1),left_margin=12px)
		for i in 1:dim
			plot!(plt, ts, pred[i,:], label = "", color=mma[1], subplot=i,
					yscale=(i==1 ? :log10 : :identity), ylim=(i==1 ? yrange : :auto),
					linewidth=lw)
		end
	else
		dim = length(pred_all[:,1])
		plt = plot(size=(1200,250*(dim ÷ 2)), layout=((dim ÷ 2),2),left_margin=12px)
		for i in 1:dim
			# proteins [S.n+1:2S.n] on left, mRNA [1:S.n] on right
			idx = vcat(2:2:dim,1:2:dim) 	# subplot index [2,4,6,...,1,3,5,...]
			plot!(plt, ts, pred_all[i,:], label = "", color=mma[1], subplot=idx[i],
				yscale=(idx[i]==1 ? :log10 : :identity),
				ylim=(idx[i]==1 ? yrange : :auto), linewidth=lw)
		end			
	end
	# output normalized by hill vs target normalized by hill
	target = 10.0.^((log10_yrange-0.1) * hill.(0.5,L.hill_k,G.input_true[1:len])
						.+ (log10_bottom + 0.05) * ones(len))
	output = 10.0.^((log10_yrange-0.1) *  hill.(S.switch_level,L.hill_k,pred[1,:])
						.+ (log10_bottom + 0.05) * ones(len))
	plot!(plt, ts, target, label = "", color=mma[2], subplot=1,
		yscale=:log10, ylim=yrange, linewidth=lw)
	plot!(plt, ts, output, label = "", color=mma[3], subplot=1,
		yscale=:log10, ylim=yrange, linewidth=lw)
	
	# show noisy signal, normalize height to max of associated protein concentration
	max_conc = maximum(pred[2,:])
	plot!(ts, max_conc * G.input_noisy[1:len], label = "", color=mma[2],
					subplot=((show_all) ? 3 : 2), linewidth=lw)
	# add vertical lines for day/night changes, horiz line in plot 1 for expression switch
	plot!([S.switch_level], seriestype =:hline, color = :black, linestyle =:dot, 
			linewidth=2, label=nothing, subplot=1)
	for i in 1:2
		subpl = (i == 1) ? 1 : ((show_all) ? 3 : 2)
		if length(day) > 0
			plot!(day, seriestype =:vline, color = :black, linestyle =:dot, 
				linewidth=2, label=nothing, subplot=subpl)
		end
		if length(night) > 0
			plot!(night, seriestype =:vline, color = :black, linestyle =:solid, 
				linewidth=2, label=nothing, subplot=subpl)
		end
	end
	(no_display) ? (return plt) : display(plot(plt))
end

function plot_stoch(p, S, L, G, L_all; samples=5, show_orig=false, display_plot=true)
	plt = plot(size=(1600,400))
	all_steps = L_all.tsteps
	ts = all_steps
	wp = 1.5
	tp = 1.5
	day = 0.5:1:all_steps[end]
	night = 1:1:all_steps[end]
	log10_yrange = 4
	log10_switch = log10(S.switch_level)
	log10_bottom = log10_switch - (log10_yrange / 2)
	idx = S.n+1	# first protein
	# for example, 1og10 range of 4 and switch at 1e3 yields range (1e1,1e5)
	yrange = (10^(log10_switch - log10_yrange/2), 10^(log10_switch + log10_yrange/2))
	local G_all
	for i in 1:samples
		loss_all, _, _, G_all, pred_all = loss(p,S,L_all)
		if show_orig
			plot!(ts,pred_all[idx,:], color=mma[1], linewidth=wp, label="", 
					ylims=yrange, yscale=:log10)
		end
		output = 10.0.^((log10_yrange-0.1)  	
						.* OptTF.hill.(S.switch_level,L.hill_k,pred_all[idx,:])
						.+ (log10_bottom + 0.05) .* ones(length(pred_all[idx,:])))
		plot!(plt, ts, output, color=mma[3], yscale=:log10, ylim=yrange, linewidth=0.75,
						label="")
	end
	# output normalized by hill vs target normalized by hill
	len = length(G_all.input_true)
	target = 10.0.^((log10_yrange-0.1) .* OptTF.hill.(0.5,L.hill_k,G_all.input_true[1:end])
						.+ (log10_bottom + 0.05) .* ones(len))
	plot!(plt, ts, target, color=mma[2], yscale=:log10, ylim=yrange, linewidth=3, label="")
	# add vertical line to show end of training
	if length(day) > 0
		plot!(day, seriestype =:vline, color = :black, linestyle =:dot, 
			linewidth=2, label=nothing)
	end
	if length(night) > 0
		plot!(night, seriestype =:vline, color = :black, linestyle =:solid, 
			linewidth=2, label=nothing)
	end
	plot!([S.switch_level], seriestype =:hline, color = :black, linestyle =:dot, 
			linewidth=2, label=nothing)
	train_end = length(L.tsteps)
	all_end = length(ts)
	if all_end > train_end
		plot!([ts[train_end]], seriestype =:vline, color = :red, linestyle =:solid,
					linewidth=tp, label="")
	end
	if display_plot display(plt) end
	return(plt)
end

# if running on remote with no display, save pdf graphic
function plot_temp(p, S, L; all_time=false)
	proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/tmp/";
	# file = S.start_time * ".pdf"
	file = "current.pdf"
	if all_time
		w, L, A = setup_refine_fit(p,S,L);
		L = (dt.S.train_frac < 1) ? make_loss_args_all(L, A) : L;
	end
	lossv, _, _, G, pred = loss(p,S,L);
	plt = OptTF.plot_callback(lossv, S, L, G, pred, true; no_display=true)
	savefig(plt, proj_output * file)
end

# calc deviations for entry into daytime and duration of daytime expression
function calc_stoch_dev_dur(p, S, L, G, L_all; samples=5)
	ts = L_all.tsteps
	deviation = NaN .* ones(samples, Integer(S.days))
	duration = NaN .* ones(samples, Integer(S.days) - 1)
	lk = ReentrantLock()
	Threads.@threads for i in 1:samples
		_, _, _, _, pred_all = loss(p,S,L_all)
		output = pred_all[S.n+1,:] .- S.switch_level
		pred = CubicSplineInterpolation(ts,output)
		roots = find_zeros(pred, 0.1, ts[end])		# skip zero at time = 0
		len = length(roots)
		curr = 0
		for j in 1:len
			r = roots[j]
			if curr < S.days && r < S.days - 1e-2 && pred(r+1e-2) > 0
				curr += 1
				lock(lk) do
					deviation[i,curr] = r - floor(r) - 0.5
					if curr < S.days && j < len
						duration[i,curr] = roots[j+1] - r - 0.5
					end
				end
			end
		end
	end
	return deviation, duration
end

# plot deviations for entry into daytime and duration of daytime expression
function plot_stoch_dev_dur(p, S, L, G, L_all; samples=5)
	deviation, duration = calc_stoch_dev_dur(p, S, L, G, L_all; samples=samples)
	return deviation, duration
end

cdf_data(data) = sort(data), (1:length(data))./length(data)

# examples
# save_summary_plots("circad-5-5_1_t6"; samples=1000, plot_dir="/Users/steve/Desktop/");
# save_summary_plots.(["circad-5-5_1_t6", "circad-6-6_2_t6"]);

function save_summary_plots(filebase; samples = 100, plot_dir="/Users/steve/Desktop/plots/")
	remove_nan!(v) = filter!(x -> !isnan(x), v)
	println("Making plots for $filebase")
	proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/";
	dt = Logging.with_logger(Logging.NullLogger()) do
   		load_data(proj_output * filebase * ".jld2");
	end
	ff = generate_tf_activation_f(dt.S.tf_in_num);
	L = loss_args(dt.L; f=ff);
	S = Settings(dt.S; diffusion=false, batch=1, solver=Tsit5());
	# plot deterministic dynamics w/standard callback
	S, L, L_all, G = remake_days_train(dt.p, S, L; days=2*S.days, train_frac=S.train_frac/2);
	plt = plot_stoch(dt.p, S, L, G, L_all; samples=1, display_plot=false)
	display(plt)
	savefig(plt, plot_dir * filebase * "_det_bias.pdf")

	loss_all, _, _, G_all, pred_all = loss(dt.p,S,L_all);
	plt = OptTF.plot_callback(loss_all, S, L_all, G_all, pred_all, true; no_display=true)
	savefig(plt, plot_dir * filebase * "_det_dyn.pdf")
	
	S = Settings(dt.S; diffusion=true, batch=5, solver=ISSEM());
	new_days = 36;
	new_train_frac = dt.S.train_frac / (new_days / dt.S.days);
	S, L, L_all, G = remake_days_train(dt.p, S, L; days=new_days, 
											train_frac=new_train_frac);
	deviation, duration = plot_stoch_dev_dur(dt.p, S, L, G, L_all; samples=samples);
	
	# plot mean and sd of deviations for time of entry in to daytime, in hours
	times = 1:length(deviation[1,:]);
	ave = mean.([remove_nan!(deviation[:,i])*24 for i in times]);
	sd = std.([remove_nan!(deviation[:,i])*24 for i in times]);
	plt = plot(times,ave,label=nothing)
	plot!(times,sd,label=nothing)
	savefig(plt, plot_dir * filebase * "_dev_sd.pdf")
	
	# plot mean and sd of duration in day state, in hours of deviation from 12h
	times = 1:length(duration[1,:]);
	ave = mean.([remove_nan!(duration[:,i])*24 for i in times]);
	sd = std.([remove_nan!(duration[:,i])*24 for i in times]);
	plt = plot(times,ave,label=nothing)
	plot!(times,sd,label=nothing)
	savefig(plt, plot_dir * filebase * "_dur_sd.pdf")
	
	# show cdf measured in hours
	plt = plot(cdf_data(remove_nan!(deviation[:,10]*24)), label="10")
	plot!(cdf_data(remove_nan!(deviation[:,20]*24)), label="20")
	plot!(cdf_data(remove_nan!(deviation[:,30]*24)), label="30")
	savefig(plt, plot_dir * filebase * "_dev_cdf.pdf")

	plt = plot(cdf_data(remove_nan!(duration[:,10]*24)), label="10")
	plot!(cdf_data(remove_nan!(duration[:,20]*24)), label="20")
	plot!(cdf_data(remove_nan!(duration[:,30,]*24)), label="30")
	savefig(plt, plot_dir * filebase * "_dur_cdf.pdf")
	
	# show densities measured in hours
	plt = density( deviation[:,10]*24, label="10")
	density!(deviation[:,20]*24, label="20")
	density!(deviation[:,30]*24, label="30")
	savefig(plt, plot_dir * filebase * "_dev_density.pdf")

	plt = density( duration[:,10]*24, label="10")
	density!(duration[:,20]*24, label="20")
	density!(duration[:,30]*24, label="30")
	savefig(plt, plot_dir * filebase * "_dur_density.pdf")
end
