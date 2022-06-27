module OptTF_plots
using OptTF, Interpolations, Roots, Logging, Distributions, Plots, StatsPlots,
		StatsPlots.PlotMeasures, DifferentialEquations, DelimitedFiles, Statistics,
		Printf
export plot_callback, plot_stoch, plot_temp, plot_stoch_dev_dur, save_summary_plots,
		plot_tf, plot_percentiles, plot_w_range

# must have file extension
file_stem(file) = basename(file[1:findlast(isequal('.'),file)-1])
file_ext(file) = basename(file[findlast(isequal('.'),file)+1:end])

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
		plt = plot(size=(1200,250*(Int(floor(dim / 2)))),
					layout=((Int(floor(dim / 2))),2),left_margin=12px)
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

# return deviations for entry into daytime and duration of daytime expression
function plot_stoch_dev_dur(p, S, L, G, L_all; samples=5)
	deviation, duration = calc_stoch_dev_dur(p, S, L, G, L_all; samples=samples)
	return deviation, duration
end

cdf_data(data) = sort(data), (1:length(data))./length(data)
remove_nan!(v) = filter!(x -> !isnan(x), v)

# examples
# save_summary_plots("circad-5-5_1_t6"; samples=1000, plot_dir="/Users/steve/Desktop/");
# save_summary_plots.(["circad-5-5_1_t6", "circad-6-6_2_t6"]);
function save_summary_plots(filebase; samples = 100,
			plot_dir="/Users/steve/sim/zzOtherLang/julia/projects/OptTF/analysis/tmp/",
			proj_output = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/")
	println("Making plots for $filebase")
	dt = Logging.with_logger(Logging.NullLogger()) do
   		load_data(proj_output * filebase * ".jld2");
	end
	ff = generate_tf_activation_f(dt.S.tf_in_num);
	L = loss_args(dt.L; f=ff);
	S = Settings(dt.S; diffusion=false, batch=1, solver=Tsit5());
	# plot deterministic dynamics w/standard callback
	S, L, L_all, G = remake_days_train(dt.p, S, L; days=2*S.days, 
						train_frac=S.train_frac/2);
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
	writedlm(plot_dir * filebase * "_dev_data.dlm", deviation)
	writedlm(plot_dir * filebase * "_dur_data.dlm", duration)
	
	
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
	
	if (3 <= S.n <= 4)
		plot_tf(filebase; save_dir=plot_dir, proj_dir=proj_output)
end
	
# 	# show densities measured in hours
# 	plt = density( deviation[:,10]*24, label="10")
# 	density!(deviation[:,20]*24, label="20")
# 	density!(deviation[:,30]*24, label="30")
# 	savefig(plt, plot_dir * filebase * "_dev_density.pdf")
# 
# 	plt = density( duration[:,10]*24, label="10")
# 	density!(duration[:,20]*24, label="20")
# 	density!(duration[:,30]*24, label="30")
# 	savefig(plt, plot_dir * filebase * "_dur_density.pdf")
end

# plot tf input-output function, file is jld2 from output
# returns array of plots, size = 1 for S.n=3, size = 4 for S.n = 4
# to display the first plot: display(OptTF.plot_tf("file.jld2")[1])
# to save all plots, set save_dir
function plot_tf(file; save_dir="",
			proj_dir="/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/")
	if typeof(findlast(isequal('.'),file)) == Nothing
		file = file * ".jld2"	# add ext if not present, must be .jld2
	end
	println("plot_tf for ", file)
	if save_dir != ""
		realpath(save_dir)	# test if exists, throws error if not
		save_base = save_dir * "/" * file_stem(file)
	end
	dt = Logging.with_logger(Logging.NullLogger()) do
		load_data(proj_dir * file)
	end
	S = dt.S
	@assert 3 <= S.n <= 4 "Can plot only for S.n = 3 or 4"
	
	f = generate_tf_activation_f(S.tf_in_num)
	p = S.opt_dummy_u0 ? dt.p[2S.n+1:end] : dt.p
	pp = OptTF.OptTF_param.linear_sigmoid.(p, S.d, S.k1, S.k2)
	pp .= (pp .* S.p_mult) .+ S.p_min
	u_p = [992.2341095, 0.9566634, 1892.6723]
	b = 10.0
	d = 0:5
	d_num = length(d)
	incr = 0.025
	xx = 0:incr:5
	yy = 0:incr:5
	plt = []
	if S.n == 3
		pl = plot(size=(S.n*400,d_num*300),layout=(d_num,S.n),
				plot_title=file, plot_titlevspan=0.02)
		for i in 1:S.n
			for j in d
				zz = [OptTF.calc_f(f,pp,[b^x,b^y,10^j],S)[i] for x in xx, y in yy]
				surface!(xx, yy, zz, zrange=(0,1), subplot=i+S.n*j,
							colorbar=false,
							xlabel=(j==5) ? "p1" : "", ylabel=(j==5) ? "p2" : "",
							zlabel=(i==1) ? "p3=$j" : "",
							title=(j==0) ? "f$i" : "")
			end
		end
		push!(plt,pl)
	elseif S.n == 4
		display_plot = false
		for k in 1:d_num
			pl = plot(size=(S.n*400,d_num*300),layout=(d_num,S.n),
				plot_title="p4 = $k: $file", plot_titlevspan=0.02)
			for i in 1:S.n
				for j in d
					zz = [OptTF.calc_f(f,pp,[b^x,b^y,10^j,10^k],S)[i] for x in xx, y in yy]
					surface!(xx, yy, zz, zrange=(0,1), subplot=i+S.n*j,
								colorbar=false,
								xlabel=(j==5) ? "p1" : "", ylabel=(j==5) ? "p2" : "",
								zlabel=(i==1) ? "p3=$j" : "",
								title=(j==0) ? "f$i" : "")
				end
			end
			push!(plt, pl)
		end
	else
		@assert false "Should not be here"
	end
	if save_dir != ""
		num_plots = length(plt)
		compress_cmd = "/Users/steve/bin/pdfcompress.py"
		if num_plots == 1
			f = save_base * "_tf.pdf"
			savefig(plt[1], f)
			if isfile(compress_cmd) 
				run(`$compress_cmd -d $f`; wait=false)
			end
		else
			for i in 1:num_plots
				f = save_base * "_tf_$i.pdf"
				savefig(plt[i], f)
				if isfile(compress_cmd) 
					run(`$compress_cmd -d $f`; wait=false)
				end
			end
		end
	end
	return(plt)
end

# plot 5, 25, 50, 75, 95 %iles from delimited data saved in save_summary_plots
# give single filebase as "basename" or array as ["basename1", "basename2", ...]
# if array of names, then use plot_percentiles.() for map over names
# max number of show_days == 3
function plot_percentiles(filebase; file_labels=nothing,
				data_dir="/Users/steve/sim/zzOtherLang/julia/projects/OptTF/analysis/plots",
				use_duration=false, show_days=[10,20,30])
	file_vec = typeof(filebase) == String ? [filebase] : filebase
	@assert typeof(file_vec) == Vector{String}
	num_files = length(file_vec)
	num_show_days = length(show_days)
	@assert 1 <= num_show_days <= 3 "Cannot have more than 3 show_days"
	x_incr_files = 1 / (num_files + 1)
	x_incr_days = if num_show_days == 3
		[-x_incr_files / 4, 0, x_incr_files / 4]
	elseif num_show_days == 2
		[-x_incr_files / 8, x_incr_files / 8]
	elseif num_show_days == 1
		[0]
	else
		@assert false "Should not be here"
	end
	plt = plot(size=(num_files*100,550),xlim=(0,1),ylim=(-8,8),legend=false,
				grid=true, showaxis=:y, xticks=false, bottom_margin=150px)
	for i in 1:num_files
		f = file_vec[i]
		file_start = data_dir * "/" * f
		use = use_duration ? "dur" : "dev"
		file = file_start * "_$use" * "_data.dlm"
		data = readdlm(file)
		samples = length(data[:,1])
		days = length(data[1,:])
		for j in 1:num_show_days
			d = show_days[j]
			@assert d <= days "Total days of $days < requested show day of $d"
			y = quantile(remove_nan!(data[:,d]),[5,25,50,75,95] ./ 100)*24
			x = i * x_incr_files + x_incr_days[j]
			plot!([x,x], [y[1],y[2]], color=:black, linewidth=2)
			plot!([x,x], [y[4],y[5]], color=:black, linewidth=2)
			scatter!([x],[y[3]], color=:black)
		end
		if file_labels != nothing && num_files == length(file_labels)
			f = file_labels[i]
		else
			f = replace(f, "circad" => "C")
			f = replace(f, "stoch" => "S")
			f = replace(f, "_t6" => "")
		end
		annotate!(i * x_incr_files, -9.0, Plots.text(f, 11, rotation=-90, :left))
	end
	display(plt)
	return(plt)
end

# for each jld2 input, plot range of w=noise_wait values in %-tile plot
# first use save_summary_plots to generate raw plots and data
# then summarize with final percentile plot for deviation distns
# label_base length must match filebase list length, if labels given
function plot_w_range(filebase; file_labels = nothing, samples=1000,
			w_val = [2., 4., 8., 16., 1000.],
			in_dir="/Users/steve/sim/zzOtherLang/julia/projects/OptTF/output/",
			out_dir="/Users/steve/sim/zzOtherLang/julia/projects/OptTF/analysis/tmp/",
			use_duration=false, show_days=[10,20,30],
			display_plot=true)
	files = typeof(filebase) == String ? [filebase] : filebase
	@assert typeof(files) == Vector{String}
	num_files = length(files)
	file_labels = typeof(file_labels) == String ? [file_labels] : file_labels
	@assert file_labels == nothing || num_files == length(file_labels) "Check file labels"
	new_basefiles = Vector{String}(undef,0)
	new_labels = file_labels == nothing ? nothing : Vector{String}(undef,0)
	for i in 1:num_files
		dt = load_data(in_dir * files[i] * ".jld2")
		ff = generate_tf_activation_f(dt.S.tf_in_num)
		for w in w_val
			S = dt.S
			L = loss_args(dt.L; f=ff, noise_wait=w)
			loss_v, _, _, GG, pred = loss(dt.p,S,L);
			S, L, L_all, G = remake_days_train(dt.p, S, L; days=S.days,
				train_frac=S.train_frac);
			base_w = files[i] * @sprintf("_w%04d", w)
			push!(new_basefiles, base_w)
			if file_labels != nothing
				push!(new_labels, file_labels[i] * @sprintf("_w%04d", w))
			end
			out_jld2 = out_dir * base_w * ".jld2"
			save_data(dt.p, S, L, GG, L_all, loss_v, pred; file=out_jld2)
			if !isfile(out_dir * base_w * "_dev_cdf.pdf")
				save_summary_plots(base_w; samples=samples,
					plot_dir=out_dir, proj_output=out_dir);
			end
			rm(out_jld2)
		end
	end
	
	plt = plot_percentiles(new_basefiles; file_labels=new_labels,
				data_dir=out_dir, use_duration=use_duration, show_days=show_days)
	if display_plot display(plt) end
	return plt
end

end	# module