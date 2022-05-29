# copied from FitODE_bayes.jl, then modified
module OptTF_bayes
using OptTF, OptTF_settings, StatsPlots, StatsBase, HypothesisTests, Printf,
			JLD2, Parameters, ForwardDiff
export psgld_sample, save_bayes, load_bayes, plot_loss_bayes, plot_sgld_epsilon,
			plot_autocorr, plot_moving_ave, p_matrix, p_ts, auto_matrix, plot_traj_bayes,
			plot_autocorr_hist, pSGLD, remake_days_train
			
@with_kw struct pSGLD
	warmup =	2000
	sample =	5000
	a =			5e-3
	b =			1e4
	g =			0.35
	pre_beta =	0.9
	pre_λ =		1e-8
end

# Reset total days and training days for use in pSGLD
function remake_days_train(p, S, L; days=12, train_frac=0.5)
	S = Settings(S; days=(days|>Float64), train_frac=(train_frac|>Float64))
	ts_all = 0.0:S.save_incr:S.days
	ts = (S.train_frac < 1) ? ts_all[ts_all .<= S.train_frac*ts_all[end]] : ts_all
	tmp_L = OptTF.loss_args(L; tsteps=ts)
	_, L, A = OptTF.setup_refine_fit(p, S, tmp_L)
	L_all = (S.train_frac < 1) ? make_loss_args_all(L, A) : L;
	G = S.f_data(S)
	return S, L, L_all, G
end

# pSGLD, see Rackauckas22.pdf, Bayesian Neural Ordinary Differential Equations
# and theory in Li et al. 2015 (arXiv:1512.07666v1)

# Use p_sgld() below, however, useful to show original sgld() for background
# Code for SGLD in https://diffeqflux.sciml.ai/stable/examples/BayesianNODE_SGLD/
# However, that code incorrectly uses ϵ instead of sqrt(ϵ) for η, variance should be ϵ
function sgld(∇L, p, t; a = 2.5e-3, b = 0.05, g = 0.35)
    ϵ = a*(b + t)^-g
    η = sqrt(ϵ).*randn(size(p))
    Δp = .5ϵ*∇L + η
    p .-= Δp
end

# precondition SGLD (pSGLD), weight by m, see Li et al. 2015, with m=G in their notation
# Corrected bug in https://github.com/RajDandekar/MSML21_BayesianNODE, they failed
# to weight noise by sqrt.(ϵ.*m)
function p_sgld(∇L, p, t, m; a = 2.5e-3, b = 0.05, g = 0.35)
    ϵ = a*(b + t)^-g
    η = sqrt.(ϵ.*m).*randn(size(p))
    Δp = .5ϵ*(∇L.*m) .+ η
    p .-= Δp
end

# use to test parameters for setting magnitude of ϵ
sgld_test(t; a = 2.5e-3, b = 0.05, g = 0.35) = a*(b + t)^-g

function psgld_sample(p_in, S, L, B::pSGLD)

	p = deepcopy(p_in)		# copy to local variable, else p_in changed
	parameters = []
	losses = Float64[]
	ks = Float64[]
	ks_times = Int[]

	# initialize moving average for precondition values
	grd = ForwardDiff.gradient(p -> loss(p, S, L)[1], p)[1]
	precond = grd .* grd

	for t in 1:(B.warmup+B.sample)
		if t % 100 == 0 println("t = " , t) end
		grad = ForwardDiff.gradient(p -> loss(p, S, L)[1], p)[1]
		# precondition gradient, normalizing magnitude in each dimension
		precond *= B.pre_beta
		precond += (1-B.pre_beta)*(grad .* grad)
		m = 1 ./ (B.pre_λ .+ sqrt.(precond))
		p_sgld(grad, p, t, m; a=B.a, b=B.b, g=B.g)
		# start collecting statistics after initial warmup period
		if t > B.warmup
			tmp = deepcopy(p)
			curr_loss = loss(p, S, L)[1]
			append!(losses, curr_loss)
			append!(parameters, [tmp])
			println(@sprintf("%5.3e", curr_loss))
			if t % 100 == 0
				half = Int(floor((t-B.warmup) / 2))
				println("Sample timesteps = ", t - B.warmup)
				first_losses = losses[1:half]
				second_losses = losses[half+1:end]
				ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
				append!(ks, ks_diff.δ)
				append!(ks_times, t-B.warmup)
				# plot
				plt = plot(size=(600,400 * 2), layout=(2,1))
				density!(first_losses, subplot=1, plot_title="KS = "
						* @sprintf("%5.3e", ks_diff.δ) * ", samples per curve = "
						* @sprintf("%d", half), label="1st" )
				density!(second_losses, subplot=1, label="2nd")
				plot!(ks_times, ks, label="ks", subplot=2, legend=nothing)
				display(plt)
			end
		else
			println(@sprintf("%5.3e", loss(p, S, L)[1]))
		end
	end
	return losses, parameters, ks, ks_times
end

# Saving and loading results
save_bayes(B, losses, parameters, ks, ks_times; file="/Users/steve/Desktop/bayes.jld2") =
					jldsave(file; B, losses, parameters, ks, ks_times)

function load_bayes(file)
	bt = load(file)
	(B = bt["B"], losses = bt["losses"], parameters = bt["parameters"],
			ks = bt["ks"], ks_times = bt["ks_times"])
end

# utility functions for extracting data

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

# params is a vector with each entry for time as a vector of parameters
# this makes a matrix with rows for time and cols for parameter values
# see https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/14
p_matrix(params) = reduce(hcat,params)' 
p_ts(params, index) = p_matrix(params)[:,index]

# matrix with each row for parameter and col for autocorr vals
function auto_matrix(params, arange=1:50)
	pm = p_matrix(params)
	auto = [autocor(pm[:,i],arange) for i=1:length(pm[1,:])]
	return reduce(hcat,auto)'
end

# plotting functions

plot_sgld_epsilon(end_time; a=1e-1, b=1e4, g=0.35) = 
	display(plot([i for i=1:10:end_time], [sgld_test(i; a=a, b=b, g=g) for i=1:10:end_time], 
		yscale=:log10))

plot_autocorr(data, auto_range=1:100) = display(plot(autocor(data, auto_range)))

plot_autocorr_hist(param, lag) = display(histogram(auto_matrix(param, 1:30)[:,lag]))

plot_moving_ave(data, n) = 
	display(plot(n/2 .+ Array(1:(length(data)-(n-1))),
		[sum(@view data[i:(i+n-1)])/n for i in 1:(length(data)-(n-1))]))

function plot_traj_bayes(param, S, L, L_all, G; samples=20, show_orig=false)
	plt = plot(size=(1600,400))
	La = L_all
	ts = L_all.tsteps
	ws = 3
	wp = 1.5
	tp = 1.5
	day = 0.5:1:L_all.tsteps[end]
	night = 1:1:L_all.tsteps[end]
	log10_yrange = 4
	log10_switch = log10(S.switch_level)
	log10_bottom = log10_switch - (log10_yrange / 2)
	idx = S.n+1	# first protein
	# for example, 1og10 range of 4 and switch at 1e3 yields range (1e1,1e5)
	yrange = (10^(log10_switch - log10_yrange/2), 10^(log10_switch + log10_yrange/2))
	for i in 1:samples
		_, _, _, _, pred = loss(param[rand(1:length(param))],S,L_all)
		if show_orig
			plot!(ts,pred[idx,:], color=mma[1], linewidth=wp, label="", 
					ylims=yrange, yscale=:log10)
		end
		output = 10.0.^((log10_yrange-0.1)  	
						.* OptTF.hill.(S.switch_level,L.hill_k,pred[idx,:])
						.+ (log10_bottom + 0.05) .* ones(length(pred[idx,:])))
		plot!(plt, ts, output, color=mma[3], yscale=:log10, ylim=yrange, linewidth=0.75,
						label="")
	end
	# output normalized by hill vs target normalized by hill
	len = length(G.input_true)
	target = 10.0.^((log10_yrange-0.1) .* OptTF.hill.(0.5,L.hill_k,G.input_true[1:end])
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
	display(plt)
	return(plt)
end

function plot_loss_bayes(losses; skip_frac=0.0, ks_intervals=10)
	plt = plot(size=(600,800), layout=(2,1))
	start_index = Int(ceil(skip_frac*length(losses)))
	start_index = (start_index==0) ? 1 : start_index
	losses = @view losses[start_index:end]
	if length(losses) < 5*ks_intervals
		println("\nWARNING: number of losses < 5*ks_intervals\n")
	end
	half = Int(floor(length(losses)/2))
	first_losses = @view losses[1:half]
	second_losses = @view losses[half+1:end]
	ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
	density!(first_losses, subplot=1, plot_title="KS = "
			* @sprintf("%5.3e", ks_diff.δ) * ", samples per curve = "
			* @sprintf("%d", half), label="1st" )
	density!(second_losses, subplot=1, label="2nd")
	
	ks_times = Int[]
	ks = []
	for i in 1:(ks_intervals-1)
		last_index = Int(floor(Float64(length(losses)*i)/ks_intervals))
		ks_losses = @view losses[1:last_index]
		half = Int(floor(length(ks_losses)/2))
	 	first_losses = @view ks_losses[1:half]
		second_losses = @view ks_losses[half+1:end]
		ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
		append!(ks, ks_diff.δ)
		append!(ks_times, length(ks_losses))
	end
	# and finally for full data set
	i = ks_intervals
	last_index = Int(floor(Float64(length(losses)*i)/ks_intervals))
	ks_losses = @view losses[1:last_index]
	half = Int(floor(length(ks_losses)/2))
	first_losses = @view ks_losses[1:half]
	second_losses = @view ks_losses[half+1:end]
	ks_diff = ApproximateTwoSampleKSTest(first_losses, second_losses)
	append!(ks, ks_diff.δ)
	append!(ks_times, length(ks_losses))
	plot!(ks_times, ks, subplot=2, legend=nothing)
	display(plt)
end

end # module
