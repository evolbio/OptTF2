function plot_callback(loss_val, S, L, G, pred_all, show_all)
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
	display(plot(plt))
end