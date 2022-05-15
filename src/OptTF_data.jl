module OptTF_data
using Plots
export generate_circadian, circadian_val

# True control input is sin wave for daily cycle with period of one
# Noisy input has intermittent signal loss and perhaps added noise
# Noise_wait arg value is expected neg binomial time for shift between
# active signal and signal loss
function generate_circadian(S; offset=true, noise_wait=0.0)
	steps_per_day = 100
	save_incr = 1.0 / steps_per_day 
	tspan = (0., S.days)
	tsteps = 0.:save_incr:S.days
	offset_value = offset ? 2π*rand() : 0.0
	input_true = [(sin(2π*t+offset_value) + 1.0) / 2.0 for t in tsteps]
	noise = false
	# use neg binomial mean of p/(1-p) for mean number of steps before switch for p not switch
	breakpoints=Vector{Int64}(undef,0)
	breakvalues=Vector{Float64}(undef,0)
	if noise_wait > 0.0
		noise = true
		n = length(input_true)
		mask = ones(n)
		x = noise_wait / save_incr
		prob_no_switch = x / (1+x)		# from negative binomial mean
		switch = false
		for i in 2:n
			if rand() < prob_no_switch
				mask[i] = mask[i-1]
				switch = false
			else
				mask[i] = (mask[i-1] == 1.0) ? 0.0 : 1.0
				push!(breakpoints,(i>2) ? i-2 : i-1)
				push!(breakvalues,mask[i-1])
				switch = true
			end
		end
		if switch==false
			push!(breakpoints,n)
			push!(breakvalues,mask[n])
		end
		input_noisy = mask .* input_true
	else
		input_noisy = copy(input_true)
	end
	return (;input_true, input_noisy, noise, breakpoints, breakvalues, offset_value,
				steps_per_day, tspan, tsteps)
end

function plot_circadian_input(input, tsteps)
	display(plot(tsteps, input))
end

# continuous values as function of time
circadian_val(G,t) = ((sin(2π*t+G.offset_value) + 1.0) / 2.0) *
			(G.noise ? G.breakvalues[searchsortedfirst(G.breakpoints,t*G.steps_per_day)] : 1.0)

# test continuous circadian_val function
# G = generate_circadian(S;noise_wait=0.5);
# plot(G.tsteps,[circadian_val(G,t) for t in G.tsteps])
# plot!(G.tsteps,G.input_noisy)

end	# module