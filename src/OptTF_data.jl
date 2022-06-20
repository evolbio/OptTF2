module OptTF_data
using Plots
export generate_circadian, circadian_val

struct Circadian
	input_true
	input_noisy
	noise
	breakpoints
	breakvalues
	offset_value
	tspan
	tsteps
	circadian_val
	steps_per_day
end

# True control input is sin wave for daily cycle with period of one
# Noisy input has intermittent signal loss and perhaps added noise
# Noise_wait arg value is expected neg binomial time in days for shift between
# active signal and signal loss
# init_on, if true, then signal on until random off, else off until random on
function generate_circadian(S; init_on=false, rand_offset=false, noise_wait=0.0)
	tspan = (0., S.days)
	tsteps = 0.:S.save_incr:S.days
	offset_value = rand_offset ? 2π*rand() : π
	input_true = [(sin(2π*t+offset_value) + 1.0) / 2.0 for t in tsteps]
	noise = false
	# use neg binomial mean of p/(1-p) for mean number of steps before switch for p not switch
	breakpoints=Vector{Int64}(undef,0)
	breakvalues=Vector{Float64}(undef,0)
	if noise_wait > 0.0
		noise = true
		n = length(input_true)
		mask = (init_on) ? ones(n) : zeros(n)
		x = noise_wait / S.save_incr
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
	steps_per_day = S.steps_per_day
	Circadian(input_true, input_noisy, noise, breakpoints, breakvalues, offset_value,
				tspan, tsteps, circadian_val, steps_per_day)
end

function plot_circadian_input(input, tsteps)
	display(plot(tsteps, input))
end

# continuous values as function of time
function circadian_val(G,t)
	first_val = searchsortedfirst(G.breakpoints,t*G.steps_per_day)
	past_end = (first_val > length(G.breakpoints)) ? true : false
	((sin(2π*t+G.offset_value) + 1.0) / 2.0) *
			((G.noise && !past_end) ? G.breakvalues[first_val] : 1.0)
end

# test continuous circadian_val function
# G = generate_circadian(S;noise_wait=0.5);
# plot(G.tsteps,[G.circadian_val(G,t) for t in G.tsteps])
# plot!(G.tsteps,G.input_noisy)

end	# module