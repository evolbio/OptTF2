module OptTF_data
using Plots
export generate_circadian

# True control input is sin wave for daily cycle with period of one
# Noisy input has intermittent signal loss and perhaps added noise
# Noise_wait arg value is expected neg binomial time for shift between
# active signal and signal loss
function generate_circadian(S; offset=true, noise_wait=0.0)
	steps_per_day = 100
	save_incr = 1.0 / steps_per_day 
	tspan_save = (0., S.days)
	tsteps_save = 0.:save_incr:S.days
	offset_value = offset ? 2π*rand() : 0.0
	input_true = [(sin(2π*t+offset_value) + 1.0) / 2.0 for t in tsteps_save]
	# use neg binomial mean of p/(1-p) for mean number of steps before switch for p not switch
	if noise_wait > 0.0
		n = length(input_true)
		mask = ones(n)
		x = noise_wait / save_incr
		prob_no_switch = x / (1+x)		# from negative binomial mean
		for i in 2:n
			if rand() < prob_no_switch
				mask[i] = mask[i-1]
			else
				mask[i] = (mask[i-1] == 1.0) ? 0.0 : 1.0
			end
		end
		input_noisy = mask .* input_true
	else
		input_noisy = copy(input_true)
	end
	return input_true, input_noisy, tspan_save, tsteps_save
end

function plot_circadian_input(input, tsteps)
	display(plot(tsteps, input))
end

end	# module