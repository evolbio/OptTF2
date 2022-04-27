module OptTF_data
using Catalyst, DifferentialEquations, Plots
export generate_repressilator

# from https://catalyst.sciml.ai/dev/tutorials/using_catalyst/#Mass-Action-ODE-Models
function generate_repressilator(S)
	repressilator = @reaction_network Repressilator begin
		hillr(P₃,α,K,n), ∅ --> m₁
		hillr(P₁,α,K,n), ∅ --> m₂
		hillr(P₂,α,K,n), ∅ --> m₃
		(δ,γ), m₁ <--> ∅
		(δ,γ), m₂ <--> ∅
		(δ,γ), m₃ <--> ∅
		β, m₁ --> m₁ + P₁
		β, m₂ --> m₂ + P₂
		β, m₃ --> m₃ + P₃
		μ, P₁ --> ∅
		μ, P₂ --> ∅
		μ, P₃ --> ∅
	end α K n δ γ β μ
	@parameters  α K n δ γ β μ
	@variables t m₁(t) m₂(t) m₃(t) P₁(t) P₂(t) P₃(t)
	pmap  = (α => .5, K => 40, n => 2, δ => log(2)/120, 
		      γ => 5e-3, β => 20*log(2)/120, μ => log(2)/60)
	u₀map = [m₁ => 0., m₂ => 0., m₃ => 0., P₁ => 20., P₂ => 0., P₃ => 0.]
	
	odesys = convert(ODESystem, repressilator)
	save_incr = 10.
	total_steps = 2000
	tspan_total = (0., total_steps*save_incr)
	save_steps = 1000
	tspan_save = (0., save_steps*save_incr)
	tsteps_save = 0.:save_incr:save_steps*save_incr
	oprob = ODEProblem(repressilator, u₀map, tspan_total, pmap)
	# rows 4:6 are proteins
	# first 10000 time intervals as warmup, return second period
	# size is (3,1001) for three proteins at 1001 timepoints
	# and 1000 save_incr steps
	# for fitting, probably sufficient to use subset of about 350 pts
	data = solve(oprob, Tsit5(), saveat=10.)[4:6,total_steps-save_steps:total_steps]
	u0 = data[:,1]
	if (S.opt_dummy_u0 == false)
		prot_dum = u0[1].*ones(S.n-S.m)
		prot_init = vcat(u0,prot_dum)
		mRNA_init = 0.1 .* prot_init
		u0 = S.use_node ?
			# vcat(u0,rand(S.n-S.m)) :					# random init
			# vcat(rand(S.n),u0,rand(S.n-S.m))
			prot_init : 
			vcat(mRNA_init,prot_init)					# see OptTF.init_ode_param
	end
	return data, u0, tspan_save, tsteps_save
end

function plot_repressilator_time(sol; show_all=false)
	stop = (show_all) ? length(sol[1,:]) : 300
	display(plot([sol[i,1:stop] for i in 1:3],yscale=:log10))
end

function plot_repressilator_total(sol)
	display(plot([sum(sol[:,j]) for j in 1:length(sol[1,:])],yscale=:identity))
end

plot_repressilator_phase(sol) =
		display(plot([log2.(Tuple([sol[i,j] for i in 1:3])) for j in 1:length(sol[1,:])],
			camera=(50,60), linewidth=2, color=mma[1], limits=(5.2,8.8),
			ticks=(5.644:8.644,string.([50,100,200,400]))))

end	# module