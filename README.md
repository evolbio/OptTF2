# OptTF: Overview

Source code for manuscript:

Frank, S. A. 2022. Optimization of transcription factor genetic circuits. bioRxiv 2022.07.05.498863, [doi:10.1101/2022.07.05.498863](https://doi.org/10.1101/2022.07.05.498863).

by Steven A. Frank, https://stevefrank.org

[License CC-By 4.0](https://creativecommons.org/licenses/by/4.0/)

---

Only on Zenodo at https://doi.org/10.5281/zenodo.6798421, the directories output/ and analysis/ contain all parameters, output data, and plots for runs used in the manuscript plus many other sample runs. Zenodo also includes the GitHub code tagged as version zenodo_1.0. 

Directories output/ and analysis/ contain all parameters, output data, and plots for runs used in the manuscript plus many other sample runs are only available on Zenodo at https://doi.org/10.5281/zenodo.6798421, which also includes the GitHub code tagged as version zenodo_1.0. 

[GitHub](https://github.com/evolbio/OptTF) has the source code along with this file but without the output/ and analysis/ directories. Small updates will be posted on GitHub without updating the Zenodo version. In other words, GitHub is the best place for the source code, and Zenodo is the best place for the extra files in the output/ and analysis/ directories.

# Getting started with the code

## Julia setup and installing the code

See the tutorials for [getting started with Julia](https://julialang.org/learning/).

My code is based on Julia version 1.8.0-rc1. After the official release of 1.8.0, it may be difficult to find rc-1, but 1.8.0 should work. Current Julia versions are on their [download page](https://julialang.org/downloads/#upcoming_release). Follow the install instructions and, to use the command line in a terminal, make sure you have a link from an executable path to the binary as described in the instructions.

Next, download the code from the GitHub repository described above. Change into that directory that had the code. Then

​	```julia -q --project=.```

to start up. Some of the code uses multiple execution threads. You can check the number of threads you have by default at the julia prompt with ```Threads.nthread()```. For stochastic runs, I often used 12 execution threads plus an additional controller thread. To raise the number of threads available, startup with the -t option, for example

​	```julia -q -t 13 --project=.```

If you are new to julia, there is a bit of a learning curve, but the following should work. The next step is to download all of the required packages. The file in the top directory of the code, Project.toml, lists the packages needed for this code. Those packages have several other package dependencies, listed in Manifest.toml, along with the version numbers that I used in the current git version.

To load those packages, type the character ']', which puts you into package management mode. Then type ```instantiate```, and things should get started. After that finishes, type ```resolve```, and see how things work out. Sometimes you have to repeat those two commands. And sometimes an inconsistency may show up. If there is a problem, usually upgrading to the latest packages solves it, but that puts you out of sync with the packages associated with my code. To upgrade, type ```up```, the repeat the instantiate and resolve commands. Finally, to escape back to the julia command prompt, type the ```backspace``` key, or if that does not work, then check the julia documentation to figure out what works for your keyboard.

## Test example

In the file src/OptTF_settings.jl, near the top, make sure the function default_ode() looks like this:

```julia
default_ode() = Settings(
	allow_self = true,
	gr_type = 1,
	n = 4,
	tf_in_num = 4,
	rtol = 1e-4,
	atol = 1e-6,
	adm_learn = 0.002,
	days = 6.0,
	train_frac = 2/3,
	max_it = 200,
	opt_dummy_u0 = true,
	jump = false,
	diffusion = false,
	batch = 1
)
```

Then type

​	```using Revise```

which might trigger a message saying that package is not available. If so, then type the ```]``` character to get back into the package manager, then ```add Revise```, and then the ```Backspace``` key to get back to the julia prompt. Revise automatically reloads files after you make revisions to the source code. It usually works but if there is a problem, either reload the files manually as shown next or sometimes you must ```exit()``` and start over, although that is rare.

Next, following the steps shown at the top of the file src/OptTF_run.jl, 

```julia
using OptTF
S = default_ode();
p_opt1,L,A  = fit_diffeq(S;noise=0.5, noise_wait=1000.0, hill_k_init=2.0);
```

The first line loads the parameters and settings. You can type ```S``` and return to see all of the settings from the file src/OptTF_settings.jl. There are a lot of them, which you can change. The second line starts an optimization run. Using the above parameters, this should be an optimization of a deterministic system with no stochasticity in the dynamics or random perturbations (or a very rare perturbation from the line ```noise_wait=1000.0```, see the manuscript's description for the parameter *w*).

Before starting, you should also have a look at the next section on Default directories. You may need to make some changes before having a successful run.

There will be various delays as Julia compiles the code, which can take up to a few minutes for each pause. That is normal and to some people rather irritating. The advantage is that the compiled code runs very fast relative to an interpreted language such as Python. If all goes well, you will within a few minutes get a graphics window that shows the progress of the optimization that attempts to fit the system dynamics to a circadian pattern (see the manuscript). The complete run would take several hours, but you should see some progress within 10-20 minutes or less. For this particular set of parameters, sometimes the optimization goes off in a failing direction and does not converge to the circadian target, but more than half of attempts should work. If you don't like what you see, you can interrupt with ```cntrl-c``` and restart with the above commands. Sometimes repeated interruptions cause a problem with connections to the graphics window, in which case you have to quit Julia and start again.

If you run the last line above to completion, you will have the optimized parameters in p_opt1, and some other key aspects of the run in L and A, which are needed for further analysis.

The various code lines in src/OptTF_run.jl provide many useful things that can be done to refine the fit, make many graphics to analyze the runs, etc. For a full understanding, you will have to read the source code and experiment.

## Check default directories

If you run various code lines in src/OptTF_run.jl, you will need to keep an eye on the file system directory defaults used in the code. You can set some of the defaults in src/OptTF_settings.jl by changing the proj_dir variable, but other parts of the code may override that. If there is a problem, trace the location in the code. When Julia puts out an error, there is a seemingly unreadable dump to the terminal. However, start at the top and go down the listing until you see  the first filename that is in the src/ directory. Usually that line of code, or in the dump the next one down, will show you where the problem is.

For example, you may get an error soon after starting the example above, because the code writes a temporary file with intermediate results after each major iterate. The directory location must be correct for that to work. 

If the attempt to write the intermediate results is causing you a problem, you can comment out the lines from src/OptTF.jl

```julia
			tmp_file = S.proj_dir * "/tmp/" * S.start_time * iter * ".jld2"
			rm(tmp_file; force=true)
```

## More complex examples with stochasticity

The real power of the optimization in this code comes from its ability to optimize stochastic differential equations with automatic differentiation. Stochastic runs are significantly slower. Check your thread count as noted above before running. One can also study entrainment to a random day/night signal, as discussed in the manuscript. Here are a few brief examples.

```julia
default_ode() = Settings(
	allow_self = true,
	gr_type = 1,
	n = 4,
	tf_in_num = 4,
	rtol = 1e-4,
	atol = 1e-6,
	adm_learn = 0.002,
	days = 6.0,
	train_frac = 2/3,
	max_it = 200,
	opt_dummy_u0 = true,
	jump = false,
	diffusion = true,
	batch = 12
)
```

Using the above lines to start a run will initiate a stochastic run via the change to ```diffusion = true```, and increases the number of repeated trajectories in each round of calculating the derivative of the loss via automatic differentiation, in the ``batch = 12`` line. This run will proceed more slowly and can take days to finish. Runs often to not converge to anything close to a good tracking of the target circadian pattern. You can interrupt and try again. If you have succeeded in getting the intermediate results written to a tmp/ directory, you can interrupt if things look good and you don't want to wait any longer. You can then use the intermediate results by using the code below the comment ```# Load tmp file and then save with full jld2 data for plotting``` in the file src/OptTF_run.jl. 

Another thing to try is increasing the extrinsic and random day/night signal that provides an opportunity to entrain to daylight but also provides a challenge because the signal comes and goes randomly. To alter the average waiting time for the signal, change ```noise_wait=1000.0``` to a smaller values, for example, ```noise_wait=2.0```. The example in the manuscript used that value.

For additional things to explore, look at src/OptTF_run.jl.

# Sample output and graphics

To start, get the output and analysis directories from Zenodo at the [link above](#OptTF: Overview). The directory analysis/plots_publish/ contains detailed graphics related to the case study presented in the manuscript. All file names have a base part that describes the run. For example, for the file stoch-4-4_1_w2_w0002_dev_cdf.pdf, the base part is stoch-4-4_1_w2, which means that the run was stochastic and had an average waiting time of w=2 between random switching of the external light entrainment signal. This run was the example used in the manuscript. The part of the name w0002_dev_cdf describes the particular plot for that run, in this case, with w=2 during the collection of data for the plots, and measuring the cdf of the deviation in entry time into the daylight period.

Other files have names such as stoch-4-4_1_w2_w4_t8_w0002_dev_cdf.pdf. Here, the base is stoch-4-4_1_w2_w4_t8, which means that the original stoch-4-4_1_w2 was further optimized under conditions of w=4 and over a period of 8 days. Such further optimizations always reduced the performance of the original run. The reason for that remains an interesting puzzle, for which the solution may provide some insight into the optimization surface and other key aspects of the TF design.

## Loading a prior run for plotting and further analysis

For the runs in the output/ directory with extension .jld2, you can load the information, which includes all of the settings from the file src/OptTF_settings.jl, the optimized parameters, and some further information needed for additional analysis.

Following the steps in src/OptTF_run.jl, in the section *Load results and complete optimization*, we can analyze a prior run. For example, to load the data from the run analyzed as the test case in the manuscript, start with

```julia
using OptTF, DifferentialEquations
proj_dir = "/Users/steve/sim/zzOtherLang/julia/projects/OptTF/";
basefile = proj_dir * "output/stoch-4-4_1_w2";
dt = load_data(basefile * ".jld2");				# may be warnings for loaded functions
S = dt.S;
ff = generate_tf_activation_f(dt.S.tf_in_num);
L = loss_args(dt.L; f=ff);
```

Use an appropriate string for your proj_dir. The variable S contains the many settings that set up the run. For example, S.git_vers will return the git version of the code used for the run. That might be useful if you run into problems because the current code version differs from the version used for a stored run in output/. You could roll the code back to the git version that matched the run, or maybe one past that version in case the run was done before saving the most recent code changes in git. Many other settings are of interest. For example, S.batch will return the batch size for each loss calculation, in this case, 12.

Various combinations of the additional commands in the src/OptTF_run.jl code for this section provide ways to look at the loaded run.

## Detailed plots

Many detailed plots can be made with the code in src/OptTF_run.jl, in the section *Stochastic runs evaluation*. You can modify the examples shown. You will probably need to have a quick look at the functions in src/OptTF_plots.jl to call those functions correctly, get the file system directories right, find the output, etc. 

