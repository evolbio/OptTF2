# OptTF2: Overview

Source code for manuscript:

Frank, S. A. 2022. An enhanced transcription factor repressilator that buffers stochasticity and entrains to an erratic external circadian signal. bioRxiv [2022.10.10.511622](https://doi.org/10.1101/2022.10.10.511622). 

by Steven A. Frank, https://stevefrank.org

[License CC-By 4.0](https://creativecommons.org/licenses/by/4.0/)

---

Only on Zenodo at https://doi.org/10.5281/zenodo.7178508, the directories output/ and analysis/ contain all parameters, output data, and plots for runs used in the manuscript. Zenodo also includes the GitHub code tagged as version tf2_zenodo_1.0.

[GitHub](https://github.com/evolbio/OptTF2) has the source code along with this file but without the output/ and analysis/ directories. Small updates will be posted on GitHub without updating the Zenodo version. In other words, GitHub is the best place for the source code, and Zenodo is the best place for the extra files in the output_node/ and analysis/ directories.

# Getting started with the code

## Look at an earlier version first

It might be easier to get started with a prior version that is a bit simpler and has instructions for deterministic runs that finish relatively quickly. The prior version is at https://github.com/evolbio/OptTF. After trying out that code, following the directions in the README file that comes with that code, you can return here. The following instructions do not include the simple deterministic examples found in the prior instructions.

The main difference between the prior code and the code here is that the prior version used a thermodynamic model for transcription factor binding and regulation, whereas this version uses a neural network for the transcription factor input-output function. See the manuscripts associated with the two versions.

## Julia setup and installing the code

See the tutorials for [getting started with Julia](https://julialang.org/learning/).

My code is based on Julia version 1.8.2. Current Julia versions are on the site's [download page](https://julialang.org/downloads/#upcoming_release). Follow the install instructions and, to use the command line in a terminal, make sure you have a link from an executable path to the binary as described in the instructions.

Next, download the code from the GitHub repository described above. Change into that directory with the code. Then

​	```julia -q --project=.```

to start up. Some of the code uses multiple execution threads. You can check the number of threads you have by default at the julia prompt with ```Threads.nthread()```. For stochastic runs, I often used 12 execution threads plus an additional controller thread. To raise the number of threads available, startup with the -t option, for example

​	```julia -q -t 13 --project=.```

If you are new to Julia, there is a bit of a learning curve, but the following should work. The next step is to download all of the required packages. The file in the top directory of the code, Project.toml, lists the packages needed for this code. Those packages have several other package dependencies, listed in Manifest.toml, along with the version numbers that I used in the current git version.

To load those packages, type the character ']', which puts you into package management mode. Then type ```instantiate```, and things should get started. After that finishes, type ```resolve```, and see how things work out. Sometimes you have to repeat those two commands. And sometimes an inconsistency may show up. If there is a problem, usually upgrading to the latest packages solves it, but that puts you out of sync with the packages associated with my code. To upgrade, type ```up```, then repeat the instantiate and resolve commands. Finally, to escape back to the julia command prompt, type the ```backspace``` key, or if that does not work, then check the Julia documentation to figure out what works for your keyboard.

## Test example

In the file src/OptTF_settings.jl, near the top, make sure the function default_ode() looks like this:

```julia
default_ode() = Settings(
	n	= 4,
	rtol = 1e-3,
	atol = 1e-3,
	adm_learn = 0.002,
	days = 6.0,
	train_frac = 2/3,
	max_it = 150,
	jump = false,
	diffusion = true,
	batch = 12
)
```

Then type

​	```using Revise```

which might trigger a message saying that package is not available. If so, then type the ```]``` character to get back into the package manager, then ```add Revise```, and then the ```Backspace``` key to get back to the julia prompt. Revise automatically reloads files after you make revisions to the source code. It usually works but if there is a problem, either reload the files manually as shown next or sometimes you must ```exit()``` and start over, although that is rare.

Next, following the steps shown at the top of the file src/OptTF_run.jl, 

```julia
using OptTF
S = default_ode();
p_opt1,L,A = fit_diffeq(S; noise=0.5, noise_wait=2.0, hill_k_init=2.0);
```

Copy and paste these lines one at a time. The first line loads the code. The second line loads the parameters and settings. You can type ```S``` and return to see all of the settings from the file src/OptTF_settings.jl. There are a lot of them, which you can change in the file and then rerun the second line. The third line starts an optimization run. Optimizing the stochastic differentiation equation takes a long time. However, you should see the progress of intermediate steps on the command line and in a graphics window.

Before starting, you should also have a look at the next section on Default directories. You may need to make some changes before having a successful run.

There will be various delays as Julia compiles the code, which can take up to a few minutes for each pause. That is normal and to some people rather irritating. The advantage is that the compiled code runs very fast relative to an interpreted language such as Python.

If you run the last line above to completion, you will have the optimized parameters in p_opt1, and some other key aspects of the run in L and A, which are needed for further analysis. On my computer, a 2022 Apple Studio Ultra M1, finishing a complete single run takes about ten days. However, you can see some progress toward fitting the circadian pattern after several hours to one day.

The various code lines in src/OptTF_run.jl provide many useful things that can be done to refine the fit, make many graphics to analyze the runs, etc. For a full understanding, you will have to read the source code and then experiment with the commands.

## Check default directories

If you run various code lines in src/OptTF_run.jl, you will need to keep an eye on the file system directory defaults used in the code. You can set some of the defaults in src/OptTF_settings.jl by changing the proj_dir variable, but other parts of the code may override that. If there is a problem, trace the location in the code. When Julia puts out an error, there is a seemingly unreadable dump to the terminal. However, start at the top and go down the listing until you see  the first filename that is in the src/ directory. Usually that line of code, or in the dump the next one down that mentions a line for the src/ files, will show you where the problem is.

For example, you may get an error soon after starting the example above, because the code writes a temporary file with intermediate results after each major iterate. The directory location must be correct for that to work. 

If the attempt to write the intermediate results is causing you a problem, you can comment out the lines from src/OptTF.jl

```julia
			tmp_file = S.proj_dir * "/tmp/" * S.start_time * iter * ".jld2"
			rm(tmp_file; force=true)
```

# Sample output and graphics

To start, get the output and analysis directories from Zenodo at the link above. Then follow along the commands in the various sections of src/OptTF_run.jl.

The README file from prior version at https://github.com/evolbio/OptTF had more extensive examples of how to work with the code. If you have trouble with the current version, you might try the prior version first to get a sense of how things work. Then come back to the newer code, which has similar structure.
