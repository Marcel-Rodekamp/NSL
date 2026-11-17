# executable and scripts
export PATH_TO_NSL_AUTOTUNER=$PROJECT_cjjsc37/sinilkov/NSL_Executables/autotuner_hasenbusch.py   # point to autotuner executable
export PATH_TO_NSL_HMC=$PROJECT_cjjsc37/sinilkov/NSL_Executables/example_autotune_MCMC   # point to hmc executable
export PATH_TO_NSL_ONE=$PROJECT_cjjsc37/sinilkov/NSL_Executables/example_k_correlators  # point to one-body executable
export PATH_TO_NSL_TWO=$PROJECT_cjjsc37/sinilkov/NSL_Executables/example_2_correlators   # point to two-body executable

# run parameters
export BANK=jjsc37
export PARTITION=dc-cpu-devel     # set to dc-cpu-devel for testing
export WALLTIME=02:00:00     # for devel testing on jureca, max walltime is 2:00:00
export STOP_MARGIN=600       # seconds before the walltime at which NSL is asked to stop safely (SIGUSR1);
                             # must exceed the time of one trajectory / one measured configuration
export MYEMAIL=...

