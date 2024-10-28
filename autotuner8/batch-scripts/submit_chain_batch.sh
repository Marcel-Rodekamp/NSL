#!/bin/bash

#Usage:
#make sure that ananconda and gnuplot are available and set paths
#set correct paths to the autotuner-scripts:
#	autotuner_hasenbusch.py, fit_masses.gp, comp-avg
#and the executable:
#	cns
#start this script with the arguments:
#	parameter file, basic name of the jobs, number of jobs

#this script does:
#	thermalise a given system and find an appropriate step size for the integration
#	analyse the system and find appropriate Hasenbusch parameters
#	find an appropriate step size with this parameters
#	run the executable with this parameters

#submit.batch.half.tuning tunes a system without Hasenbusch-parameters

function submit.batch.half.tuning {
    yaml=$1
    dirname=$2
    maxCfg=$3
    gpu=$4
    walltime=$WALLTIME
    partition=$PARTITION
    bank=$BANK
    email=$MYEMAIL
    
    cat <<EOF > reSubmit.batch
#!/bin/bash
#SBATCH --job-name="$dirname"
#SBATCH --partition=$partition
#SBATCH --account=$bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=block:block:block,Pack
#SBATCH --time=$walltime
#SBATCH --mail-user=$email
#SBATCH --mail-type=FAIL
#SBATCH --output=output/$dirname.%j
#SBATCH --error=output/$dirname-err.%j

source /p/scratch/cslnpp/rodekamp/Perylene/Perylene/bin/activate

# get local variables
source env.sh

autotuner=\${PATH_TO_NSL_AUTOTUNER}
executable_hmc=\${PATH_TO_NSL_HMC}
executable_one=\${PATH_TO_NSL_ONE}
executable_two=\${PATH_TO_NSL_TWO}

finalCfg=$(($maxCfg - 1))

maxCfgID=\$(h5ls $dirname.h5/markovChain | grep -o "\$finalCfg")
particle=\$(h5ls $dirname.h5/markovChain/\${finalCfg}/correlators/single | awk '\$1 == "particle" { print \$1 }')
hole=\$(h5ls $dirname.h5/markovChain/\${finalCfg}/correlators/single | awk '\$1 == "hole" { print \$1 }')
two=\$(h5ls $dirname.h5/markovChain/\${finalCfg}/correlators/ | awk '\$1 == "twobody" { print \$1 }')


echo "\$SLURM_JOB_ID"

echo -n "Host:"
hostname
echo -n "PWD:"
pwd

echo -n "Starting at: "
date

# submit job-dependent script if possible
if [ ! "\${maxCfgID}" ] || [ ! "\${particle}" ] || [ ! "\${hole}" ] || [ ! "\${two}" ]; then
    echo "sbatch --dependency=afterany:\${SLURM_JOB_ID} reSubmit.batch"
    sbatch --dependency=afterany:\${SLURM_JOB_ID} reSubmit.batch
fi

if [ ! "\${maxCfgID}" ]; then
    echo "Generating configurations"
    srun python \$autotuner ${yaml} \$executable_hmc $gpu
else
    if [ ! "\${particle}" ] || [ ! "\${hole}" ]; then
        echo "Generating One-Body correlation functions"
        srun \$executable_one --file ${yaml} $gpu
    else
        echo "Generating Two-Body correlation functions"
        srun \$executable_two --file ${yaml} $gpu
    fi
fi

echo -n "Ending at: "
date; 
EOF


    if ! (squeue -o %j | grep -q -F $dirname); then 

	sbatch reSubmit.batch

    else
	echo "WARNING! Job with name=$dirname already queued! Abort submission."
    fi 
}

source env.sh
submit.batch.half.tuning $1 $2 $3 $4
