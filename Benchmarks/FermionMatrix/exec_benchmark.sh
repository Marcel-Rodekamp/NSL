#!/bin/sh

#threads
threads=(1 2 4 8 16 32 64 128)

#directory for storing job scripts
jobdir=${HOME}/job
mkdir -p ${jobdir}

#directory for storing job outputs
output_dir=${HOME}/benchmark_out
mkdir -p ${output_dir}

#looping through each module
for thread in ${threads[@]}
do
echo $thread

done
