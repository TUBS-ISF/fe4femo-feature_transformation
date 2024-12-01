#!/bin/bash
####################################
## simple Slurm submitter script to setup   ##
## a chain of jobs using Slurm                    ##
####################################
## ver.  : 2018-11-27, KIT, SCC

## Define maximum number of jobs via positional parameter 1, default is 5
max_nojob=25

## Define your jobscript (e.g. "~/chain_job.sh")
chain_link_job=$1
queue="${2:-single}"

## Define type of dependency via positional parameter 2, default is 'afterok'
dep_type="after"
## -> List of all dependencies:
## https://slurm.schedmd.com/sbatch.html

myloop_counter=0
## Submit loop
while [ ${myloop_counter} -le ${max_nojob} ] ; do
   ##
   ## Differ slurm_opt depending on chain link number
   if [ ${myloop_counter} -eq 0 ] ; then
      slurm_opt=""
   else
      slurm_opt="-d ${dep_type}:${jobID}"
   fi
   ##
   ## Print current iteration number and sbatch command
   echo "Chain job iteration = ${myloop_counter}"
   echo "   sbatch --export=myloop_counter=${myloop_counter} ${slurm_opt} ${chain_link_job}"
   ## Store job ID for next iteration by storing output of sbatch command with empty lines
   jobID=$(sbatch -p ${queue} --export=ALL,myloop_counter=${myloop_counter},OWNmodifier==${myloop_counter} --mail-type=END --mail-user="raphael.dunkel@uni-ulm.de" ${slurm_opt} ${chain_link_job} 2>&1 | sed 's/[S,a-z]* //g')
   ##
   ## Check if ERROR occured
   if [[ "${jobID}" =~ "ERROR" ]] ; then
      echo "   -> submission failed!" ; exit 1
   else
      echo "   -> job number = ${jobID}"
   fi
   ##
   ## Increase counter
   let myloop_counter+=1
done
