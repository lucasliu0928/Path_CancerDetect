#!/bin/bash

#SBATCH -t 5-00:00:00                           #Time for the job to run (5days...)
#SBATCH --job-name=test                         #Name of the job
#SBATCH -n 1                                    #Number of tasks to be launched
#SBATCH -N 1                                    #Number of nodes required (can't run one task on seperated node)
#SBATCH --cpus-per-task 30                      #Number of CPUs required per task. 
#SBATCH --mem=400g                              #ram asked
#SBATCH --mail-type ALL                         #Send email on start/end
#SBATCH --mail-user jliu6@fredhutch.org         #Where to send email
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --exclusive


python3 wsi_inference_prob-map_json100and2.py 
