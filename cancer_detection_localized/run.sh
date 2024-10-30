#!/bin/bash

#SBATCH -t 5-00:00:00                           #Time for the job to run (5days...)
#SBATCH --job-name=XXXXX                        #Name of the job
#SBATCH -n 1                                    #Number of tasks to be launched
#SBATCH -N 1                                    #Number of nodes required (can't run one task on seperated node)
#SBATCH --cpus-per-task 30                      #Number of CPUs required per task. 
#SBATCH --mem=500g                              #ram asked
#SBATCH --mail-type ALL                         #Send email on start/end
#SBATCH --mail-user XXXXXXX@fredhutch.org       #Where to send email
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --exclusive


python3 tissueidentification.py 
