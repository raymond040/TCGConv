#!/bin/bash
#PBS -P Project
#PBS -j oe
#PBS -q volta_gpu
#PBS -l select=1:ncpus=8:mem=80gb:ngpus=1
#PBS -l walltime=12:00:00

cd #PBS_o_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);

image="/home/svu/e0407728/SIF/edge-hpc_v0.1.sif"
singularity exec -e $image bash << EOF > $PBS_JOBID.$PBS_JOBNAME.out 2> $PBS_JOBID.$PBS_JOBNAME.err

python3 "/hpctmp/e0407728/FYP/Ver1-26Jul/HPC_Scripts/Baseline3_F_HPC.py" --dataset "CC" --percentage 1 --type_ED "sub" --num_batches 100 --root "/hpctmp/e0407728/FYP/Ver1-26Jul/" --num_version $PBS_JOBID --seed $PBS_JOBID 
EOF
