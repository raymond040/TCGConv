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

python3 "/home/svu/e0407728/My_FYP/TCGConv/main.py" --dataset_name "CC" --percentage 1 --type_ED "sub" --num_groups 100 --root "/home/svu/e0407728/My_FYP/TCGConv/" --num_version $PBS_JOBID --seed $PBS_JOBID --alpha 0.0423596838172482 --dropout 0.5 --gamma 2 --lr 1.000e-04 --hidden_chnl 32 --num_layers 3 --model_type "CONCAT"
EOF
