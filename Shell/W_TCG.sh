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

python3 "/home/svu/e0407728/My_FYP/TCGConv/main.py" --dataset_name "W" --percentage 1 --type_ED "sub" --num_groups 24 --root "/home/svu/e0407728/My_FYP/TCGConv/" --num_version $PBS_JOBID --seed $PBS_JOBID --alpha 0.00387038222 --gamma 3 --lr 4.19e-5 --weight_decay 4.47e-6 --hidden_chnl 128 --dropout 0.5 --num_layers 2 --model_type "TCGConv"
EOF
