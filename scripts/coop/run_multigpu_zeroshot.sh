#!/bin/bash
#
#SBATCH --exclude=p3-r52-a.g42cloud.net
#SBATCH --exclude=p4-r68-a.g42cloud.net
#SBATCH --exclude=p4-r67-b.g42cloud.net
#SBATCH --job-name=VL-LTR
#SBATCH --output=./slurm_out_multig/%J.out
#SBATCH --partition=multigpu
#SBATCH --account=mbzuai
#SBATCH --time=6-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64

# #echo "Module = $(module avail)"

echo "Python Interpreter = $(which python)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

NCCL_LL_THRESHOLD=0

echo "which nvcc = " $(which nvcc)

echo "nvcc --version = " $(nvcc --version)

#echo "NCCL_DEBUG = " $NCCL_DEBUG

echo "CUDA_VISIBLE_DEVICES = " $CUDA_VISIBLE_DEVICES

# echo "NCCL_LL_THRESHOLD = " $NCCL_LL_THRESHOLD


cd /nfs/users/ext_amaya.dharmasiri/repos/CoOp

set -x

# custom config
DATA=/nfs/users/ext_amaya.dharmasiri/repos/VL-LTR/data
TRAINER=ZeroshotCLIP
DATASET=$1 #imagenet imagenet_LT
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

/nfs/users/ext_amaya.dharmasiri/miniconda3/envs/dassl/bin/python \
-u train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only