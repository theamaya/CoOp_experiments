#!/bin/bash
#
#SBATCH --exclude=p3-r52-a.g42cloud.net
#SBATCH --job-name=VL-LTR
#SBATCH --output=./slurm_out/%J.out
#SBATCH --partition=default-short
#SBATCH --account=mbzuai
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
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
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=$1
CFG=$2 
PRETRAINED_BACKBONE=$3

for SEED in 1 #2 3
do
    /nfs/users/ext_amaya.dharmasiri/miniconda3/envs/dassl/bin/python \
    -u train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 50 \
    --eval-only \
    --pretrained_backbone ${PRETRAINED_BACKBONE} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done