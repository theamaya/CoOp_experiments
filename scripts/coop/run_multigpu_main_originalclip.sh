#!/bin/bash
#
# SBATCH --exclude=p3-r52-a.g42cloud.net
# SBATCH --exclude=p4-r68-a.g42cloud.net
# SBATCH --exclude=p4-r67-b.g42cloud.net
#SBATCH --job-name=VL-LTR
#SBATCH --output=./slurm_out_multig/%J.out
# SBATCH --partition=multigpu
#SBATCH --partition=multigpu
#SBATCH --account=mbzuai
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
# SBATCH --gpus=16
#SBATCH --gpus=16
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

DATASET=$1 # imagenetLT , imagenet
CFG=$2  # config file rn50
CTP=end  # $3  # class token position (end or middle)
NCTX=8  # $4  # number of context tokens 8
SHOTS=0  # $5  # number of shots (1, 2, 4, 8, 16) 0
CSC=False  # $6  # class-specific context (False or True)  True - class-specific
RUN=$3 # give a name to the run

for SEED in 1 #2 3
do
    DIR=output/${DATASET}/${TRAINER}/${RUN}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        /nfs/users/ext_amaya.dharmasiri/miniconda3/envs/dassl/bin/python \
        -u train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --resume ${DIR}/prompt_learner \
        --run ${RUN} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done