#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A obbaran
#SBATCH --time=20:03:25
#SBATCH -J SIN
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --workdir=/truba/home/obbaran/withoutForgetting/FewShotWithoutForgetting
#SBATCH --mail-type=ALL
#SBATCH --mail-user=---orhun_baran06@hotmail.com
#SBATCH --output=/truba/home/obbaran/withoutForgetting/FewShotWithoutForgetting/slurm-%j.err
#SBATCH --error=/truba/home/obbaran/withoutForgetting/FewShotWithoutForgetting/slurm-%j.err
#SBATCH --gres=gpu:1

# Also you can use your own cuda libraries.
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64

#CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifier
#CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAvgN1
#CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAvgN5
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAvgN1
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAvgN5
