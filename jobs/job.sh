#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=NNGP_launch
#SBATCH --mail-user=lhalice@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/lhalice/CALM_NNGP/jobs/logs/launch_%j.log


# Run the script
python -u CL_Driver.py --no_checkpoint --log_every_epoch --epochs=120 --lambda_rec=1.0 --GP_train_size_per_class=2000 --GP_test_size_per_class=1000 --GP_train_otc_size=50 --GP_num_indcpts=1000 --GP_package=laGP