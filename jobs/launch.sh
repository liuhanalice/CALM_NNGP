#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=NNGP_launch
#SBATCH --mail-user=lhalice@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/lhalice/CALM_NNGP/jobs/logs/launch_%j.log

module purge

module load cuda/12.6.3
module load R/4.5.1

source /home/lhalice/miniconda3/etc/profile.d/conda.sh
conda activate NNGP

# Debug
echo "=== Python ==="
which python
python -V
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
nvidia-smi
echo "=== Conda Env ==="
conda info --envs
echo "=== GPU Info ==="
nvidia-smi
echo "=== R ==="
which R
R --version

# Run the script
python -u CL_Driver.py --log_every_epoch --epochs0=120 --epochs=120 --lambda_rec=1.0 --GP_train_size_per_class=3000 --GP_test_size_per_class=1000 --GP_train_otc_size=50 --GP_num_indcpts=60 --GP_package=laGP --num_replay=3000 --GP_package=laGP --ce_onall --retrain_all_gp --lambda_feat=0 --lambda_logit=0 --GP_score_threshold=0.7 --GP_sample_mode=inducing --GP_inducing_noise_scale=1.0