srun --account nvr_av_end2endav \
    --partition interactive,polar,grizzly -t 03:59:00 \
    --cpus-per-task=160 \
    --exclusive \
    -J debug \
    --container-image="/lustre/fsw/portfolios/nvr/users/yuewang/workspace/dockers/pytorch:24.01-py3.sqsh" \
    --container-mounts=/lustre/:/lustre/ \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:8 \
    --pty /bin/bash

rm /usr/local/cuda;
cd /usr/local/;
ln -s /lustre/fsw/portfolios/nvr/users/yuewang/workspace/cuda-12.1 cuda;
. /lustre/fsw/portfolios/nvr/users/yuewang/workspace/anaconda3/etc/profile.d/conda.sh;
conda activate /lustre/fsw/portfolios/nvr/users/yuewang/workspace/envs/camvla;
# conda activate /lustre/fsw/portfolios/nvr/users/yuewang/workspace/envs/rlds_env;
cd /lustre/fsw/portfolios/nvr/users/yuewang/stoRm/camvla;

export TORCH_HOME=/lustre/fsw/portfolios/nvr/users/yuewang/workspace/cache/torch_home;
export HF_HOME=/lustre/fsw/portfolios/nvr/users/yuewang/workspace/cache/hf_home;
export PYTHONPATH=$PWD;
export WANDB_API_KEY=c77be41cc79d6dda886279d4793150907790661c