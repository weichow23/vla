setup="
rm /usr/local/cuda;
cd /usr/local/;
ln -s /lustre/fsw/portfolios/nvr/users/yuewang/workspace/cuda-12.1 cuda;
. /lustre/fsw/portfolios/nvr/users/yuewang/workspace/anaconda3/etc/profile.d/conda.sh;
conda activate /lustre/fsw/portfolios/nvr/users/yuewang/workspace/envs/camvla;
cd /lustre/fsw/portfolios/nvr/users/yuewang/stoRm/camvla;
export TORCH_HOME=/lustre/fsw/portfolios/nvr/users/yuewang/workspace/cache/torch_home;
export HF_HOME=/lustre/fsw/portfolios/nvr/users/yuewang/workspace/cache/hf_home;
export PYTHONPATH=$PWD;
export WANDB_API_KEY=c77be41cc79d6dda886279d4793150907790661c;
"

exp_name=stoRm
submit_job --nodes 1 --gpu 8 --cpu 80 --mem 0 --account nvr_av_foundations \
    --partition polar,polar3,polar4,grizzly --tasks_per_node=1 \
    -n $exp_name --duration 4 \
    --exclusive \
    --mounts='/lustre/:/lustre/' \
    --image='/lustre/fsw/portfolios/nvr/users/jiawyang/dockers/pytorch:24.01-py3.sqsh' \
    --command "${setup} bash \
    data_preprocessing/run_preproc.sh"
