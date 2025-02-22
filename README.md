## Environment set up
```
conda create -p /lustre/fsw/portfolios/nvr/users/yuewang/workspace/envs/
conda activate /lustre/fsw/portfolios/nvr/users/yuewang/workspace/envs/camvla

git clone https://github.com/Jay-Ye/camvla --recursive

cd third_party/sam2/
pip install -e .

pip install -e ".[notebooks]"

cd checkpoints && \
./download_ckpts.sh && \
cd ..

pip install tensorflow
pip install tensorflow_datasets
pip install gcsfs
pip install timm
pip install imageio
pip install imageio[ffmpeg]
pip install scipy
pip install wandb