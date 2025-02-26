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
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
pip install plotly
```

## Main functions

### Extract ee tracks

```
python data_preprocessing/ee_tracks_extractionv2_tfds.py \
    --dataset_name bridge_orig \
    --trainval train \
    --data_path /PATH/TO/OXE \
    # --mul_cam \ # whether to calibrate all cameras in the dataset
    --meta_path data_preprocessing/meta_data/ \ # where to save the estimated camera poses
    --vis_path vis/ \ # where to save the videos
```

For multiple GPUs, run the above script multiple times with different gpu_id.
```
python data_preprocessing/ee_tracks_extractionv2_tfds.py \
    --num_episodes 53192 \ # used to split the dataset into multiple parts
    --gpu_id 0 \
    --data_path /PATH/TO/OXE \
```