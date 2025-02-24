sudo apt-get update -yqq
sudo apt-get -yqq install libegl1-mesa libegl1
sudo apt-get -yqq install libgl1
sudo apt-get -yqq install libosmesa6-dev
sudo apt-get -yqq install patchelf

sudo apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
sudo apt-get install -yqq ffmpeg

# sudo mkdir -p /usr/share/vulkan/icd.d
# sudo cp -r simpler_setup_configs/vulkan/icd.d /usr/share/vulkan/
# sudo cp simpler_setup_configs/10_nvidia.json /usr/share/glvnd/egl_vendor.d/

# Install numpy<2.0 
pip install numpy==1.24.4 # 这个没用



git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules

SIMPLER_ROOT=$(pwd)/SimplerEnv

cd ${SIMPLER_ROOT}/ManiSkill2_real2sim
pip3 install -e .
cd ..
pip3 install -e . # 可能要修改setup.py中的python版本
pip3 install mediapy decord
cd ..
# 验证
python3 eval/simpler/env_test.py