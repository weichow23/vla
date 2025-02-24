sudo apt-get update -yqq

sudo apt-get -yqq install libegl1-mesa libegl1
sudo apt-get -yqq install libgl1
sudo apt-get -yqq install libosmesa6-dev
# sudo apt-get -yqq install patchelf

sudo apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
sudo apt-get install -yqq ffmpeg

# sudo mkdir -p /usr/share/vulkan/icd.d
# sudo cp -r simpler_setup_configs/vulkan/icd.d /usr/share/vulkan/
# sudo cp simpler_setup_configs/10_nvidia.json /usr/share/glvnd/egl_vendor.d/

conda install -c conda-forge gcc=12.1.0 gxx_linux-64 -y

pip install mediapy decord

# Install numpy<2.0 
pip install numpy==1.24.4

git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules

SIMPLER_ROOT=$(pwd)/SimplerEnv

cd ${SIMPLER_ROOT}/ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .