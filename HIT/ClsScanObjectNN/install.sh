# # # Install CUDA Toolkit 10.2 on Ubuntu 22.04
# # Download CUDA Toolkit 10.2
# wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

# # Install CUDA Toolkit 10.2
# sh cuda_10.2.89_440.33.01_linux.run --override
# refer to the subsection 2.1 and 2.2 in https://zhuanlan.zhihu.com/p/198161777?from=groupmessage

# # Add CUDA Path to '~/.bashrc'
# export CUDA_HOME="/home/meicheng/cuda-10.2/"
# export PATH="/home/meicheng/cuda-10.2/bin:$PATH"
# export LD_LIBRARY_PATH="/home/meicheng/cuda-10.2/lib64"

# # Save Path
# source ~/.bashrc

# # # Install CUDA Toolkit 11.3 on Ubuntu 22.04
# # Download CUDA Toolkit 11.3

# # Install CUDA Toolkit 11.3
# sh cuda_11.3.0_465.19.01_linux.run --override
# refer to the subsection 2.1 and 2.2 in https://zhuanlan.zhihu.com/p/198161777?from=groupmessage

# # Add CUDA Path to '~/.bashrc'
# export CUDA_HOME="/home/meicheng/cuda-11.3/"
# export PATH="/home/meicheng/cuda-11.3/bin:$PATH"
# export LD_LIBRARY_PATH="/home/meicheng/cuda-11.3/lib64"

# # Save Path
# source ~/.bashrc

# # # Install gcc-8 and g++-8 on Ubuntu 22.04
# refer to https://askubuntu.com/questions/1446863/trying-to-install-gcc-8-and-g-8-on-ubuntu-22-04

# # Install gcc-8 on Ubuntu 22.04
# sudo apt update
# wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_amd64.deb
# wget http://mirrors.edge.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_amd64.deb
# wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb
# wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_amd64.deb
# wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_amd64.deb
# wget http://mirrors.kernel.org/ubuntu/pool/main/i/isl/libisl22_0.22.1-1_amd64.deb
# sudo apt install ./libisl22_0.22.1-1_amd64.deb ./libmpx2_8.4.0-3ubuntu2_amd64.deb ./cpp-8_8.4.0-3ubuntu2_amd64.deb ./libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb ./gcc-8-base_8.4.0-3ubuntu2_amd64.deb ./gcc-8_8.4.0-3ubuntu2_amd64.deb

# # Install g++-8 on Ubuntu 22.04
# wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb
# wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_amd64.deb
# sudo apt install ./libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb ./g++-8_8.4.0-3ubuntu2_amd64.deb

# # Add gcc-8 and g++-8 to '~/.bashrc'
# export CC="/usr/bin/gcc-8"
# export CXX="/usr/bin/g++-8"

# # Save Path
# source ~/.bashrc

# # # Configure Mirrors
# conda config --add channels https://mirrors.nju.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.nju.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# conda config --add channels https://mirrors.nju.edu.cn/anaconda/cloud/conda-forge/
# conda config --add channels https://mirrors.nju.edu.cn/anaconda/cloud/msys2/
# conda config --add channels https://mirrors.nju.edu.cn/anaconda/cloud/bioconda/
# conda config --add channels https://mirrors.nju.edu.cn/anaconda/cloud/menpo/

# # # Create Environment

# conda deactivate
# conda remove -n HIT --all
# conda create -n HIT -y python=3.7 numpy=1.20 numba
# conda activate HIT

conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r requirements.txt

cd lib/hilbertcurve_ops_lib/
python setup.py install
cd ../..

cd lib/pointcloud_ops_lib/
python setup.py install
cd ../..

# # # Visualization
# pip install open3d==0.15.2
# pip install opencv-python==4.5.1.48
