conda deactivate
conda env list
conda remove -n FSTR --all
conda create -n FSTR -y python=3.8
conda activate FSTR
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

pip install -r requirements.txt

pip install openmim
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmdet==2.24.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc5

pip install flash-attn==0.2.2


## install Spconv-plus
# https://github.com/dvlab-research/spconv-plus
# https://blog.csdn.net/weixin_44013732/article/details/133207622

export SPCONV_DISABLE_JIT="1"

pip install pccm==0.3.4 ccimport==0.3.7 cumm==0.2.8 wheel  






# the problem of missing libcudnn.so.8 
# https://blog.csdn.net/qq_44703886/article/details/112393149

# pip install onnx
# pip install onnxruntime


