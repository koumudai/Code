conda deactivate
conda env list
conda remove -n OPCD --all
conda create -n OPCD -y python=3.8 numpy=1.22 numba
conda activate OPCD

conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

# the problem of missing libcudnn.so.8 
# https://blog.csdn.net/qq_44703886/article/details/112393149

# pip install onnx
# pip install onnxruntime


