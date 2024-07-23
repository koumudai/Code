conda deactivate
conda env list
conda remove -n FSTR --all
conda create -n FSTR -y python=3.8
conda activate FSTR
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html


pip install -r requirements.txt

pip install openmim

# the problem of missing libcudnn.so.8 
# https://blog.csdn.net/qq_44703886/article/details/112393149

# pip install onnx
# pip install onnxruntime


