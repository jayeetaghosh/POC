# Adapted from:
# https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/

This code uses Keras to build MiniGoogLeNet deep learning architecture for Cifar10 dataset - multi gpu way

# Python - 3.5.2 (python -V)
# keras - 2.1.5 (python -c 'import keras; print(keras.__version__)')
# tensorflow - 1.4.0 (python -c 'import tensorflow; print(tensorflow.__version__)')

# Make sure to get latest tensorflow docker container from compute.nvidia.com
# Pulled on 4/18/18
docker pull nvcr.io/nvidia/tensorflow:18.03-py3

# Start docker container
nvidia-docker run -it  --name ktf -v /home/jghosh/keras-minigooglenet-cifar10:/home/jghosh/keras-minigooglenet-cifar10 nvcr.io/nvidia/tensorflow:18.03-py3

# If we already have the container running then attach to it
docker attach ktf

cd /home/jghosh/keras-minigooglenet-cifar10
./pkg.sh
#!/bin/sh

pip install --upgrade pip
pip install keras
pip install opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
pip install h5py
pip install pillow
pip install sklearn
pip install matplotlib
apt-get install python3-tk
pip install vprof # This is a profiling library

# python train.py --output figures/1_gpu_18_4_23.png --batchsize 128
# python train.py --output figures/2_gpu_18_4_23.png --gpus 2 --batchsize 128
# python train.py --output figures/4_gpu_18_4_23.png --gpus 4 --batchsize 128
# python train.py --output figures/8_gpu_18_4_23.png --gpus 8 --batchsize 128

# More on Multi-GPU and distributed training using Keras
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#multi-gpu-and-distributed-training 
