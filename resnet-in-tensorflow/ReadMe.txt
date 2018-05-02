Resnet in TensorFlow - python 2.7 
Look at the GitHub repo for code description
https://github.com/wenxinxu/resnet-in-tensorflow
--------------------

git clone https://github.com/wenxinxu/resnet-in-tensorflow.git
docker images

# As the Code is written in Python 2.7, make sure to pull the right docker image
docker pull nvcr.io/nvidia/tensorflow:18.02-py2

nvidia-docker run -it  -p 6006:6006 --name resnet3-13 -v /home/jghosh/resnet50/resnet-in-tensorflow:/home/jghosh/resnet50/resnet-in-tensorflow nvcr.io/nvidia/tensorflow:18.02-py2

cd home/jghosh/resnet50/resnet-in-tensorflow

pkg.sh
#!/bin/sh

pip install opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
pip install pandas

nohup python cifar10_train.py & 
tensorboard --logdir=/home/jghosh/resnet50/resnet-in-tensorflow/logs_test_110

