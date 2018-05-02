## Keras MiniGoogLeNet for Cifar10 image classification ##
In this POC I am trying to use a simple version of GoogLeNet network to train Cifar10 data for image classification. Here is the [link](https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/) to the blog. 

Please look at the blog for the code description.

 
I had to modify the code a bit to make sure it runs on our lab. To run this code on DGX-1 first login to t3lab and make sure you have the latest version of TensorFlow docker image. This code is written in Python 3.x, so make sure to pull the right version of TF.
```
docker images
``` 
Check on compute.nvidia.com to get the latest image. Use following command to pull the current version:
```
docker pull nvcr.io/nvidia/tensorflow:18.03-py3
```
Start the docker container by using following command:
```
nvidia-docker run -it  --name ktf -v /home/jghosh/keras-minigooglenet-cifar10:/home/jghosh/keras-minigooglenet-cifar10 nvcr.io/nvidia/tensorflow:18.03-py3
```
If we already have the container running then attach to it
```
docker attach ktf
```
Navigate to the right directory:
```
cd /home/jghosh/keras-imageclass-inference
```


We need to make sure we have a few packages before we try to run the python code. I put them in pkg.sh file, change the permission (chmod 777 pkg.sh) and run it by ./pkg.sh
```
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
```

```
./pkg.sh
```

After this I can just call the train.py with location of the output, number of gpus to use, batchsize etc as below:

```
python train.py --output figures/1_gpu_18_4_23.png --batchsize 128
python train.py --output figures/2_gpu_18_4_23.png --gpus 2 --batchsize 128
python train.py --output figures/4_gpu_18_4_23.png --gpus 4 --batchsize 128
python train.py --output figures/8_gpu_18_4_23.png --gpus 8 --batchsize 128
```


### What the code is doing ###
The blogger Adrian Rosebrock has done a very good job of explaining the code and is very intuitive.

### References ###

* [Keras Blog](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#multi-gpu-and-distributed-training)
* [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)

### Contributors ###
* Jayeeta Ghosh
