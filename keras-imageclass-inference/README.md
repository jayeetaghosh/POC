## Keras Inference for Image classification ##
In this POC I am trying to use pretrained models that are shipped with Keras to use as inference. Here is the [link](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/) to the blog. Please look at the blog for description of different CNN architectures as well as the code description.


Models we can try:

* vgg16
* vgg19
* resnet
* inception
* xception 

 
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
nvidia-docker run -it  --name ktf -v /home/jghosh/keras-imageclass-inference:/home/jghosh/keras-imageclass-inference nvcr.io/nvidia/tensorflow:18.03-py3
```
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
```

```
./pkg.sh
```

After this I can just call the classify_image.py with location of the image file and the model choice as below:

```
python classify_image.py --image images/dog1.jpg --model vgg16
python classify_image.py --image images/dog1.jpg --model vgg19
python classify_image.py --image images/dog1.jpg --model resnet
```


### What the code is doing ###
The blogger Adrian Rosebrock has done a very good job of explaining the code and is very intuitive.


### Contributors ###
* Jayeeta Ghosh
