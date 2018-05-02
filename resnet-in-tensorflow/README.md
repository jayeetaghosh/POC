## TensorFlow - ResNet on CIFAR-10 images (Image classification) ##
In this POC I am trying to replicate ResNet (Residual Network) to classify images. 
I cloned the code from [github](https://github.com/wenxinxu/resnet-in-tensorflow). 
```
git clone https://github.com/wenxinxu/resnet-in-tensorflow.git
```
Here is the Box [link](https://trace3.app.box.com/folder/38046596390) to a presentation that I am working on to explain Convolutional Neural Network (CNN), differnt types of CNN architectures including Residual Network (ResNet), and differnt wellknown public datasets.


To run this code on DGX-1 first login to t3lab and make sure you have the latest version of TensorFlow docker image. This code is written in Python 2.7, so make sure to pull the right version of TF.
```
docker images
``` 
Check on compute.nvidia.com to get the latest image. Use following command to pull the current version:
```
docker pull nvcr.io/nvidia/tensorflow:18.02-py2
```
Start the docker container by using following command:
```
nvidia-docker run -it  -p 6006:6006 --name resnet3-13 -v /home/jghosh/resnet50/resnet-in-tensorflow:/home/jghosh/resnet50/resnet-in-tensorflow nvcr.io/nvidia/tensorflow:18.02-py2
```
I am using port mapping 6006 to make sure I can check TensorBoard during the training process. Next navigate to the right directory where you cloned the code.
```
home/jghosh/resnet50/resnet-in-tensorflow
```

We need to make sure we have a few packages before we try to run the python code. I put them in pkg.sh file, change the permission (chmod 777 pkg.sh) and run it by ./pkg.sh
```
#!/bin/sh

pip install opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
pip install pandas
```
After this I can just call the cifar10_train.py and it will automatically download the data first time and start the training process. I use nohup and start TensorBoard as below. Wait a while before starting TensorBoard, check if "logs_test_110" folder has been created. It will take about two hours to finish the training process.
To Do: I still need to make sure I can use tmux to keep the session running within docker image.
```
nohup python cifar10_train.py & 
tensorboard --logdir=/home/jghosh/resnet50/resnet-in-tensorflow/logs_test_110
```
Now point to the browser at [http://10.75.19.87:6006](to http://10.75.19.87:6006) to check the training process.

To look at the process at GPU level use following command at another terminal:
```
nvidia-smi
```
or to constantly monitor use -l option.
```
nvidia-smi -l 1
``` 
See nvidia-smi -h to get help or see the the [online documentation](http://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf)

### What the code is doing ###
cifar10_train.py is the main file that contains the class “Train” which contains functions like placeholders, build train validation graph, train, test, and bunch of helper functions like loss, top_k_error, generate validation batch, generate augmented train batch, train operations, validation operations, full validation. This essentially calls the helper functions defined in resnet.py and cifar10_input.py. 

resnet.py is the main resnet network structure file that defines the activation summary, conv layers, batch normalization layer, and output layers. 

cifar10_input.py is the main file to download the data, read one batch, read all data, data augmentation functions like horizonal_flip, whitening_image, random_crop_and_flip. 

hyperparameter.py on the other hand as the name suggests defines all the hyperparameters for training and modifying training network. 
Please see [github](https://github.com/wenxinxu/resnet-in-tensorflow) for the description of the code structure from the authors.

* Get data: The process starts with maybe_download_and_extract() within cifar10_train.py that downloads cifar data [from] (http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and saves within cifar10_data directory. The dataset is divided into five training batches and one test batch, each with 10000 images in python pickle format. Once loaded properly, each of the batch files contains a dictionary with the following elements. Next it instantiates train object and starts the training process.

	*_data_: Each image is 32x32 color image (RGB channels). So, each image is represented as 32x32x3 = 3072 pixel stored in an array. The data in each batch is of size 10000x3072 as each batch contains 10K images

	*_labels_: A list of 10000 numbers between 0-9 that corresponds to the 10 classes for these images. 
	Please see Cifar-10 documentation for more [information](http://www.cs.toronto.edu/~kriz/cifar.html) about Cifar-10 dataset.


* Train: This process starts with preparation of the training data and validation data that includes generation of augmented training batch. Augmentation process includes random crop, horizontal flip, and whitening them. After that it runs for number of training steps that was defined in hyperparameter file with checkpoint files and error files saved intermittently.  



### Contributors ###
* Jayeeta Ghosh
