## Keras fine tune pretrained weights for Cifar10 and Distracted driver classification ##

This code uses Keras to finetune some of the well known CNN architectures like VGG16, VGG19, ResNet50 etc. The goal is to use ImageNet weights and fine tune to train on: 

* Cifar10 dataset
* Distracted driver dataset ([Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection))

The code is adapted from the [blog](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html) with [code](https://github.com/flyyufelix/cnn_finetune). I had to modify the code a bit to make sure it runs on our lab. To run this code on DGX-1 first login to t3lab and make sure you have the latest version of TensorFlow docker image. This code is written in Python 3.x, so make sure to pull the right version of TF.

```
docker images
``` 
Check on compute.nvidia.com to get the latest image. Use following command to pull the current version:
```
docker pull nvcr.io/nvidia/tensorflow:18.03-py3
```
Start the docker container by using following command:
```
nvidia-docker run -it  --name ktf -v /home/jghosh/keras-finetune-cifar10-distracteddriver:/home/jghosh/keras-finetune-cifar10-distracteddriver nvcr.io/nvidia/tensorflow:18.03-py3
```

If we already have the container running then attach to it
```
docker attach ktf
```

Navigate to the right directory:
```
cd /home/jghosh/keras-finetune-cifar10-distracteddriver
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
python train_cifar10.py --output figures/8_gpu_20K.png --gpus 8 --numepoch 5 --batchsize 16
```
You can specify which device you want  
```
CUDA_VISIBLE_DEVICES=0,3 python train.py --output figures/2_gpu_50K_64.png --gpus 2 --batchsize 64
```

To modify the code for Distracted Driver I used the concepts from the [blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

To get started we have to download the dataset from Kaggle using kagle api. Here is the [instruction](https://github.com/Kaggle/kaggle-api).
Once kaggle is installed and kaggle.json is placed in proper directory use following command to download the dataset and store in distracted_driver directory.
```
kaggle competitions download -c state-farm-distracted-driver-detection
```

To run the code:
```
python train_distracted_driver.py --output figures/8_gpu_dd.png --gpus 8 --numepoch 5 --batchsize 16
```
To run in the backgound I am using nohup option:
```
nohup python train_distracted_driver.py --output figures/8_gpu_dd.png --gpus 8 --numepoch 1 --batchsize 16 &
```

### What the code is doing ###
ADD more

fit_generator --> Fits the model on data generated batch-by-batch by a Python generator.

The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.

### Good links ###

* [In what order does 'flow_from_directory' function in keras takes the images](https://github.com/keras-team/keras/issues/3296)
* [How use the model.predict_generator to predict the probabilities of multi-labels](https://github.com/keras-team/keras/issues/9724)
* [A concrete example for using data generator for large datasets such as ImageNet](https://github.com/keras-team/keras/issues/1627)


### Contributors ###
* Jayeeta Ghosh
