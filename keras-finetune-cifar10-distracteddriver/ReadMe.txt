https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
https://github.com/flyyufelix/cnn_finetune
https://www.kaggle.com/c/state-farm-distracted-driver-detection

1. This code uses Keras to finetune some of the well known CNN architectures like VGG16, VGG19, ResNet50 etc. The goal is to use ImageNet weights and fine tune to train on Cifar10 dataset.

2. Downloaded distracted driver dataset from Kaggle using kaggle api. Here is the instruction: https://github.com/Kaggle/kaggle-api
Once kaggle is installed and kaggle.json is placed in proper directory use following command to download 
kaggle competitions download -c state-farm-distracted-driver-detection

# Python - 3.5.2 (python -V)
# keras - 2.1.5 (python -c 'import keras; print(keras.__version__)')
# tensorflow - 1.4.0 (python -c 'import tensorflow; print(tensorflow.__version__)')

# Make sure to get latest tensorflow docker container from compute.nvidia.com
# Pulled on 4/18/18
docker pull nvcr.io/nvidia/tensorflow:18.03-py3

# Start docker container
nvidia-docker run -it  --name ktf -v /home/jghosh/keras-finetune-cifar10-distracteddriver:/home/jghosh/keras-finetune-cifar10-distracteddriver nvcr.io/nvidia/tensorflow:18.03-py3

# If we already have the container running then attach to it
docker attach ktf

cd /home/jghosh/keras-finetune-cifar10-distracteddriver
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

python train_cifar10.py --output figures/8_gpu_20K.png --gpus 8 --numepoch 5 --batchsize 16
# You can specify which device you want  
CUDA_VISIBLE_DEVICES=0,3 python train.py --output figures/2_gpu_50K_64.png --gpus 2 --batchsize 64


https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
fit_generator -->
Fits the model on data generated batch-by-batch by a Python generator.

The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.

In what order does 'flow_from_directory' function in keras takes the images? https://github.com/keras-team/keras/issues/3296 
How use the model.predict_generator to predict the probabilities of multi-labels
https://github.com/keras-team/keras/issues/9724
A concrete example for using data generator for large datasets such as ImageNet
https://github.com/keras-team/keras/issues/1627

python train_distracted_driver.py --output figures/8_gpu_dd.png --gpus 8 --numepoch 5 --batchsize 16
nohup python train_distracted_driver.py --output figures/8_gpu_dd.png --gpus 8 --numepoch 1 --batchsize 16 &

kaggle competitions submit -c state-farm-distracted-driver-detection -f 1_submission.csv -m "try1"