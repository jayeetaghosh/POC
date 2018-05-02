https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

# This is the code to use Keras pretrained models to classify images

# models: vgg16, vgg19, resnet, inception, xception
# Python - 3.5.2 (python -V)
# keras - 2.1.5 (python -c 'import keras; print(keras.__version__)')
# tensorflow - 1.4.0 (python -c 'import tensorflow; print(tensorflow.__version__)')


# Make sure to get latest tensorflow docker container from compute.nvidia.com
docker pull nvcr.io/nvidia/tensorflow:18.03-py3

# Start docker container
nvidia-docker run -it  --name ktf -v /home/jghosh/keras-imageclass-inference:/home/jghosh/keras-imageclass-inference nvcr.io/nvidia/tensorflow:18.03-py3

cd /home/jghosh/keras-imageclass-inference
./pkg.sh
python classify_image.py --image images/dog1.jpg --model vgg16
1. Ibizan_hound: 57.97%
2. basenji: 19.47%
3. whippet: 4.29%
4. borzoi: 3.43%
5. wire-haired_fox_terrier: 3.42%

python classify_image.py --image images/dog1.jpg --model vgg19
1. Ibizan_hound: 67.13%
2. English_foxhound: 16.45%
3. Walker_hound: 5.52%
4. toy_terrier: 2.62%
5. Saluki: 1.72%

python classify_image.py --image images/dog1.jpg --model resnet
1. Ibizan_hound: 48.69%
2. English_foxhound: 11.80%
3. borzoi: 9.92%
4. toy_terrier: 8.53%
5. Walker_hound: 6.35%

python classify_image.py --image images/dog1.jpg --model inception
1. toy_terrier: 54.65%
2. wire-haired_fox_terrier: 33.89%
3. English_foxhound: 0.92%
4. papillon: 0.51%
5. basenji: 0.39%

python classify_image.py --image images/dog1.jpg --model xception
1. toy_terrier: 50.26%
2. wire-haired_fox_terrier: 33.81%
3. English_foxhound: 2.29%
4. basenji: 1.68%
5. Walker_hound: 1.01%
