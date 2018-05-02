# set the matplotlib backend so figures can be saved in the background
# (uncomment the lines below if you are using a headless server)
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
from vgg16 import vgg16_model
from load_cifar10 import load_cifar10_data

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
ap.add_argument("-b", "--batchsize", type=int, default=16,
	help="batchsize per gpu")
ap.add_argument("-e", "--numepoch", type=int, default=20,
	help="number of epochs")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]
batch_size = args["batchsize"]
nb_epoch = args["numepoch"]

# # Create augmented images from one image
# datagen = ImageDataGenerator(
#         # rotation_range=40,
#         width_shift_range=0.2
#         # height_shift_range=0.2,
#         # shear_range=0.2,
#         # zoom_range=0.2,
#         # horizontal_flip=True,
#         # fill_mode='nearest'
#         )
#
# img = load_img('distracted_driver/train/c1/img_115.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='distracted_driver/preview', save_prefix='img_6', save_format='jpeg'):
#     i += 1
#     if i > 5:
#         break  # otherwise the generator would loop indefinitely
# # End - Create augmented images from one image


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Delete the contents of preview directory
# import shutil
# shutil.rmtree('distracted_driver/previewtrain/')

import os, shutil
folder = 'distracted_driver/previewtrain'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

print("Start train generator")
train_generator = train_datagen.flow_from_directory(
        'distracted_driver/train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical',  # since we use categorical_crossentropy loss, we need categorical labels
        save_to_dir='distracted_driver/previewtrain',
        save_prefix='aug',
        save_format='jpeg',
        subset='training')

# this is a similar generator, for validation data
print("Start validation generator")
validation_generator = train_datagen.flow_from_directory(
        'distracted_driver/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

# This is test generator - for prediction
test_generator = test_datagen.flow_from_directory(
        'distracted_driver/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None, # only data, no labels
        shuffle=False) # keep data in same order as labels

img_rows, img_cols = 224, 224  # Resolution of inputs
channel = 3
num_classes = 10
# nb_epoch = 5#100
# train_size = 50000
# valid_size = 10000
INIT_LR = 1e-3
model = vgg16_model(img_rows, img_cols, channel, num_classes)
# initialize the optimizer and model
print("[INFO] compiling model...")
print("Start timing...")
import timeit
tic=timeit.default_timer()
# Learning rate is changed to 0.001
sgd = SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

H = model.fit_generator(
        train_generator,
        steps_per_epoch=17943 // (G*batch_size),#2000
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=4481 // (G*batch_size))#800
model.save_weights('exp_models/first_try.h5')  # always save your weights after training or during training

print("End timing...")
toc=timeit.default_timer()
print("Time elapsed:", toc - tic) #elapsed time in seconds

print("Start predictions on test images...")
# predictions
pred_prob = model.predict_generator(test_generator)
# Get predictions
import pandas as pd
df = pd.DataFrame(pred_prob)
df['class']=df.idxmax(axis=1)
# Get file names
# print (test_generator.filenames[0:10])
dffilename = pd.DataFrame(test_generator.filenames)
result = pd.concat([dffilename, df], axis=1)

result.to_csv("exp_models/prediction.csv")

print("Finish predictions on test images...")

# grab the history object dictionary
H = H.history

# plot the training loss and accuracy
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H["loss"], label="train_loss")
plt.plot(N, H["val_loss"], label="test_loss")
plt.plot(N, H["acc"], label="train_acc")
plt.plot(N, H["val_acc"], label="test_acc")
plt.title("VGG16 on Distracted Driver with finetune")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# save the figure
plt.savefig(args["output"])
plt.close()
print("Done!")

# The 10 classes to predict are:
#
# c0: safe driving
# c1: texting - right
# c2: talking on the phone - right
# c3: texting - left
# c4: talking on the phone - left
# c5: operating the radio
# c6: drinking
# c7: reaching behind
# c8: hair and makeup
# c9: talking to passenger