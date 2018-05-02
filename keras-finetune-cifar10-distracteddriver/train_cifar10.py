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

img_rows, img_cols = 224, 224  # Resolution of inputs
channel = 3
num_classes = 10
# nb_epoch = 100
train_size = 50000
valid_size = 10000
INIT_LR = 1e-3

# Load Cifar10 data. Please implement your own load_data() module for your own dataset
print("[INFO] loading CIFAR-10 data...")
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols, train_size, valid_size, num_classes)

print("Num of GPU requested", G)
if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
        # initialize the model
        model = vgg16_model(img_rows, img_cols, channel, num_classes)

    # make the model parallel
    model = multi_gpu_model(model, gpus=G)

# initialize the optimizer and model
print("[INFO] compiling model...")
print("Start timing...")
import timeit
tic=timeit.default_timer()
# Learning rate is changed to 0.001
sgd = SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# opt = SGD(lr=INIT_LR, momentum=0.9)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Start Fine-tuning - train the network
print("[INFO] training network...")
H = model.fit(X_train, Y_train,
          batch_size=batch_size * G,
          epochs=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )

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
plt.title("VGG16 on CIFAR-10 with finetune")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# save the figure
plt.savefig(args["output"])
plt.close()

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

# Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)
print("Log loss score for validation", score)
print("End timing...")
toc=timeit.default_timer()
print("Time elapsed:", toc - tic) #elapsed time in seconds
print("Done!")