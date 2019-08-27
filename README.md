# Proof of Concept for DGX-1 #

This Repository contains some usecases for running Deep Learning experiments on DGX-1
### What is this repository for? ###

* [Resnet in TensorFlow using Cifar10 data](https://bitbucket.org/Trace3/poc_dgx-1/src/master/resnet-in-tensorflow/)
* [Keras pretrained models for image classification inference](https://bitbucket.org/Trace3/poc_dgx-1/src/master/keras-imageclass-inference/)
* [Keras MiniGoogLeNet for Cifar10 data](https://bitbucket.org/Trace3/poc_dgx-1/src/master/keras-minigooglenet-cifar10/)
* [Keras fine tune pretrained models for Cifar10 and Distracted Drivers image classification](https://bitbucket.org/Trace3/poc_dgx-1/src/master/keras-finetune-cifar10-distracteddriver/)
* [Predictive Maintenance using LSTM](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Predictive-Maintenance-using-LSTM/)
* PyTorch - Seq2Seq language translation
* [Medical-Imaging-Clara](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Medical-Imaging-Clara/)


### Some useful docker commands ###
Show running containers
```
docker ps 
```
Show all containers
```
docker ps -a
```
Show the latest created containers
```
docker ps -l
```
Remove individual stopped docker container. This will remove the container from docker ps -a list, loosing its "state" (the layered filesystems written on top of the image filesystem). It cannot remove a running container (unless called with -f, in which case it sends SIGKILL directly).
```
docker rm <name or id>
```
Remove all
```
docker rm $(docker ps -aq)
```
Stop individual docker container. preserves the container in the docker ps -a list (which gives the opportunity to commit it if you want to save its state in a new image).
It sends SIGTERM first, then, after a grace period, SIGKILL.
```
docker stop <name or id>
```
Use docker attach to attach your terminals standard input, output, and error (or any combination of the three) to a running container using the containers ID or name. This allows you to view its ongoing output or to control it interactively, as though the commands were running directly in your terminal.
Note: The attach command will display the output of the ENTRYPOINT/CMD process. This can appear as if the attach command is hung when in fact the process may simply not be interacting with the terminal at that time.

You can attach to the same contained process multiple times simultaneously, from different sessions on the Docker host.
Ctrl+c will stop the job and bring into the container for further process
```
docker attach <name or id>
```
#### To come out of a docker container there are two options ####
* exit ==> this will stop the container but will not remove. This is similar to 'docker stop' command. We can start again by "docker start name_or_id" and consequently attach to it later by "docker attach name_or_id" 
* Ctrl+p+q ==> this will bring back to terminal with out stopping the container. This can be attached by "docker attach name_or_id".  To stop and remove this container either use "docker stop name_or_id" + "docker rm name_or_id" or "docker rm -f name_or_id"


## PyTorch  Seq2Seq language translation ##
[Tutorial link](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

This tutorial is about translating French to English using Sequence to Sequence Network and Attention. This is a very well written and easy to follow tutorial. Just running the code is not super hard but understanding what is happening and a lot of the DL details are hard to grasp. The tutorial has good links that can be followed. 

[Extra Video link: Stanford Lecture 10](https://www.youtube.com/watch?v=IxQtK2SjWWM) 

#### What this code is doing: ####
1.	Read text file that contains English to French pairs (eng-fra.txt)
2.	Split the text file into lines, split lines into pairs
3.	Normalize text  convert Unicode string to plain ASCII, lower case, trim, and remove non-Letter characters
4.	Keep only 10 words or less sentences
5.	Keep only those sentences that translate to the form I am or He is or you are etc
6.	Make word lists from sentences in pairs

#### To run the code: ####
* Step1: Login to DGX-1. [See instructions](https://bitbucket.org/Trace3/t3-di-docs/wiki/people/setup/LabSetup.md) 
* Step2: Get the code from trace3 [bitbucket](https://bitbucket.org/Trace3/poc_dgx-1/src). You could get it from the tutorial link but I made little changes to make sure it runs on our lab. 
* Step3: Get latest NVIDIA docker image for PyTorch if not already present
docker images --> to check if pytorch is available 
login to [compute.nvidia.com](https://compute.nvidia.com/login) and get the below command
docker pull nvcr.io/nvidia/pytorch:17.12
* Step4: Start docker image and run the python code
```
nvidia-docker run -it --name pytorchrun1 -v /home/jghosh/pytorchplay:/home/jghosh/pytorchplay nvcr.io/nvidia/pytorch:17.12
```
or 
```
docker run -it --name pytorchrun1 -v /home/jghosh/pytorchplay:/home/jghosh/pytorchplay nvcr.io/nvidia/pytorch:17.12
```
--Now within container--
```
source activate pytorch-py3.6
cd /home/jghosh/pytorchplay/
python seq2seq_translation_tutorial.py
source deactivate pytorch-py3.6
exit or Ctrl+p+q
```

### Contributors ###
* Jayeeta Ghosh
* Eric Hankins
