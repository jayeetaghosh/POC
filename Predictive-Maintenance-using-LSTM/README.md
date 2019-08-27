# Recurrent Neural Networks for Predictive Maintenance
The code is adapted from [github](https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM)
This is originally coming from Azure Predictive Maintenance Template:
* [Link1](https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2)
* [Link2](https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb)


## Colab
You can try the code directly on [Colab](https://colab.research.google.com/drive/1tjIOud2Cc6smmvZsbl-QDBA6TLA2iEtd).
Save a copy in your drive and enjoy It!


## To run on DGX-1
Make sure to get latest tensorflow docker container from compute.nvidia.com
```
docker pull nvcr.io/nvidia/tensorflow:18.03-py3
```

Start docker container
```
nvidia-docker run -it  --name rtf -v /home/jghosh/RNN:/home/jghosh/RNN nvcr.io/nvidia/tensorflow:18.03-py3
```
If we already have the container running then attach to it
```
docker attach rtf 
```
```
cd /home/jghosh/RNN
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
```
```
nohup python binary_classification.py &
nohup python regression.py &

python binary_classification_multigpu.py --gpus 1 --batchsize 200 --numepoch 3
```
## Problem Description
In this example I build an LSTM network in order to predict remaining useful life (or time to failure) of aircraft engines <a href="https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan">[3]</a> based on scenario described at <a href="https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb">[1]</a> and <a href="https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2">[2]</a>.
The network uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future so that maintenance can be planned in advance.
The question to ask is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"
We re-formulate this question into two closely relevant questions and answer them using two different types of machine learning models:

	* Regression models: How many more cycles an in-service engine will last before it fails?
	* Binary classification: Is this engine going to fail within w1 cycles?

## Data
In the **Dataset** directory there are the training, test and ground truth datasets.
The training data consists of **multiple multivariate time series** with "cycle" as the time unit, together with 21 sensor readings for each cycle.
Each time series can be assumed as being generated from a different engine of the same type.
The testing data has the same data schema as the training data.
The only difference is that the data does not indicate when the failure occurs.
Finally, the ground truth data provides the number of remaining working cycles for the engines in the testing data.
The following picture shows a sample of the data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/datasetSample.png"/>
</p>
You can find more details about the data at <a href="https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb">[1]</a> and <a href="https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2">[2]</a>.
 
## Results of Regression model

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
|12|0.7965|

The following pictures shows the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_regression_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_mae.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_regression_verify.png"/>
</p>
         
## Results of Binary classification 

|Accuracy|Precision|Recall|F-Score|
|----|----|----|----|
|0.97|0.92|1.0|0.96|

The following pictures shows trend of loss Function, Accuracy and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_accuracy.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_verify.png?raw=true"/>
</p>
           
## References

- [1] Deep Learning for Predictive Maintenance https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- [2] Predictive Maintenance: Step 2A of 3, train and evaluate regression models https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- [3] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research Center, Moffett Field, CA 
- [4] Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/
         