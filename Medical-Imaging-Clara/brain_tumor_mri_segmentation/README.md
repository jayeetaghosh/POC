## 3D MRI brain tumor segmentation using Clara ##
Here we are trying to utilize Clara Transfer Learnign Toolkit (tlt) to build 3D MRI segmentation model and fine tune from pretrained model parameters. 

* Training from scratch (based on BraTS 2018 data)
* Fine-tuning a pre-trained model on a set of new images

### Data ###
Download the [Brats2018](https://www.med.upenn.edu/sbia/brats2018/registration.html) data, and place it in the brdemo/brats2018challenge folder. In order to download the data from BraTS we need to create an a/c at [CBICA](ipp.cbica.upenn.edu), wait for approval, get download url in email and then use that to download the actual data.

Data are in NifTi (nii) file format which is very common and easier to use datatype for 3D images with isotropic 1x1x1 mm voxel spacing. Input is 4 channel 3D MRI: T1, T1c, T2, FLAIR. 

The annotations were combined into 3 nested subregions: whole tumor (WT), tumor core (TC) and enhancing tumor (ET). 

![Embed 3D image segmentation image](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Medical-Imaging-Clara/brain_tumor_mri_segmentation/Ref/annotations.jpg)

[3D image segmentation image](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Medical-Imaging-Clara/brain_tumor_mri_segmentation/Ref/annotations.jpg) shows a typical segmentation example with true and predicted labels overlaid over T1c MRI axial, sagittal and coronal slices. The whole tumor (WT) class includes all visible labels (a union of green, yellow and red labels), the tumor core (TC) class is a union of red and yellow, and the enhancing tumor core (ET) class is shown in yellow (a hyperactive tumor part). The predicted segmentation results match the ground truth well.



### MRI 101 some useful definitions ###

MR images can be acquired with several different techniques (pulse sequences) and acquisition parameters (called e.g. echo time, TE, repetition time TR etc.)
resulting in different image contrast. The 4 main modalities are:

* T1: T1-weighted MRI: image contrast is based predominantly on the T1 (longitudinal) relaxation time of tissue; tissue with short T1 relaxation time appears brighter (hyperintense)
* T2: T2-weighted MRI: image contrast is based predominantly on the T2 (transverse) relaxation time of tissue; tissue with long T2 relaxation time appears brighter (hyperintense)
* T1C: T1-weighted MRI after administration of contrast media: many tumors show signal enhancement after administration of contrast agent
* FLAIR: fluid-attenuated inversion-recovery MRI: bright signal of the CSF (cerebrospinal fluid) is suppressed which allows a better detection of small hyperintense lesions.

Segmentation of tumors will probably be most successful if several (or better all) of these images are combined. If you insist on taking only one dataset, then try the T1C images (but this depends on the tumor type).

Brats2018 training data contains HGG and LGG. What are those?

The most common type of primary brain tumors are gliomas, which arise from brain glial cells. Gliomas account for 29%-35% of the central nervous system 
(CNS) tumors in adolescents and young adults, with approximately two-thirds being low-grade glioma (LGG) and the remaining being high-grade glioma (HGG).

* HGG: High Grade Gliomas
* LGG: Low Grade Gliomas

Dataset is stored at /nfs/brdemo/brats2018challenge

### Tutorial Part 1 ###
To start the tutoroal, please open Clara-Train-SDK-Transfer-Learning-Getting-Started-GuideEA.pdf from [ref](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Medical-Imaging-Clara/clara_documentation/?at=master) directory and follow along. They provided tutorial_brats.ipynb with all the necessary commands.

Here is the specific docker command I used:
```
docker run --runtime=nvidia -it --rm --name j1 -p 8880:8880 -v /home/jghosh/tlt-experiments:/workspace/tlt-experiments -v /home/jghosh/tlt-experiments/brdemo:/mnt/brdemo -w /opt/nvidia/medical/segmentation/examples/brats nvcr.io/nvtltea/med/tlt-annotate:v0.1-py3 jupyter notebook --ip 0.0.0.0 --port=8880 --allow-root --no-browser
```

now point the browser to

dgx1.di.labs.trace3.com:8880/?token=0af3a3e4903f150917a209a1e7eb304d540419f93b2908d5 ==> make sure to copy the right token from your session

Changes required to run on multi-gpus:

* make "multi_gpu": true in config_brats_8
* wrap the tlt-train command inside of the mpirun, as shown below:
```
%env NUM_GPU=8

!mpirun -np $NUM_GPU -H localhost:$NUM_GPU -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib --allow-run-as-root tlt-train segmentation \
    -e $TRAIN_CONFIG \
    -d $DATA_LIST \
    -r $TRAIN_OUTPUT_DIR
``` 
==> 100% utilization of all 8 gpus seen with following command:
```
nvidia-smi -l 1
```

Once training is finished follow along the tutorial to evaluate the model and run inference on new images. You can open the output_path folder and inspect the final segmentation masks in your
3D viewer (e.g. ITK-SNAP). I installed ITL-SNAP on my desktop to visualize the segmentation results. The outpul from training process produces mask for three annotations separately, needed to combine them in order to visualize.

Downloaded [convert3D](http://www.itksnap.org/pmwiki/pmwiki.php?n=Convert3D.Convert3D ), another tool to merge 3 different segments. Once installed on my windows m/c I used following command to combine 3 different nii files into a single nii file with 3 different masks for 3 different annotations.
```
C3d Brats18_2013_10_1_t1ce_ET.nii Brats18_2013_10_1_t1ce_WT.nii -replace 1 2 -add -o ET_WT.nii
C3d Brats18_2013_10_1_t1ce_TC.nii ET_WT.nii -replace 2 2 3 3 -add -o ET_WT_TC.nii
```
Once I had the merged nii file, I was able to overlay on top of the original input image in 4 different modalities as shown in [images](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Medical-Imaging-Clara/brain_tumor_mri_segmentation/images/) with 3 views, Axial, Sagittal, and Coronal.



### Tutorial Part 2 ###
For the second part of the tutorial to fine tune an existing model we need to download the pretrained network files using tlt-pull command. It was straight forward to follow along the tutorial.

Pretrained model is stored at /nfs/brdemo/pretrained

### Things to consider to adapt this for a real usecase ###

* How to handle other 3D data types. Nvidia has a tool tlt-dataconvert to convert all dicom volumes in your/data/directory to NIfTI format and optionally re-samples them to the provided resolution. If the images to be converted are segmentation labels, an option -l needs to be added, and the resampler will use nearest neighbor interpolator (otherwise linear interpolator is used). 
* Need to consider other data converters incase tlt-dataconvert does not work for any specific image format
* The algorithm is a ResNet based encoder-decoder, though they did not provide the actual model script files but we might be able to get them through our Nvidia contacts. Once we have access to the scripts we should be able to utilize other algorithms like V-net, DenseNet or other ensemble approaches.
* The data pipeline as described by the Nvidia [article](https://arxiv.org/abs/1810.11654) considers augmentation by random intensity shift, scale on input image channels, and random axis mirror flip. How to add more augmentation recipes in case we have to deal with small dataset size.  
* The tutorial utilizes TensorRT optimization using `tlt-export` command to convert the checkpoint to TensorRT optimized model and run the model on some new data without the ground truth, and save the output segmentation masks. This will save the segmentation masks (as NIfTI files) for all input files listed in validation provided in config_brats18_datalist1.json. Need to utilize this using Kubeflow/Kubernetes for operationalization. 

### Contributors ###
* Jayeeta Ghosh
