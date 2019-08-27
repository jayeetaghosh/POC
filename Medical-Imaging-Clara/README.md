## Clara - Medical Imaging SDK from Nvidia ##
In this POC I am trying to run the brain tumor mri segmentation training and finetuning tutorial. 

As a first step start from the original link to Clara official website from [Nvidia](https://developer.nvidia.com/clara) to familiarize yourself. Clara Train SDK is 
only available through their Early Access program. Eric applied for early access and got the docker container pulled on our DGX. 

Check docker images: 
```
docker images ==> nvcr.io/nvtltea/med/tlt-annotate
```

Next step should be to familiarize with the NGC models avaiable regarding Clara SDK. There are two models for brain mri segmentation:

* Clara AI Segmentation_MRI_Brain_br16_full: A pre-trained model for volumetric (3D) segmentation of brain tumors from multimodal MRIs based on [BraTS 2018 data](https://www.med.upenn.edu/sbia/brats2018/data.html). The model is trained to segment 3 nested subregions of primary (gliomas) brain tumors: the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT) based on 4 input MRI scans ( T1c, T1, T2, FLAIR). The ET is described by areas that show hyper intensity in T1c when compared to T1, but also when compared to "healthy" white matter in T1c. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR. For more detailed description of tumor regions, please see the Brats2018 data page. This model was trained using a similar approach described in 3D MRI brain tumor segmentation using autoencoder regularization, which was a winning method in Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018. The model was trained using BraTS 2018 training data (285 cases).
* Clara AI Segmentation_MRI_Brain_br16_t1c2tc: The model is similar to "segmentation_brain_br16_full" model, except the input is only 1 channel MRI (T1c) and the output is only 1 channel for brain tumor subregion (TC - Tumor Core) - a pre-trained model for volumetric (3D) brain tumor segmentation (only TC from T1c images). The model was trained using [BraTS 2018 training data](https://www.med.upenn.edu/sbia/ brats2018/data.html) (285 cases)

To follow along the tutorial please continue [brain_tumor_mri_segmentation](https://bitbucket.org/Trace3/poc_dgx-1/src/master/Medical-Imaging-Clara/brain_tumor_mri_segmentation/)

### Contributors ###
* Jayeeta Ghosh
