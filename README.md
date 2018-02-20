===============================================================================
# Detect to Track and Track to Detect

This repository contains the code for our ICCV 2017 paper:

    Christoph Feichtenhofer, Axel Pinz, Andrew Zisserman
    "Detect to Track and Track to Detect"
    in Proc. ICCV 2017

* This repository also contains results for a ResNeXt-101 and Inception-v4 backbone network that perform slightly better (**81.6%** and **82.1%** mAP on ImageNet VID val) than the ResNet-101 backbone (80.0% mAP) used in the conference version of the paper 

* This code builds on the original [Matlab version of R-FCN](https://github.com/daijifeng001/r-fcn)

* We are preparing a [Python version of D&T](https://github.com/feichtenhofer/py-Detect-Track) that will support end-to-end training and inference of the RPN, Detector & Tracker. 


If you find the code useful for your research, please cite our paper:

        @inproceedings{feichtenhofer2017detect,
          title={Detect to Track and Track to Detect},
          author={Feichtenhofer, Christoph and Pinz, Axel and Zisserman, Andrew},
          booktitle={International Conference on Computer Vision (ICCV)},
          year={2017}
        }

# Requirements

The code was tested on Ubuntu 14.04, 16.04 and Windows 10 using NVIDIA Titan X or Z GPUs. 

If you have questions regarding the implementation please contact:

    Christoph Feichtenhofer <feichtenhofer AT tugraz.at>

================================================================================

## Setup

1. Download the code ```git clone --recursive https://github.com/feichtenhofer/detect-track```
  *  This will also download a [modified version](https://github.com/feichtenhofer/caffe-rfcn)  of the [Caffe](http://caffe.berkeleyvision.org/) deep learning framework. In case of any issues,   please follow the [installation](https://github.com/feichtenhofer/caffe-rfcn) instructions in the corresponding README as well as on the Caffe [website](http://caffe.berkeleyvision.org/installation.html).

2. Compile the code by running ```rfcn_build.m```.

3. Edit the file get_root_path.m to adjust the models and data paths.
    * Download the ImageNet VID dataset from http://image-net.org/download-images
    * Download pretrained model files and the RPN proposals, linked below and unpack them into your models/data directory.
    * In case the models are not present, the function `check_dl_model` will attempt to download the model to the respective directories
    * In case the RPN files are not present, the function `download_proposals` will attempt to download & extract the proposal files to the respective directories
## Training
- You can train your own models on ImageNet VID as follows
    - `script_Detect_ILSVRC_vid_ResNet_OHEM_rpn();` to train the image-based **D**etection network.
    - `script_DetectTrack_ILSVRC_vid_ResNet_OHEM_rpn();` to train the video-based **D**etection & **T**acking network.

## Testing
- The scripts above have subroutines that test the learned models after training. You can also test our trained, final models available for download below. We provide three testing functions that work with a different numbers of frames at a time (i.e. processed by one GPU during the forward pass)
    1. `rfcn_test();` to test the image-based **D**etection network.
    1. `rfcn_test_vid();` to test the video-based **D**etection & **T**acking network with 2 frames at a time.
    1. `rfcn_test_vid_multiframe();` to test the video-based **D**etection & **T**acking network with 3 frames at a time.
- Moreover, we provide multiple testing network definitions that can be used for interesting experiments, for examüple
    - `test_track.prototxt` is the most simple form of **D&T** testing
    - `test_track_reg.prototxt` is a **D&T** version that additionally regresses the tracking boxes before performing the ROI tracking. Therefore, this procedure produces tracks that tightly encompass the underlying objects, whereas the above function tracks the proposal region (and therefore also the background area). 
    - `test_track_regcls.prototxt` is a **D&T** version that additionally classifies the tracked region and computes the detection confidence as the mean of the detection score from the current frame, as well as the detection score of the tracked region in the next frame. Therefore, this method produces better results, especially if the temporal distance between the frames becomes larger and more complementary information can be integrated from the tracked region


## Results on ImageNet VID 
* The networks are trained as decribed in the paper; i.e. on an intersection of the [ImageNet](http://image-net.org/) object detection from video (VID) dataset which contains 30 classes in 3862 training videos and and the [ImageNet](http://image-net.org/) object detection (DET) dataset (only using the data from the 30 VID classes). Validation results on the  555 videos of ImageNet VID validation are shown below.


|<sub> Method </sub>  |<sub>  test structure </sub> |  <sub> ResNet-50  </sub>     |  <sub> ResNet-101 | <sub> ResNeXt-101 </sub> |  <sub> Inception-v4 </sub> |
|:------------------|:-------------------|:--------------:|:--------------:|:--------------:| :-----------------:|
| <sub> **D**etect</sub> | <sub>test.prototxt</sub> |  72.1 |   74.1 | 75.9 | 77.9 |
| <sub> **D**etect & **T**rack </sub> |  <sub>test_track.prototxt</sub> |   76.5 | 79.8 |   81.4 |  82.0 |
| <sub> **D**etect & **T**rack </sub> |  <sub>test_track_regcls.prototxt</sub> |   76.7 |   80.0 | 81.6 | 82.1|

* We show different testing network definitions in the rows and backbone networks in columns. The reported performance is mAP (in %), averaged over all videos and classes in the ImageNet VID validation subset.


## Trained models
- Download our backbone and final networks trained on ImageNet here:
    - ImageNet CLS models: [ResNet-50](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNet-50-model.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNDZJm2Ug3w9igI1A) / [ResNet-101](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNet-101-model.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNG0ZfNh2S3Q0o2-w) /  [ResNeXt-101](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/resnext101-32x4d-merge.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNFgSF2xtYqIk9-mQ)
    - **D**etect models: [ResNet-50](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNet-50-D-ilsvrc-vid.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNEP_BIC2icrSwOpQ) / [ResNet-101](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNet101-D-ilsvrc-vid.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNI38M1s05H7lGUog) /  [ResNeXt-101](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNeXt101-D-ilsvrc-vid.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNHwvK4FPBGkTuvcA)
    - **D**etect & **T**rack models:  [ResNet-50](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNet-50-DT-ilsvrc-vid.caffemodel) [[OneDrive]]() / [ResNet-101](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNet-101-DT-ilsvrc-vid.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNJ7ELTXp26rAPmNw) /  [ResNeXt-101](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/ResNeXt101-DT-ilsvrc-vid.caffemodel) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNKs6NmlNdoLjVe8w)

## Data
Our models were trained using region proposals extracted using a Region Proposal Network that is trained on the same data as D&T. We use the RPN from [craftGBD](https://github.com/craftGBD/craftGBD/tree/master/proposal_gen) and provide the extracted proposals for training and testing on ImageNet VID and the DET subsets below.

Pre-computed object proposals for
- ImageNet DET: [[FTP server]](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/data/proposals/RPN_proposals_DET.zip) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0sNAfj3b7_fl9xIsJA) 
- ImageNet VID_train: [[FTP server]](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/data/proposals/RPN_proposals_VID_train.zip) [[OneDrive part1]](https://1drv.ms/u/s!AnKHschO7aEz0sNLGMF-eZJdOKbF2g)  [[OneDrive part2]](https://1drv.ms/u/s!AnKHschO7aEz0sNMEOPou7sXsVOhIw)
- ImageNet VID_val: [[FTP server]](http://ftp.tugraz.at/pub/feichtenhofer/detect-track/data/proposals/RPN_proposals_VID_val.zip) [[OneDrive]](https://1drv.ms/u/s!AnKHschO7aEz0Zo45zRTRaBnjvJTMg) 
