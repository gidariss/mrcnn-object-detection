## *Object detection via a multi-region & semantic segmentation-aware CNN model*

###################################################################

Introduction:

This code implements the following ICCV2015 accepted paper:  
Title: "Object detection via a multi-region & semantic segmentation-aware CNN model"  
Authors: Spyros Gidaris, Nikos Komodakis  
Institution: Universite Paris Est, Ecole des Ponts ParisTech  
Technical report: http://arxiv.org/abs/1505.01749  
code: https://github.com/gidariss/mrcnn-object-detection  

Abstract:  
"We propose an object detection system that relies on a multi-region deep convolutional neural network (CNN) that also encodes semantic segmentation-aware features. The resulting CNN-based representation aims at capturing a diverse set of discriminative appearance factors and exhibits localization sensitivity that is essential for accurate object localization. We exploit the above properties of our recognition module by integrating it on an iterative localization mechanism that alternates between scoring a box proposal and refining its location with a deep CNN regression model. Thanks to the efficient use of our modules, we detect objects with very high localization accuracy. On the detection challenges of PASCAL VOC2007 and PASCAL VOC2012 we achieve mAP of 78.2% and 73.9% correspondingly, surpassing any other published work by a significant margin."   

If you find this code useful in your research, please consider citing:  

@article{gidaris2015object,
  title={Object detection via a multi-region \& semantic segmentation-aware CNN model},
  author={Gidaris, Spyros and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1505.01749},
  year={2015}
}

###################################################################

License:  
This code is released under the MIT License (refer to the LICENSE file for details).  

###################################################################

Requirements:  

-- Software: 

1) MATLAB (tested with R2014b)

2) Caffe: https://github.com/BVLC/caffe

3) LIBLINEAR (only for training)  

4) Edge Boxes code: https://github.com/pdollar/edges

5) Piotr's image processing MATLAB toolbox: http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html

6) Selective search code: http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip

-- Data: 

1) PASCAL VOC2007  
2) PASCAL VOC2012    

###################################################################

Installation:

1. Install CAFFE https://github.com/BVLC/caffe
2. Place a soft link to caffe directory on {path-to-mrcnn-object-detection}/external/caffe  
3. Place the VGG16 model pre-trained on ImageNet for the Image Classification task on:
	{path-to-mrcnn-object-detection}/data/vgg_pretrained_models

4.  open matlab from the directory {path-to-mrcnn-object-detection}/
5.  Run mrcnn_build.m  
6.  Run startup.m  
7. add on matlab the paths to installation directories of the Edge Boxes, Piotr's image processing MATLAB toolbox, and Selective Search code.

If you will use the pre-trained object detection models then

8. Place the pre-trained models on the following directories:  
	I)  {path-to-mrcnn-object-detection}/models-exps/MRCNN_VOC2007_2012  : multi-region recognition model.  
	II) {path-to-mrcnn-object-detection}/models-exps/vgg_bbox_regression_R0013_voc2012_2007/ : bounding box regression model.

If you are going to do experiments on PASCAL VOC2007 or VOC2012 datasets then:

9. Place the VOCdevkit of VOC2007 on {path-to-mrcnn-object-detection}/datasets/VOC2007/VOCdevkit and its data on {path-to-mrcnn-object-detection}/datasets/VOC2007/VOCdevkit/VOC2007 
10. Place the VOCdevkit of VOC2012 on {path-to-mrcnn-object-detection}/datasets/VOC2012/VOCdevkit and its data on {path-to-mrcnn-object-detection}/datasets/VOC2012/VOCdevkit/VOC2012

###################################################################

Testing the pre-trained models on VOC2007 test set: 

1. Pre-cache the VGG16 conv5 features of the images in VOC2007 test set (for the scales 480, 576, 688, 874, and 1200) by running on matlab:  
script_extract_vgg16_conv_features('test', '2007', 'gpu_id', 1);   
gpu_id is a one-based index; if a non positive value is given then the CPU will be used instead. It should take around 2-3 hours for the VOC2007 test set.  

2. To test the multi-region CNN recognition model coupled with the iterative bounding box localization module on the VOC2007 test set run:  
script_test_object_detection_iter_loc('MRCNN_VOC2007_2012', 'vgg_bbox_regression_R0013_voc2012_2007', 'gpu_id', 1);   
By default, the above script uses the edge box proposals.  

###################################################################
