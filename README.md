## *Object detection via a multi-region & semantic segmentation-aware CNN model*

*###########################################################################*

# Introduction:

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

*###########################################################################*

License:  
This code is released under the MIT License (refer to the LICENSE file for details).  

*###########################################################################*

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

*###########################################################################*

Installation:

1. Install CAFFE https://github.com/BVLC/caffe
2. Place a soft link of caffe directory on {path-to-mrcnn-object-detection}/external/caffe  
3. Place a soft link of the edge boxes installation directory on {path-to-mrcnn-object-detection}/external/edges
4. Place the VGG16 model pre-trained on ImageNet for the Image Classification task on:
	{path-to-mrcnn-object-detection}/data/vgg_pretrained_models

5.  open matlab from the directory {path-to-mrcnn-object-detection}/
6.  Edit the startup.m script by setting the installation directories paths of Edge Boxes, Piotr's image processing MATLAB toolbox, and Selective Search to the proper variables (see startup.m).
7.  Run startup.m  

If you are going to use the pre-trained object detection models then

9. Place the pre-trained models on the following directories:  
	I)   {path-to-mrcnn-object-detection}/models-exps/MRCNN_VOC2007_2012  : multi-region recognition model.  
    II)  {path-to-mrcnn-object-detection}/models-exps/MRCNN_SEMANTIC_FEATURES_VOC2007_2012  : multi-region with the semantic segmentation aware cnn featues recognition model.  
	III) {path-to-mrcnn-object-detection}/models-exps/vgg_bbox_regression_R0013_voc2012_2007/ : bounding box regression model.

For running experiments on PASCAL VOC2007 or VOC2012 datasets then:

10. Place the VOCdevkit of VOC2007 on {path-to-mrcnn-object-detection}/datasets/VOC2007/VOCdevkit and its data on {path-to-mrcnn-object-detection}/datasets/VOC2007/VOCdevkit/VOC2007 
11. Place the VOCdevkit of VOC2012 on {path-to-mrcnn-object-detection}/datasets/VOC2012/VOCdevkit and its data on {path-to-mrcnn-object-detection}/datasets/VOC2012/VOCdevkit/VOC2012

*###########################################################################*

Demos:
1) "{path-to-mrcnn-object-detection}/code/example/demo_MRCNN_detection.m" detects objects in an image using the Multi-Region CNN recognition model (section 3 of the technical report). For this demo the semantic segmentation aware features and the object localization module are not being used.
2) "{path-to-mrcnn-object-detection}/code/example/demo_MRCNN_with_Iterative_Localization.m" detects objects in an image using the Multi-Region CNN recognition model (section 3 of the technical report) and the Iterative Localization scheme (section 5 of the technical report). For this demo the semantic segmentation aware features are not being used.
3) "{path-to-mrcnn-object-detection}/code/example/demo_MRCNN_with_SCNN_detection.m" detects objects in an image using the Multi-Region with the semantic segmentation-aware CNN features recognition model (sections 3 and 4 of the technical report). For this demo the object localization module is not being used.
4) "{path-to-mrcnn-object-detection}/code/example/demo_MRCNN_with_SCNN_and_Iterative_Localization.m" detects objects in an image using the Multi-Region with the semantic segmentation-aware CNN features recognition model (sections 3 and 4 of the technical report) and the Iterative Localization scheme (section 5 of the technical report). 

To run the above demos you will require a GPU with at least 12 Gbytes of memory

*###########################################################################*

Testing the pre-trained models on VOC2007 test set: 

1. Test the multi-region CNN recognition model coupled with the iterative bounding box localization module on the VOC2007 test set by running:  
a) script_extract_vgg16_conv_features('test', '2007', 'gpu_id', 1); 
b) script_test_object_detection_iter_loc('MRCNN_VOC2007_2012', 'vgg_bbox_regression_R0013_voc2012_2007', 'gpu_id', 1, 'image_set_test', 'test', 'voc_year_test','2007');
During the a) step, the VGG16 conv5 feature maps for the scales 480, 576, 688, 874, and 1200 are pre-cached (see activation maps in section 3 of the technical report). 
During the b) step, the detection pipeline is applied on the images of VOC2007 test set. By default, this script uses the edge box proposals as input to the detection pipeline. 
The gpu_id parameter is a one-based index of the GPU that will be used for running the experiments; if a non positive value is given then the CPU will be used instead.
 
2. Test the multi-region with the semantic segmentation aware cnn features recognition model coupled with the iterative bounding box localization module on the VOC2007 test set by running:  
a) script_extract_vgg16_conv_features('test', '2007', 'gpu_id', 1); 
b) script_extract_sem_seg_aware_features('test', '2007', 'gpu_id', 1);
c) script_test_object_detection_iter_loc('MRCNN_SEMANTIC_FEATURES_VOC2007_2012', 'vgg_bbox_regression_R0013_voc2012_2007', 'gpu_id', 1, 'image_set_test', 'test', 'voc_year_test','2007');   
During the a) step, the VGG16 conv5 feature maps for the scales 576, 874, and 1200 are pre-cached (see section 3 of the technical report). 
During the b) step, the semantic segmentation aware activation maps for the scales 480, 576, 688, 874, and 1200 are pre-cached (see section 4 of the technical report). To run this script you must run the script of step a) before.
During the c) step, the detection pipeline is applied on the images of VOC2007 test set. By default, this script uses the edge box proposals as input to the detection pipeline. 

The above script can run on a GPU with at least 6 Gbytes of memory

*###########################################################################*
