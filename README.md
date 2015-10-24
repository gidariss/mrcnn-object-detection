## *Object detection via a multi-region & semantic segmentation-aware CNN model*

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

License:  
This code is released under the MIT License (refer to the LICENSE file for details).  

Requirements:  
Software:  
MATLAB   
Caffe: https://github.com/BVLC/caffe. 
LIBLINEAR (only for training)  

Data:  
PASCAL VOC2007  
PASCAL VOC2012  
selective search box proposals for the above datasets.    
edge box proposals for the above datasets.    

Preparation for Testing on VOC2007 test set:  
1. Place a soft link of caffe directory on ./external/caffe  
2. Place the VOCdevkit of VOC2007 on ./datasets/VOC2007/VOCdevkit and its data on ./datasets/VOC2007/VOCdevkit/VOC2007  
3. Place the selective search and edge box proposals on the following directories   
i)  ./projects/object-localization/data/edge_boxes_data  
ii) ./projects/object-localization/data/selective_search_data  

4. Place the models on the following directories:  
  
	I)   ./models-exps/vgg_R0010_voc2012_2007  : original box region only recogntion model.  
	II)  ./models-exps/MRCNN_VOC2007_2012  : multi-region recognition model.  
	III) the multi-region recognition model with the semantic segmentation aware feautures is not ready yet because there is still some cleaning up needed for the corresponding code. It will be available after the CVPR2016 deadline.  
	IV)  ./models-exps/vgg_bbox_regression_R0013_voc2012_2007/ : bounding box regression model.  
	V)   ./data/vgg_pretrained_models: the VGG16 model pretained trained on ImageNet for Image Classification  task.  

5. Run mrcnn_build.m  
6. Run startup.m  
7. Pre-cache the VGG16 conv5 features of the images in VOC2007 test set (for the scales 480, 576, 688, 874, and 1200) by running on matlab:  
script_extract_vgg16_conv_features('test', '2007', 'gpu_id', 1);   
gpu_id is a one-based index; if the a non positive value is given then the CPU will be used instead. It should take around 2-3 hours for the VOC2007 test set.  

8. To test the multi-region CNN recognition model coupled with the iterative bounding box localization module on the VOC2007 test set run:  
script_test_object_detection_iter_loc('MRCNN_VOC2007_2012', 'vgg_bbox_regression_R0013_voc2012_2007', 'gpu_id', 1);   
By default, the above script uses the edge box proposals.  

