
% extact the vgg16 conv5 image features from pascal voc 2007 test set
script_extract_vgg16_conv_features('test', '2007', 'gpu_id', 1); 

% extact the semantic segmentation aware CNN image features from the 
% pascal voc 2007 test set
script_extract_sem_seg_aware_features('test', '2007', 'gpu_id', 1);

