function demo_MRCNN_detection()

gpu_id = 1; % gpu_id is a one-based index; if a non positive value is given
% then the CPU will be used instead.  

caffe_set_device( gpu_id );
caffe.reset_all();

%***************************** LOAD MODEL *********************************

% set the path of the bounding box recognition moddel for object detection
model_rec_dir_name  = 'MRCNN_VOC2007_2012'; % model's directory name 
full_model_rec_dir  = fullfile(pwd, 'models-exps', model_rec_dir_name); % full path to the model's directory
use_detection_svms  = true;
model_rec_mat_name  = 'detection_model_svm.mat'; % model's matlab filename
full_model_rec_path = fullfile(full_model_rec_dir, model_rec_mat_name); % full path to the model's matlab file
assert(exist(full_model_rec_dir,'dir')>0);
assert(exist(full_model_rec_path,'file')>0);

% Load the bounding box recognition moddel for object detection
ld = load(full_model_rec_path, 'model');
model_obj_rec = ld.model; 
model_phase_rec = 'test';
clear ld; 

model_obj_rec = load_object_recognition_model_on_caffe(...
    model_obj_rec, use_detection_svms, model_phase_rec, full_model_rec_dir);

% Load the activation maps module that is responsible for producing the
% convolutional features (called activation maps) of an image. For the
% activation maps module we use the convolutional layers (till conv5_3)
% of the VGG16 model

% set the path to the directory that contain the caffe defintion and 
% weights files of the  activation maps module
net_files_dir = fullfile(pwd,'data','vgg_pretrained_models'); 
% path to the defintion file of the activation maps module
model_obj_rec.act_net_def_file     = fullfile( net_files_dir,'vgg16_conv5_deploy.prototxt');
% path to the weights file of the activation maps module
model_obj_rec.act_net_weights_file = {fullfile(net_files_dir,'VGG_ILSVRC_16_layers.caffemodel')};
assert(exist(net_files_dir,'dir')>0);
assert(exist(model_obj_rec.act_net_def_file ,'file')>0);
assert(exist(model_obj_rec.act_net_weights_file{1},'file')>0);
% image scales that are being used for extracting the activation maps 
model_obj_rec.scales   = [480 576 688 874 1200]; 
% mean pixel value per color channel for image pre-processing before 
% feeding it to the VGG16 convolutional layers.
model_obj_rec.mean_pix = [103.939, 116.779, 123.68]; 
% load the activation maps module on caffe
model_obj_rec.act_maps_net = caffe_load_model( model_obj_rec.act_net_def_file, model_obj_rec.act_net_weights_file);
%**************************************************************************

img  = imread('./code/examples/images/fish-bike.jpg'); % load image

conf = struct;
% the threholds that are being used for removing easy negatives before the
% non-max-suppression step
conf.thresh = -3 * ones(length(model_obj_rec.classes),1); % It contains the 
% threshold per category that will be used for removing scored boxes with 
% low confidence prior to applying the non-max-suppression step.
conf.nms_over_thrs = 0.3; % IoU threshold for the non-max-suppression step
conf.box_method = 'edge_boxes'; % string with the box proposals algorithm that
% will be used in order to generate the set of candidate boxes. Currently 
% it supports the 'edge_boxes' or the 'selective_search' types only.

% detect object in the image
[ bbox_detections ] = demo_object_detection( img, model_obj_rec, conf );


score_thresh = -1.5;

all_dets = [];
for i = 1:length(bbox_detections)
  all_dets = cat(1, all_dets, ...
      [i * ones(size(bbox_detections{i}, 1), 1) bbox_detections{i}]);
end

[~, ord] = sort(all_dets(:,end), 'descend');
for i = 1:length(ord)
  score = all_dets(ord(i), end);
  if score < score_thresh
    break;
  end
  cls = model_obj_rec.classes{all_dets(ord(i), 1)};
  showboxes(img, all_dets(ord(i), 2:5));
  title(sprintf('det #%d: %s score = %.3f', ...
      i, cls, score));
  drawnow;
  pause;
end

fprintf('No more detection with score >= 0\n');
end