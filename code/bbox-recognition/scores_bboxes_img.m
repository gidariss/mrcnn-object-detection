function [ scores, feats_pool ] = scores_bboxes_img( model, in, bboxes )
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

num_classes = length(model.classes);

if ~isempty(bboxes)
    feats_pool  = convFeat_to_poolFeat_multi_region(model.pooler, in, bboxes(:,1:4));
    outputs = caffe_forward_net(model.net, feats_pool);
    scores  = outputs{1}';
else
    feats_pool = {};
    scores     = zeros(0,num_classes,'single');
end
% in case of background class 
if size(scores,2) == (num_classes + 1), scores = scores(:,2:end); end
end
