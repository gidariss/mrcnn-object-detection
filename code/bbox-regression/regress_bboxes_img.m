function bboxes_out = regress_bboxes_img( model, in, bboxes_in, img_size)
% bboxes_in : is a N x 5 matrix. Each row is a bounding with the values 
%   [x0,y0,x1,y1,class_index], where x0, y0 are the coordinates of the top
%   left corner and x1,y1 are the coordinates of the bottom right corner.
%   class_index is the class index to which the bounding box belogns to.
%
% bboxes_out : a N x 5 matrix with the refined bounding boxes. It has the
%   same format as bboxes_in
% 
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
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
if ~isempty(bboxes_in)
    feats_pool  = convFeat_to_poolFeat_multi_region(model.pooler, in, bboxes_in(:,1:4));
    outputs     = caffe_forward_net(model.net, feats_pool);
    pred_vals   = outputs{1}';
    assert(size(pred_vals,1) == size(bboxes_in,1));
    assert(size(pred_vals,2) == 4*num_classes);
    pred_vals = reshape(pred_vals, [size(pred_vals,1),4,num_classes]);
    class_indices = bboxes_in(:,5);
    
    bboxes_out = zeros(size(pred_vals,1),4,'single');
    for i = 1:size(bboxes_in,1)
        bboxes_out(i,:) = pred_vals(i,:,class_indices(i));
    end
    
    bboxes_out = decode_reg_vals_to_bbox_targets(bboxes_in(:,1:4), bboxes_out);
    bboxes_out = clip_bbox_inside_the_img(bboxes_out, img_size);
    bboxes_out = [bboxes_out, class_indices];
else
    bboxes_out = zeros(0,5,'single');
end
end

function bboxes = clip_bbox_inside_the_img(bboxes, img_size)
bboxes(:,1) = max(1,           bboxes(:,1));
bboxes(:,2) = max(1,           bboxes(:,2));
bboxes(:,3) = min(img_size(2), bboxes(:,3));
bboxes(:,4) = min(img_size(1), bboxes(:,4));
end
