function [feat_out, regions] = convFeat_to_poolFeat(pooler, feat, boxes, random_scale)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% Part of the code in this file come from the SPP-Net code: 
% https://github.com/ShaoqingRen/SPP_net
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% --------------------------------------------------------- 

if ~exist('random_scale', 'var'), random_scale = false; end

assert(all(pooler.scale_outer >= pooler.scale_inner));

assert(isfield(feat, 'rsp') && ~isempty(feat.rsp));
available_scales = feat.scale(:)'; 
assert(length(feat.rsp) == length(available_scales));

min_img_sz = min(feat.im_height, feat.im_width);

if isempty(boxes)
    feat_out = single([]);
    regions  = single([]);
    return;
end

image_size = [feat.im_height, feat.im_width];
boxes      = transform_bboxes(pooler, boxes);

if isempty(pooler.scale_outer), pooler.scale_outer = 1.0; end
if isempty(pooler.scale_inner), pooler.scale_inner = 0.0; end

boxes_outer = scale_bboxes(boxes, pooler.scale_outer);
boxes_inner = scale_bboxes(boxes, pooler.scale_inner);

regions = single([boxes_outer,boxes_inner]);

expected_scale = spm_expected_scale(min_img_sz, boxes_outer, pooler);

[~, best_scale_ids] = min(abs(bsxfun(@minus, available_scales, expected_scale(:))), [], 2);   
if random_scale
    best_scale_ids      = best_scale_ids + randi([-2,2], size(best_scale_ids));
    best_scale_ids      = max(1,min(length(available_scales), best_scale_ids));
end

boxes_scales       = available_scales(best_scale_ids(:));
scaled_boxes_outer = bsxfun(@times, (boxes_outer - 1), (boxes_scales(:) - 1)) / (min_img_sz - 1) + 1;
scaled_boxes_inner = bsxfun(@times, (boxes_inner - 1), (boxes_scales(:) - 1)) / (min_img_sz - 1) + 1;

scaled_regions = [scaled_boxes_outer, scaled_boxes_inner];

feat_out = adaptive_region_pooling(...
    feat.rsp, pooler.spm_divs, scaled_regions, best_scale_ids, ...
    pooler.offset0, pooler.offset, pooler.step_standard); 

end

function [ bboxes ] = scale_bboxes( bboxes, scale_ratio )

if numel(scale_ratio) == 1, scale_ratio(2) = scale_ratio(1); end
assert(numel(scale_ratio) == 2);
scale_ratio = single(scale_ratio);


bboxes_center      = [(bboxes(:,1)+bboxes(:,3)), (bboxes(:,2)+bboxes(:,4))]/2;
bboxes_width_half  = (bboxes(:,3) - bboxes(:,1))/2;
bboxes_width_half  = bboxes_width_half * scale_ratio(2);

bboxes_height_half = (bboxes(:,4) - bboxes(:,2))/2;
bboxes_height_half = bboxes_height_half * scale_ratio(1);

bboxes = round([bboxes_center(:,1) - bboxes_width_half, ...
                bboxes_center(:,2) - bboxes_height_half, ...
                bboxes_center(:,1) + bboxes_width_half, ...
                bboxes_center(:,2) + bboxes_height_half]);
end

function boxes = transform_bboxes(pooler, boxes)
if isfield(pooler, 'half_bbox') && ~isempty(pooler.half_bbox) && pooler.half_bbox > 0
    boxes = get_half_bbox( boxes, pooler.half_bbox );
end
end

function [ bboxes ] = get_half_bbox( bboxes, half_bbox )

assert(half_bbox >= 1 && half_bbox <= 4);

switch half_bbox
    case 1 % left half 
        bboxes_half_width = floor((bboxes(:,3) - bboxes(:,1)+1)/2);
        bboxes(:,3) = bboxes(:,1) + bboxes_half_width;
    case 2 % right half
        bboxes_half_width = floor((bboxes(:,3) - bboxes(:,1)+1)/2);
        bboxes(:,1) = bboxes(:,3) - bboxes_half_width;
    case 3 % up half
        bboxes_half_height = floor((bboxes(:,4) - bboxes(:,2)+1)/2);
        bboxes(:,4) = bboxes(:,2) + bboxes_half_height;
    case 4 % down half
        bboxes_half_height = floor((bboxes(:,4) - bboxes(:,2)+1)/2);
        bboxes(:,2) = bboxes(:,4) - bboxes_half_height;
end
bboxes = round(bboxes);
end

function feats_pooled = adaptive_region_pooling(feats, spm_divs, ...
    scaled_regions, best_scale_ids, offset0, offset, min_times)

feats = cellfun(@(x) single(x), feats, 'UniformOutput', false);
scaled_regions = double(scaled_regions);
boxes_outer    = scaled_regions(:,1:4)';
boxes_inner    = scaled_regions(:,5:8)';

% trans from (height, width, channel) to (channel, width, height)
feats = cellfun(@(x) permute(x, [3, 2, 1]), feats, 'UniformOutput', false);
best_scale_ids = int32(best_scale_ids);

offset0   = double(offset0);
offset    = double(offset);
min_times = double(min_times);
spm_divs  = double(spm_divs);

feats_pooled = adaptive_region_pooling_mex(...
	feats, spm_divs, boxes_outer, boxes_inner, best_scale_ids, ...
    [offset0, offset, min_times]);
end
