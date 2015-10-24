function aboxes_out = post_process_scored_bboxes(...
    aboxes_in, nms_over_thrs, thresh_val, varargin)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% Part of the code in this file comes from the R-CNN code: 
% https://github.com/rbgirshick/rcnn
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addParamValue('is_per_class',      false, @islogical);
ip.addParamValue('do_bbox_voting',    false, @islogical);
ip.addParamValue('box_ave_thresh',      0.5, @isnumeric);
ip.addParamValue('use_not_static_thr', true, @islogical);
ip.addParamValue('ave_per_image',         5, @isnumeric);
ip.addParamValue('max_per_image',       200, @isnumeric);
ip.addParamValue('add_val',             1.5, @isnumeric);

ip.parse(varargin{:});
opts = ip.Results;

is_per_class       = opts.is_per_class;
do_bbox_voting     = opts.do_bbox_voting;
box_ave_thresh     = opts.box_ave_thresh;
use_not_static_thr = opts.use_not_static_thr;
max_per_image      = opts.max_per_image;
ave_per_image      = opts.ave_per_image;
add_val            = opts.add_val;

if ~is_per_class
    num_imgs    = length(aboxes_in);
    num_classes = size(aboxes_in{1},2) - 4;
else
    num_classes = length(aboxes_in);
    num_imgs    = length(aboxes_in{1});
end

max_per_set   = ceil(ave_per_image * num_imgs);
aboxes_out    = cell(num_classes, 1);
for i = 1:num_classes, aboxes_out{i} = cell(num_imgs, 1); end

thresh_val_all = ones(num_classes,1,'single') * thresh_val;

for i = 1:num_imgs    
    for j = 1:num_classes
        if ~is_per_class
            assert(size(aboxes_in{i},2) == (4 + num_classes));
            bbox_det_cands = [aboxes_in{i}(:,1:4), aboxes_in{i}(:,4+j)];
        else
%             size(aboxes_in{j}{i})
            assert(size(aboxes_in{j}{i},2) == 5);
            bbox_det_cands = aboxes_in{j}{i};
        end
        
        reject = (any(isnan(bbox_det_cands),2) | any(isinf(bbox_det_cands),2));
        bbox_det_cands = bbox_det_cands(~reject,:);
        bbox_dets      = post_process_bboxes(bbox_det_cands, thresh_val_all(j), nms_over_thrs, max_per_image);

        if (do_bbox_voting && ~isempty(bbox_dets))
            bbox_dets = bbox_voting(bbox_det_cands, bbox_dets, box_ave_thresh, add_val);
        end
        
        aboxes_out{j}{i} =  bbox_dets;
    end
    
    if (mod(i, 1000) == 0) && use_not_static_thr
        for j = 1:num_classes
            [aboxes_out{j}, thresh_val_all(j)] = keep_top_k(aboxes_out{j}, i, max_per_set, thresh_val_all(j));
        end
    end
end

if use_not_static_thr
    disp(thresh_val_all(:)');
    for i = 1:num_classes
        % go back through and prune out detections below the found threshold
        aboxes_out{i} = prune_detections(aboxes_out{i}, thresh_val_all(i));
    end
end

total_num_bboxes = zeros(num_classes, 1);
for j = 1:num_classes
    total_num_bboxes(j) = sum(cellfun(@(x) size(x,1), aboxes_out{j}, 'UniformOutput', true));
end

fprintf('Average number of bounding boxes per class\n');
disp(total_num_bboxes(:)' / num_imgs);
fprintf('Average number of bounding boxes in total\n');
disp(sum(total_num_bboxes) / num_imgs);
end

function [bbox_dets, indices] = post_process_bboxes(bbox_det_cands, score_thresh, nms_over_thrs, max_per_image)

bbox_dets = zeros(0, 5, 'single');
if ~isempty(bbox_det_cands)
    indices = find(bbox_det_cands(:,5) > score_thresh);
    keep    = nms(single(bbox_det_cands(indices,:)), nms_over_thrs);
    indices = indices(keep);
    if ~isempty(indices)
        [~, order] = sort(bbox_det_cands(indices,5), 'descend');
        order      = order(1:min(length(order), max_per_image));
        indices    = indices(order);
        bbox_dets  = bbox_det_cands(indices,:);
    end
end

end

function bbox_dets = bbox_voting(bbox_dets_cands, bbox_dets, over_thresh, add_val)

num_dets = size(bbox_dets,1);

for p = 1:num_dets
    overlap = boxoverlap(bbox_dets_cands(:,1:4), bbox_dets(p,1:4));
    keep    = overlap >= over_thresh;
    bbox_dets_local_this = bbox_dets_cands(keep,:);
    bbox_dets_local_this(:,5) = eps + max(0, bbox_dets_local_this(:,5) + add_val);
    if any(keep)
        bbox_tmp         = sum(bbox_dets_local_this(:,1:4) .* repmat(bbox_dets_local_this(:,5), [1, 4]), 1);
        bbox_tmp         = bbox_tmp ./ repmat(sum(bbox_dets_local_this(:,5)), [1, 4]);
        bbox_dets(p,1:4) = bbox_tmp;
    end
end

end

function [boxes, thresh] = keep_top_k(boxes, end_at, top_k, thresh)
% ------------------------------------------------------------------------
% Keep top K
X = cat(1, boxes{1:end_at});
if isempty(X), return; end

scores = sort(X(:,end), 'descend');
thresh = scores(min(length(scores), top_k));
for image_index = 1:end_at
    bbox = boxes{image_index};
    keep = find(bbox(:,end) >= thresh);
    boxes{image_index} = bbox(keep,:);
end

end

function bbox_dets = prune_detections(bbox_dets, thresh)
for j = 1:length(bbox_dets)
    if ~isempty(bbox_dets{j})
        bbox_dets{j}(bbox_dets{j}(:,end) < thresh ,:) = [];
    end
end
end

