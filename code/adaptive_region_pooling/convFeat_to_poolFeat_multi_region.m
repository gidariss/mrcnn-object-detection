function [pool_feat, regions] = convFeat_to_poolFeat_multi_region(pooler, feat, boxes, random_scale)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% Part of the code in this file comes from the SPP-Net code: 
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

num_poolX  = length(pooler);
pool_feat  = cell(num_poolX,1);
regions    = cell(num_poolX,1);

for p = 1:num_poolX, pool_feat{p} = single([]); end
for p = 1:num_poolX, regions{p}   = single([]); end

if isempty(boxes), return; end

if length(feat) == 1
    for p = 1:num_poolX 
        [pool_feat{p}, regions{p}] = convFeat_to_poolFeat(pooler(p), feat, boxes, random_scale);
    end
else
    if iscell(feat)
        for p = 1:num_poolX 
            [pool_feat{p}, regions{p}]  = convFeat_to_poolFeat(pooler(p), feat{pooler(p).feat_id}, boxes, random_scale);
        end         
    elseif isstruct(feat)
        for p = 1:num_poolX 
            [pool_feat{p}, regions{p}]  = convFeat_to_poolFeat(pooler(p), feat(pooler(p).feat_id), boxes, random_scale);
        end        
    end
end
end
