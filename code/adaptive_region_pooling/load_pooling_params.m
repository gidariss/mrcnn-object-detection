function pool_params = load_pooling_params(pool_params_def, varargin)
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

ip = inputParser;
ip.addParamValue('scale_inner',   [], @isnumeric);
ip.addParamValue('scale_outer',   [], @isnumeric);
ip.addParamValue('half_bbox',     [], @isnumeric);
ip.addParamValue('feat_id',        1,  @isnumeric);

ip.parse(varargin{:});
opts = ip.Results;

[~, ~, ext] = fileparts(pool_params_def);
if isempty(ext), pool_params_def = [pool_params_def, '.m']; end
assert(exist(pool_params_def, 'file') ~= 0);

% change folder to avoid too long path for eval()
cur_dir = pwd;
[pool_def_dir, pool_def_file] = fileparts(pool_params_def);

cd(pool_def_dir);
pool_params = eval(pool_def_file);
cd(cur_dir);

pool_params.scale_inner = opts.scale_inner;
pool_params.scale_outer = opts.scale_outer;
pool_params.half_bbox   = opts.half_bbox;
pool_params.feat_id     = opts.feat_id;
end
