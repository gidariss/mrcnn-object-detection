function feat_data = get_conv_feat_data(net, image, scales, mean_pix)
% get_conv_feat_data extract the convolutional features of one image
% for the specified scales using the convolutional neural network net. 
% mean_pix is a 3 x 1 or 1 x 3 array with the mean pixel value per color 
% channel that is subtracted from the scaled image before is being fed to
% the convolutional neural network.
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

feat_data = init_feat_data();
feat_data.feat.im_height = size(image,1);
feat_data.feat.im_width  = size(image,2);
[feat_data.feat.rsp, feat_data.feat.scale] = extract_conv_features(...
    net, image, scales, mean_pix);
end

function d = init_feat_data()
d.gt       = []; 
d.overlap  = []; 
d.boxes    = []; 
d.class    = []; 
d.feat     = [];
end

