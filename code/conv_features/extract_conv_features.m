function [rsp, scale] = extract_conv_features(net, img, scale, mean_pix)
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


if size(img,3) == 1, img = repmat(img, [1, 1, 3]); end
rsp = {};
num_scales = length(scale);
% fprintf('scales = {')
for i = 1:num_scales
    scale_this = scale(i);
    img_scaled = preprocess_img(img, scale_this, mean_pix);
    
    if (numel(img_scaled) > 1200*2400*3) 
        scale = scale(1:(i-1));
        rsp   = rsp(1:(i-1));
%         fprintf('Does not fit in the memory\n');
        break;
    end
%     fprintf('[%d %d] ', size(img_scaled,2), size(img_scaled,1));
    
    net = caffe_reshape_net(net, {[size(img_scaled), 1]});
    response = net.forward({img_scaled});
    rsp{i} = permute(response{1}, [2, 1, 3]);
end
%fprintf('}')
end

function im_scaled_size = get_scaled_image_size(im_size, scale_sel)
im_height = im_size(1);
im_width  = im_size(2);
if (im_width < im_height)
    im_resized_width  = scale_sel;
    im_resized_height = ceil(im_resized_width * im_height / im_width);
else
    im_resized_height = scale_sel;
    im_resized_width  = ceil(im_resized_height * im_width / im_height);
end
im_scaled_size = [im_resized_height, im_resized_width];
end

function img = preprocess_img(img, scale, mean_pix)
if numel(mean_pix) == 1, mean_pix = [mean_pix, mean_pix, mean_pix]; end

im_height = size(img, 1);
im_width  = size(img, 2);
im_scaled_size = get_scaled_image_size([im_height, im_width], scale);

if (scale <= 224)
    img = imresize(img, [im_scaled_size(1), im_scaled_size(2)], 'bilinear');
else
    img = imresize(img, [im_scaled_size(1), im_scaled_size(2)], 'bilinear', 'antialiasing', false);
end

img = single(img);
img = img(:, :, [3 2 1]);
for c = 1:3, img(:,:,c) = img(:,:,c) - mean_pix(c); end
img = permute(img, [2 1 3]);
end
