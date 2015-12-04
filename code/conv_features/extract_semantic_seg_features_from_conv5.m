function rsp = extract_semantic_seg_features_from_conv5(net, inputs, opts)
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

interleave_num_steps = 0;
if (isfield(opts, 'do_interleave') && opts.do_interleave) 
    interleave_num_steps = opts.interleave_num_steps;
end

if isnumeric(inputs), inputs = {inputs}; end

num_inputs = length(inputs);
rsp = {};
for i = 1:num_inputs
    in = inputs{i};
    in = permute(in, [2, 1, 3]); % change order of width, height for compatibility with caffe
    in = preprocess_input(in, interleave_num_steps);
    
    net         = caffe_reshape_net(net, {[size(in,1), size(in,2), size(in,3), size(in,4)]});
    response    = net.forward({in});
    response{1} = postprocess_output(response{1}, interleave_num_steps);
    rsp{i}      = permute(response{1}, [2, 1, 3]);
end
rsp = rsp(:);
end

function input = preprocess_input(input, interleave_num_steps)
if (interleave_num_steps > 1)
    length_1   = size(input,1) - (interleave_num_steps-1);
    length_2   = size(input,2) - (interleave_num_steps-1);
    batch_size = interleave_num_steps * interleave_num_steps;
    
    batch = cell([1,1,1,batch_size]);
        
    for j = 1:interleave_num_steps
        for i = 1:interleave_num_steps
            inter_i  = (i-1)+(1:length_1);
            inter_j  = (j-1)+(1:length_2);
            batch_id = (j-1)*interleave_num_steps + i;
            batch{batch_id} = input(inter_i,inter_j,:);
        end
    end
    input = cell2mat(batch);
end
end

function final_output = postprocess_output(output, interleave_num_steps)
if (interleave_num_steps > 1)
    final_out_size = [size(output,1)*interleave_num_steps, size(output,2)*interleave_num_steps, size(output,3)];
    final_output   = zeros(final_out_size, 'like', output);
    for j = 1:interleave_num_steps
        for i = 1:interleave_num_steps
            inter_i  = (i-1) + (1:interleave_num_steps:size(final_output,1));
            inter_j  = (j-1) + (1:interleave_num_steps:size(final_output,2));
            batch_id = (j-1)*interleave_num_steps + i;
            final_output(inter_i,inter_j,:) = output(:,:,:,batch_id);
        end
    end
else
    final_output = output;
end
end