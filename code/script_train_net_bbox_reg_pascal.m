function script_train_net_bbox_reg_pascal(model_dir_name, varargin)

 %************************** OPTIONS *************************************
ip = inputParser;
ip.addParamValue('gpu_id', 0,        @isscalar);
ip.addParamValue('feat_cache_names', {'VGG_ILSVRC_16_layers'}, @iscell);

ip.addParamValue('train_set',              {'trainval','trainval'})
ip.addParamValue('voc_year_train',         {'2007','2012'})
ip.addParamValue('proposals_method_train', {'selective_search','edge_boxes'});
ip.addParamValue('train_use_flips',        true, @islogical);

% ip.addParamValue('train_set',              {'trainval'})
% ip.addParamValue('voc_year_train',         {'2007'})
% ip.addParamValue('proposals_method_train', {'selective_search'});
% ip.addParamValue('train_use_flips',        true, @islogical);

ip.addParamValue('val_set',               {'test'})
ip.addParamValue('voc_year_val',          {'2007'})
ip.addParamValue('proposals_method_val',  {'selective_search'});
ip.addParamValue('val_use_flips',         false, @islogical);
 
ip.addParamValue('vgg_pool_params_def',    fullfile(pwd,'data/vgg_pretrained_models/vgg_region_config.m'), @ischar); 
ip.addParamValue('net_file',               fullfile(pwd,'data/vgg_pretrained_models/VGG_ILSVRC_16_Fully_Connected_Layers.caffemodel'), @ischar);
ip.addParamValue('finetune_net_def_file',  'VGG16_Region_Adaptation_BBox_Regression_Module_train_test_solver.prototxt', @ischar);
ip.addParamValue('solverstate',  '', @ischar)

ip.addParamValue('test_only',              false, @ischar)

ip.addParamValue('scale_inner',   0.0, @isnumeric);
ip.addParamValue('scale_outer',   1.3, @isnumeric);
ip.addParamValue('half_bbox',     [],  @isnumeric);
ip.addParamValue('feat_id',        1,  @isnumeric);

ip.addParamValue('num_threads',     6,  @isnumeric);

ip.addParamValue('fg_threshold',      0.4, @isnumeric);
ip.addParamValue('fg_threshold_test', 0.4, @isnumeric);

ip.addParamValue('save_mat_model_only', false, @islogical);
ip.addParamValue('finetuned_modelname', '',    @ischar);


ip.parse(varargin{:});
opts = ip.Results;

clc;

opts.finetune_rst_dir       = fullfile(pwd, 'models-exps', model_dir_name);
opts.finetune_net_def_file  = fullfile(pwd, 'model-defs', opts.finetune_net_def_file);
mkdir_if_missing(opts.finetune_rst_dir);

% opts.net_file = '/home/spyros/Documents/projects/mrcnn-object-detection/models-exps/vgg_bbox_reg_R0013_voc2012_2007/best_model_iter_234000.caffemodel';

disp(opts)
if ~opts.save_mat_model_only
    image_db_train = load_image_dataset(...
        'image_set', opts.train_set, ...
        'voc_year', opts.voc_year_train, ...
        'proposals_method', opts.proposals_method_train,...
        'feat_cache_names', opts.feat_cache_names, ...
        'use_flips', opts.train_use_flips);
    
    image_db_val = load_image_dataset(...
        'image_set', opts.val_set, ...
        'voc_year', opts.voc_year_val, ...
        'proposals_method', opts.proposals_method_val,...
        'feat_cache_names', opts.feat_cache_names, ...
        'use_flips', opts.val_use_flips); 
end

[solver_file, ~, test_net_file, opts.max_iter, opts.snapshot_prefix] = ...
    parse_copy_finetune_prototxt(...
    opts.finetune_net_def_file, opts.finetune_rst_dir);

opts.finetune_net_def_file = fullfile(opts.finetune_rst_dir, solver_file);
assert(exist(opts.finetune_net_def_file,'file')>0)
assert(exist(opts.net_file,'file')>0)

data_param                   = struct;
data_param.img_num_per_iter  = 128; 
data_param.random_scale      = 1;
data_param.iter_per_batch    = 250; 
data_param.fg_threshold      = opts.fg_threshold;
data_param.test_iter         = 2 * data_param.iter_per_batch;
data_param.test_interval     = 2 * 4 * data_param.iter_per_batch; 
data_param.nTimesMoreData    = 2;
data_param.feat_dim          = 512 * 7 * 7;
data_param.num_classes       = 20;
data_param.num_threads       = opts.num_threads;
data_param.labels_split_div  = [data_param.num_classes*4, data_param.num_classes*4];
opts.data_param              = data_param;

pooler = load_pooling_params(opts.vgg_pool_params_def, ...
    'scale_inner', opts.scale_inner, ...
    'scale_outer', opts.scale_outer, ...
    'half_bbox',   opts.half_bbox, ...
    'feat_id',     opts.feat_id);

if ~isempty(opts.solverstate)
    opts.solver_state_file = fullfile(opts.finetune_rst_dir, [opts.solverstate, '.solverstate']);
    assert(exist(opts.solver_state_file,'file')>0);
end

if opts.save_mat_model_only
    finetuned_model_path = fullfile(opts.finetune_rst_dir, [opts.finetuned_modelname,'.caffemodel']);
else
    caffe.reset_all();
    caffe_set_device( opts.gpu_id );
    finetuned_model_path = train_net_bbox_reg(...
        image_db_train, image_db_val, pooler, opts);
    diary off;
    caffe.reset_all();    
end

assert(exist(finetuned_model_path,'file')>0);

deploy_net_file        = 'deploy_regression.prototxt';
model_net_def_file     = fullfile(opts.finetune_rst_dir, deploy_net_file);
feat_blob_name         = {'fc7'};
VOCopts                = initVOCOpts( '/home/spyros/Documents/projects/VOC2007/VOCdevkit', '2007');

model                  = struct;
model.net_def_file     = model_net_def_file;
model.net_weights_file = {finetuned_model_path};
model.pooler           = pooler;
model.feat_blob_name   = feat_blob_name;
model.feat_cache       = opts.feat_cache_names;
model.classes          = VOCopts.classes;
model_filename         = fullfile(opts.finetune_rst_dir, 'regression_model.mat');
save(model_filename, 'model');
end