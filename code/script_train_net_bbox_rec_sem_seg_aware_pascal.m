function script_train_net_bbox_rec_sem_seg_aware_pascal(model_dir_name, varargin)

 %************************** OPTIONS *************************************
ip = inputParser;
ip.addParamValue('gpu_id', 0,        @isscalar);
ip.addParamValue('feat_cache_names', {'VGG_ILSVRC_16_layers'}, @iscell);

ip.addParamValue('train_set',              {'trainval','trainval'})
ip.addParamValue('voc_year_train',         {'2007','2012'})
ip.addParamValue('proposals_method_train', {'selective_search','edge_boxes'});
ip.addParamValue('train_use_flips',         true, @islogical);

ip.addParamValue('val_set',              {'test'})
ip.addParamValue('voc_year_val',         {'2007'})
ip.addParamValue('proposals_method_val', {'selective_search'});
ip.addParamValue('val_use_flips',         false, @islogical);

ip.addParamValue('vgg_pool_params_def',   fullfile(pwd,'data/vgg_pretrained_models/vgg_region_config.m'), @ischar); 
ip.addParamValue('net_file',              fullfile(pwd,'data/vgg_pretrained_models/VGG_ILSVRC_16_layers_Scaled05.caffemodel'), @ischar);
ip.addParamValue('finetune_net_def_file',  'VGG16_Region_Adaptation_Module_train_test_solver.prototxt', @ischar);
ip.addParamValue('solverstate',  '', @ischar)

ip.addParamValue('scale_inner',   0.0,   @isnumeric);
ip.addParamValue('scale_outer',   1.0,   @isnumeric);
ip.addParamValue('half_bbox',      [],   @isnumeric);
ip.addParamValue('feat_id',         1,   @isnumeric);
ip.addParamValue('num_threads',     6,   @isnumeric);

ip.addParamValue('finetuned_modelname', '', @ischar);

ip.parse(varargin{:});
opts = ip.Results;

clc;

opts.finetune_rst_dir       = fullfile(pwd, 'models-exps', model_dir_name);
opts.finetune_net_def_file  = fullfile(pwd, 'model-defs', opts.finetune_net_def_file);
opts.finetune_cache_name    = opts.feat_cache_names{1};
mkdir_if_missing(opts.finetune_rst_dir);

opts.save_mat_model_only = false;
if ~isempty(opts.finetuned_modelname)
    opts.save_mat_model_only = true;
end

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

data_param = struct;
data_param.img_num_per_iter = 128; % should be same with the prototxt file
data_param.random_scale     = 1;
data_param.iter_per_batch   = 125; % for load data effectively 
data_param.fg_fraction      = 0.25;
data_param.fg_threshold     = 0.5;
data_param.bg_threshold     = [0.1 0.5];
data_param.test_iter        = 4  * data_param.iter_per_batch;
data_param.test_interval    = 16 * data_param.iter_per_batch; 
data_param.nTimesMoreData   = 3;
data_param.feat_dim         = 512 * 7 * 7;
data_param.num_classes      = 20;

data_param.num_threads      = opts.num_threads;
opts.data_param             = data_param;

pooler = load_pooling_params(opts.vgg_pool_params_def, ...
    'scale_inner',   opts.scale_inner, ...
    'scale_outer',   opts.scale_outer, ...
    'half_bbox',       opts.half_bbox, ...
    'feat_id',         opts.feat_id);

if ~isempty(opts.solverstate)
    opts.solver_state_file = fullfile(opts.finetune_rst_dir, [opts.solverstate, '.solverstate']);
    assert(exist(opts.solver_state_file,'file')>0);
end

if opts.save_mat_model_only
    finetuned_model_path = fullfile(opts.finetune_rst_dir, [opts.finetuned_modelname,'.caffemodel']);
else
    caffe.reset_all();
    caffe_set_device( opts.gpu_id );
    finetuned_model_path = train_net_bbox_rec(...
        image_db_train, image_db_val, pooler, opts);
    diary off;
    caffe.reset_all();    
end

assert(exist(finetuned_model_path,'file')>0);

deploy_net_file        = 'deploy_softmax.prototxt';
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
model.score_out_blob   = 'fc8_pascal';
model_filename         = fullfile(opts.finetune_rst_dir, 'detection_model_softmax.mat');
save(model_filename, 'model');

deploy_net_file        = 'deploy_svm.prototxt';
model_net_def_file     = fullfile(opts.finetune_rst_dir, deploy_net_file);
model                  = struct;
model.net_def_file     = model_net_def_file;
model.net_weights_file = {finetuned_model_path};
model.pooler           = pooler;
model.feat_blob_name   = feat_blob_name;
model.feat_cache       = opts.feat_cache_names;
model.classes          = VOCopts.classes;
model.score_out_blob   = 'pascal_svm';
model_filename         = fullfile(opts.finetune_rst_dir, 'detection_model_svm.mat');
save(model_filename, 'model');
end

function script_train_net_bbox_rec_sem_seg_aware_pascal_vvv(model_dir_name, varargin)

 %************************** OPTIONS *************************************
ip = inputParser;
ip.addParamValue('gpu_id', 0,        @isscalar);
ip.addParamValue('feat_cache_names', {'VGG_ILSVRC_16_layers_3_scales_dense_f7_512'}, @iscell);

ip.addParamValue('train_set',              {'trainval','trainval'})
ip.addParamValue('voc_year_train',         {'2007','2012'})
ip.addParamValue('proposals_method_train', {'selective_search','edge_boxes'});
ip.addParamValue('train_use_flips',         false, @islogical);

ip.addParamValue('val_set',              {'test'})
ip.addParamValue('voc_year_val',         {'2007'})
ip.addParamValue('proposals_method_val', {'selective_search'});
ip.addParamValue('val_use_flips',         false, @islogical);

ip.addParamValue('vgg_pool_params_def',   fullfile(pwd,'data/vgg_pretrained_models/vgg_semantic_region_config'), @ischar); 
ip.addParamValue('net_file',              fullfile(pwd,'data/vgg_pretrained_models/VGG_ILSVRC_16_layers.caffemodel'), @ischar);
ip.addParamValue('finetune_net_def_file', '/VGG_OBJECT_DETECTION_MRCNN/Semantic_segmentation_aware_net_pascal_solver.prototxt', @ischar);

ip.addParamValue('solver_state_filename', '', @ischar)
ip.addParamValue('test_only',             false, @ischar)

ip.addParamValue('context_inner',   0.0,  @isnumeric);
ip.addParamValue('context_outer',   1.5,  @isnumeric);
ip.addParamValue('outer_relative',  true, @islogical); 
ip.addParamValue('half_bbox',       [],   @isnumeric);
ip.addParamValue('do_zero_padding', 1,    @isnumeric);  
ip.addParamValue('feat_id',         1,    @isnumeric);
ip.addParamValue('num_threads',     6,    @isnumeric);
ip.addParamValue('selected_scales', [576 874 1200], @isnumeric);


ip.addParamValue('spm_divs',           9, @isnumeric);
ip.addParamValue('translation',       [], @isnumeric);
ip.addParamValue('reshape_input',   true, @islogical);
ip.addParamValue('test_iter_mult',     1, @isnumeric);

ip.addParamValue('save_mat_model_only', false, @islogical);
ip.addParamValue('finetuned_modelname', '', @ischar);


ip.parse(varargin{:});
opts = ip.Results;

clc;

opts.finetune_rst_dir       = fullfile(pwd, 'models-exps', model_dir_name);
opts.finetune_net_def_file  = fullfile(pwd, 'model-defs', opts.finetune_net_def_file);
opts.finetune_cache_name    = opts.feat_cache_names{1};

mkdir_if_missing(opts.finetune_rst_dir);

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

data_param = struct;
data_param.img_num_per_iter = 128; % should be same with the prototxt file
data_param.random_scale     = 1;
data_param.iter_per_batch   = 250;%250; % for load data effectively 
data_param.fg_fraction      = 0.25;
data_param.fg_threshold     = 0.5;
data_param.bg_threshold     = [0.1 0.5];
data_param.test_iter        = 1 * data_param.iter_per_batch;
data_param.test_interval    = 2 * 4 * data_param.iter_per_batch; 
data_param.nTimesMoreData   = 3;
data_param.feat_dim         = 512 * opts.spm_divs * opts.spm_divs;
data_param.num_classes      = 20;

if opts.reshape_input
    data_param.reshape_data = [opts.spm_divs, opts.spm_divs, 512, data_param.img_num_per_iter];
end

data_param.num_threads = opts.num_threads;
opts.data_param        = data_param;

pooler = load_pooling_params(opts.vgg_pool_params_def, ...
    'context_inner',   opts.context_inner, ...
    'context_outer',   opts.context_outer, ...
    'outer_relative',  opts.outer_relative, ...
    'half_bbox',       opts.half_bbox, ...
    'do_zero_padding', opts.do_zero_padding, ...
    'feat_id',         opts.feat_id, ...
    'selected_scales', opts.selected_scales);
pooler.spm_divs = opts.spm_divs;

if ~isempty(opts.translation) 
    assert(length(opts.translation)==2);
    pooler.translation = opts.translation; 
end

if ~isempty(opts.solver_state_filename)
    opts.solver_state_file = fullfile(opts.finetune_rst_dir, [opts.solver_state_filename, '.solverstate']);
    assert(exist(opts.solver_state_file,'file')>0);
end

if opts.save_mat_model_only
    finetuned_model_path = fullfile(opts.finetune_rst_dir, [opts.finetuned_modelname,'.caffemodel']);
else
    caffe.reset_all();
    caffe_set_device(opts.gpu_id);
    finetuned_model_path = train_net_bbox_rec(...
        image_db_train, image_db_val, pooler, opts);
    diary off;
    caffe.reset_all();    
end

assert(exist(finetuned_model_path,'file')>0);

deploy_net_file        = 'deploy_softmax.prototxt';
model_net_def_file     = fullfile(opts.finetune_rst_dir, deploy_net_file);
feat_blob_name         = {'fc1'};
VOCopts                = initVOCOpts( '/home/spyros/Documents/projects/VOC2007/VOCdevkit', '2007');

model                  = struct;
model.net_def_file     = model_net_def_file;
model.net_weights_file = {finetuned_model_path};
model.pooler           = pooler;
model.feat_blob_name   = feat_blob_name;
model.feat_cache       = opts.feat_cache_names;
model.classes          = VOCopts.classes;
model.score_out_blob   = 'fc8_pascal';

model_filename         = fullfile(opts.finetune_rst_dir, 'detection_model_softmax.mat');
save(model_filename, 'model');

deploy_net_file        = 'deploy_svm.prototxt';
model_net_def_file     = fullfile(opts.finetune_rst_dir, deploy_net_file);
model                  = struct;
model.net_def_file     = model_net_def_file;
model.net_weights_file = {finetuned_model_path};
model.pooler           = pooler;
model.feat_blob_name   = feat_blob_name;
model.feat_cache       = opts.feat_cache_names;
model.classes          = VOCopts.classes;
model.score_out_blob   = 'pascal_svm';

model_filename         = fullfile(opts.finetune_rst_dir, 'detection_model_svm.mat');
save(model_filename, 'model');
end
