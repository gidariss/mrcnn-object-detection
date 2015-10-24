function startup()

curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir, 'code')));

addpath(fullfile(curdir, 'bin'));

mkdir_if_missing(fullfile(curdir, 'external'));
caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');
if exist(caffe_path, 'dir') == 0
    error('matcaffe is missing from external/caffe/matlab; See README.md');
end
addpath(genpath(caffe_path));

mkdir_if_missing(fullfile(curdir, 'models-exps'));
mkdir_if_missing(fullfile(curdir, 'feat_cache'));
mkdir_if_missing(fullfile(curdir, 'data'));

end
