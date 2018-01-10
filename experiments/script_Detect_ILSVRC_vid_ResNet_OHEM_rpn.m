% D training using ImageNet model and RPN proposals
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
clc;
clear mex;
clear is_valid_handle;
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe-rfcn';
opts.gpu_id                 = auto_select_gpu;

active_caffe_mex(opts.gpu_id, opts.caffe_version);

res50 = 0 ;
with_ohem = true;
concat = false;
modality = 'rgb';
with_resNext = 1;
root_path = get_root_path;

if res50  
  model                       = Model.ResNet50_for_RFCN_ILSVRCvid_dilation();
  opts.cache_name             = 'rfcn_ILSVRC_ResNet50_OHEM_rpn_dilation';
  conf                        = rfcn_config_ohem('image_means', model.mean_image);
else
  model                       = Model.ResNet101_for_RFCN_ILSVRCvid_dilation();
  opts.cache_name             = 'rfcn_ILSVRC_ResNet101_OHEM_rpn_dilation';
  conf                        = rfcn_config_ohem('image_means', model.mean_image);
  if with_resNext
    model                       = Model.ResNetX101_for_RFCN_ILSVRCvid_dilation();
    opts.cache_name             = 'rfcn_ILSVRC_ResNeXt101_OHEM_rpn_dilation';
    conf                        = rfcn_config_ohem('image_means', model.mean_image);
  end
end

% check base model existance
check_dl_model(model.net_file);

conf.root_path = root_path ;
fprintf('Loading dataset...')
dataset = [];
dataset = Dataset.ilsvrc15(dataset, 'VIDval16lite3frames', false, root_path);
dataset = Dataset.ilsvrc15(dataset, 'VID+DETtrain16sample', false, root_path);
% do validation after val_interval iters, or not
opts.do_val                 = true; 

%% -------------------- TRAINING --------------------
caffe.reset_all(); 
conf.input_modality = 'rgb';
conf.ims_per_batch = 2 ;
conf.max_size = [1000 ];

conf.sample_vid = false;
conf.use_flipped = true;
opts.cache_name_  = [opts.cache_name '_' conf.input_modality '300RPN_batch4_1e3LR_160kiter' ];

conf.num_proposals = 300;          

% model.net_file = fullfile(root_path, 'output', 'rfcn_cachedir', opts.cache_name_, imdbs_name , ...
%   'iter_140000');

opts.rfcn_model        = rfcn_train(conf, dataset.imdb_train, dataset.roidb_train, ...
                                'offline_roidb',           true, ...
                                'output_dir',     root_path, ...
                                'do_val',           opts.do_val, ...
                                'imdb_val',         dataset.imdb_test, ...
                                'roidb_val',        dataset.roidb_test, ...
                                'solver_def_file',  model.solver_def_file, ...
                                'net_file',         model.net_file, ...
                                'cache_name',       opts.cache_name_, ...
                                'caffe_version',    opts.caffe_version, ...
                                'resume_iter', 0, ...
                                'val_iters', 500, ...
                                'val_interval', 5000, ...
                                'visualize_interval', 0);

%% -------------------- TESTING --------------------
%%
caffe.reset_all(); 
imdbs_name = cell2mat(cellfun(@(x) x.name, dataset.imdb_train, 'UniformOutput', false));
opts.rfcn_model = fullfile(root_path, 'output', 'rfcn_cachedir', opts.cache_name_, imdbs_name , ...
  'iter_140000');
proposal_path = fullfile(root_path, 'Data/VID/val/RPN_proposals/');

conf.root_path = root_path ;
imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16', false, root_path);
% imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16liteStride10', false, root_path);
% imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16lite3frames', false, root_path);
test_classes = [];

conf.num_proposals = 300; 
rfcn_test(conf, imdb_val_lite.imdb_test, imdb_val_lite.roidb_test, ...
      'net_def_file',     model.test_net_def_file, ...
      'net_file',         opts.rfcn_model, ...
      'cache_name',       opts.cache_name_,...
      'ignore_cache',     false, ...
      'proposal_path', proposal_path, ...
      'visualize_every', 0, ...
      'test_classes', test_classes, ...
      'suffix', '');



%% proposal test
recall_per_cls = compute_recall_ilsvrc15(proposal_path, 300, imdb_val_lite.imdb_test);
mean_recall = mean([recall_per_cls.recall]);
fprintf('mean rec:: %.2f\n\n', 100*mean_recall);
