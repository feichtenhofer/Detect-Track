% D&T training using D model and RPN proposals
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe-rfcn';
opts.gpu_id                 =  auto_select_gpu ;
onlyVal = 1 ;

active_caffe_mex(opts.gpu_id, opts.caffe_version);
res50 = 0 ;
modality = 'rgb';
root_path = get_root_path;
with_resNext = 1;

if res50
  model                       = Model.ResNet50_for_DetectTrack_ILSVRCvid();
  opts.cache_name             = 'rfcn_ILSVRC_ResNet50_DetectTrack';
  conf                        = rfcn_config_ohem('image_means', model.mean_image);
else
  model                       = Model.ResNet101_for_DetectTrack_ILSVRCvid();
  opts.cache_name             = 'rfcn_ILSVRC_ResNet101_corr';
  conf                        = rfcn_config_ohem('image_means', model.mean_image);
  if with_resNext
    model                       = Model.ResNeXt101_for_DetectTrack_ILSVRCvid();
    opts.cache_name             = 'rfcn_ILSVRC_ResNeXt101_corr';
    conf                        = rfcn_config_ohem('image_means', model.mean_image);
    conf.max_size = 1000;
  end
end
conf.root_path = root_path ;

% check base model existance
check_dl_model(model.net_file);

dataset = [];
if ~onlyVal
  fprintf('Loading dataset...')
  dataset = Dataset.ilsvrc15(dataset, 'VID+DETtrain16sample', false, root_path);
  dataset.imdb_train{1}     = imdb_from_ilsvrc15vid(root_path, 'vid_train', 0);
  dataset.roidb_train{1}     = dataset.imdb_train{1}.roidb_func(dataset.imdb_train{1}, ...
            'roidb_name_suffix', '', 'rootDir', root_path);
  dataset = Dataset.ilsvrc15(dataset, 'VIDval16', false, root_path);
  imdbs_name = cell2mat(cellfun(@(x) x.name, dataset.imdb_train, 'UniformOutput', false));
else 
  imdbs_name ='ilsvrc15_vid_trainilsvrc15_train';
end     
% do validation after val_interval iters, or not
opts.do_val                 = true; 

%% -------------------- TRAINING --------------------
caffe.reset_all(); 
conf.input_modality = modality;
conf.ims_per_batch = 4 ;
conf.sample_vid = true; conf.nFramesPerVid = 2; conf.ims_per_batch = 1 ;
conf.input_modality = 'rgb';
conf.time_stride = 1:1 ;
conf.use_flipped = true;
opts.cache_name_  = [opts.cache_name '_' conf.input_modality 'tstride_' num2str(conf.time_stride(end)) 'track_reg_lowerLR_batch8'];

% opts.cache_name_  = [opts.cache_name '_' conf.input_modality 'tstride_' num2str(conf.time_stride(end)) 'track_reg_lowerLR_maxSz1000'];
% opts.cache_name_  = [opts.cache_name '_' conf.input_modality 'tstride_' num2str(conf.time_stride(end)) 'track_reg_lowerLR_lossw1'];

conf.num_proposals = 300; 
conf.regressTracks = true; conf.regressAllTracks = false;

% model.net_file = fullfile(root_path, 'output', 'rfcn_cachedir', opts.cache_name_, imdbs_name , ...
%   'iter_80000');
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
                                'val_iters', 1500, ...
                                'val_interval', 5000, ...
                                'visualize_interval', 0);
%% -------------------- TESTING --------------------
%%
% chose test prototxt \in {test_track_bidir.prototxt, test_track_reg.prototxt, 
% test_track_regcls.prototxt,% test_track_regcls_poolreg.prototxt}
model.test_net_def_file = strrep(model.test_net_def_file,'test_track.prototxt','test_track_reg.prototxt');


conf.sample_vid = true; conf.nFramesPerVid = 2; conf.ims_per_batch = 1 ; conf.regressTracks = true;
conf.bidirTrack = false;
opts.rfcn_model = fullfile(root_path, 'output', 'rfcn_cachedir', opts.cache_name_, imdbs_name , ...
  'final');

proposal_path = fullfile(root_path, 'Data/VID/val/RPN_proposals/');

imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16', false, root_path);
% imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16liteStride10', false, root_path);
% imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16liteStride5', false, root_path);
% imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16lite3frames', false, root_path);

%% -------------------- TESTING --------------------
caffe.reset_all(); 
rfcn_test_vid(conf, imdb_val_lite.imdb_test, imdb_val_lite.roidb_test, ...
                  'net_def_file',     model.test_net_def_file, ...
                  'net_file',         opts.rfcn_model, ...
                  'cache_name',       opts.cache_name_,...
                  'ignore_cache',     true, ...
                  'proposal_path', proposal_path, ...
                  'visualize_every', 0 , ...
                  'write_vid', 0 , ...       
                  'test_classes', [], ...
                  'suffix', '', ...
                  'chain_boxes_forward', false, ...
                  'time_stride', 1);

%% -------------------- MULTI-FRAME TESTING --------------------
caffe.reset_all(); 
conf.sample_vid = true; conf.nFramesPerVid = 3; conf.ims_per_batch = 1 ; conf.regressTracks = true;
conf.bidirTrack = false;
model                       = Model.ResNet101_for_RFCN_ILSVRCvid_corr_fusion();
model.test_net_def_file = strrep(model.test_net_def_file,'test_track.prototxt','test_track_3frames_reg.prototxt');
rfcn_test_vid_multiframe(conf, imdb_val_lite.imdb_test, imdb_val_lite.roidb_test, ...
                  'net_def_file',     model.test_net_def_file, ...
                  'net_file',         opts.rfcn_model, ...
                  'cache_name',       opts.cache_name_,...
                  'ignore_cache',     true, ...
                  'proposal_path', proposal_path, ...
                  'visualize_every', 0, ...
                  'test_classes', test_classes, ...
                  'suffix', '', ...
                  'chain_boxes_forward', true, ...
                  'time_stride', 1, ...
                  'thresh_tracks', 0.01);
