function save_model_path = rfcn_train(conf, imdb_train, roidb_train, varargin)
% save_model_path = rfcn_train(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% R-FCN implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2016, Jifeng Dai
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb_train',                        @iscell);
    ip.addRequired('roidb_train',                       @iscell);
    ip.addParamValue('do_val',          false,          @isscalar);
    ip.addParamValue('imdb_val',        struct(),       @isstruct);
    ip.addParamValue('roidb_val',       struct(),       @isstruct);
    ip.addParamValue('val_iters',       500,            @isscalar); 
    ip.addParamValue('val_interval',    5000,           @isscalar); 
    ip.addParamValue('snapshot_interval',...
                                        10000,          @isscalar);
    ip.addParamValue('solver_def_file', fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_res3a', 'solver_80k120k_lr1_3.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',      'ResNet-50L_res3a', ...
                                                        @isstr);
    ip.addParamValue('caffe_version',   'Unkonwn',      @isstr);
    
    ip.addParamValue('offline_roidb',          false,          @isscalar);
    ip.addParamValue('output_dir',                       @isstr);
    ip.addParameter('resume_iter',          0,             @isscalar);
    ip.addParameter('visualize_interval',          0,             @isscalar);
    ip.addParameter('net_file',    '' );

    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    
%% try to find trained model
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(opts.output_dir, 'output', 'rfcn_cachedir', opts.cache_name, imdbs_name); 
    save_model_path = fullfile(cache_dir, 'final');
    if exist(save_model_path, 'file')
        return;
    end
    
%% init
    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);
    
    % init caffe solver
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(opts.solver_def_file);
    
    if (opts.resume_iter == 0)
        if iscell(opts.net_file)
          arrayfun(@(i) caffe_solver.net.copy_from(opts.net_file{i}), 1:numel(opts.net_file));
        else
          caffe_solver.net.copy_from(opts.net_file);
        end
    else
        % loading solverstate, resume
        caffe_solver.net.copy_from(fullfile(cache_dir, ...
            sprintf('iter_%d', opts.resume_iter )));
    end
    if isfield(conf, 'nFrames') && strcmp(conf.input_modality, 'rgbflow')
      weights_conv1 = caffe_solver.net.layer_vec(caffe_solver.net.name2layer_index('conv1')).params(1).get_data();
      conv1_flow  = caffe_solver.net.layer_vec(caffe_solver.net.name2layer_index('conv1_flow')).params(1).get_data();
      diff =   10 - size(weights_conv1,3) ;
      conv1_flow = padarray(weights_conv1, [0 0 diff 0], 'symmetric', 'post');
      caffe_solver.net.layer_vec(caffe_solver.net.name2layer_index('conv1_flow')).params(1).set_data(conv1_flow);
    end
       
    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);

    % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    
    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);
    
%% making tran/val data
    if opts.offline_roidb
      load('bbox_moments.mat')
      bbox_means(:,1) = bbox_means(:,1) * 0; % set horizontal mean to zero for online target flipping
    else
      fprintf('Preparing training data...');
      [image_roidb_train, bbox_means, bbox_stds] = rfcn_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train);
      fprintf('Done.\n');
    end
      if opts.do_val
      % linear sample val vids
         inds_val =  1:length(opts.imdb_val.image_ids);
         inds_val = inds_val(:,floor(linspace(1,size(inds_val,2), opts.val_iters*conf.ims_per_batch)));

         inds_val = reshape(inds_val , conf.ims_per_batch, []);
         inds_val = num2cell(inds_val, 1);
         shuffled_inds_val = inds_val;
        if isfield(conf,'sample_vid') && conf.sample_vid
            [shuffled_inds_val,shuffled_frames_val, ~] = video_generate_random_minibatch([], [], opts.imdb_val, conf);
            shuffled_frames_val = cellfun(@(x) x(1:conf.nFramesPerVid), shuffled_frames_val, 'UniformOutput' , false); % sample first frames
        end
      end
%% training
    shuffled_inds = []; shuffled_vids  = []; shuffled_frames = [];
    train_results = [];  shuffled_inds2 = []; 
    val_results = [];  
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();

    p = new_parpool(2);
    parfor i=1:2
        seed_rand(conf.rng_seed);
    end
    if isfield(conf,'sample_vid') && conf.sample_vid
      [shuffled_vids,shuffled_frames, sub_db_inds] = video_generate_random_minibatch(shuffled_vids, shuffled_frames, imdb_train{1}, conf);
    else
      [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, imdb_train{1}, conf.ims_per_batch);
    end
    if numel(imdb_train) > 1
      [shuffled_inds2, ~] = generate_random_minibatch(shuffled_inds2, imdb_train{2}, conf.ims_per_batch);
    end
    [image_roidb_train, ~, ~] = rfcn_prepare_image_roidb_offline(conf, opts.imdb_train{1}, roidb_train{1}.rois(sub_db_inds) , opts.output_dir, sub_db_inds, bbox_means, bbox_stds);
    parHandle = parfeval(p, @rfcn_get_minibatch, 1, conf, image_roidb_train);
    tic

    while (iter_ < max_iter)
      caffe_solver.net.set_phase('train');
      % generate minibatch training data
      [~, net_inputs] = fetchNext(parHandle);
      %  [net_inputs] = feval(@rfcn_get_minibatch, conf, image_roidb_train);

      if iscell(net_inputs) && min([size(net_inputs{1},1) size(net_inputs{1},2)]) > max(conf.scales)
       fprintf('Problem with getbatch...pausing\n');
       pause;
      end
      % generate minibatch training data
      if numel(imdb_train) == 1 || ~mod(iter_,2) % sample from db1 
      if isfield(conf,'sample_vid') && conf.sample_vid
        [shuffled_vids,shuffled_frames, sub_db_inds] = video_generate_random_minibatch(shuffled_vids, shuffled_frames, imdb_train{1}, conf);
      else
        [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, conf.ims_per_batch);
      end        
        [image_roidb_train, ~, ~] = rfcn_prepare_image_roidb_offline(conf, opts.imdb_train{1}, roidb_train{1}.rois(sub_db_inds) , opts.output_dir, sub_db_inds, bbox_means, bbox_stds);
      else
        [shuffled_inds2, sub_db_inds2] = generate_random_minibatch(shuffled_inds2, image_roidb_train, conf.ims_per_batch);
        [image_roidb_train, ~, ~] = rfcn_prepare_image_roidb_offline(conf, opts.imdb_train{2}, roidb_train{2}.rois(sub_db_inds2) , opts.output_dir, sub_db_inds2, bbox_means, bbox_stds);
      end

      parHandle = parfeval(p, @rfcn_get_minibatch, 1, conf, image_roidb_train);

      if  ~iscell(net_inputs) %|| isempty(image_roidb_train)
        continue;
      end

      if opts.visualize_interval >0 && ~mod(iter_, opts.visualize_interval) 
        vis_inputs;
      end
      caffe_solver.net.reshape_as_input(net_inputs);

      % one iter SGD update
      caffe_solver.net.set_input_data(net_inputs);
      caffe_solver.step(1);
      rst = caffe_solver.net.get_output();
      train_results = parse_rst(train_results, rst);

      % do valdiation per val_interval iterations
      if ~mod(iter_, opts.val_interval)  && iter_ >0 
        if opts.do_val
            caffe_solver.net.set_phase('test');   
            % use all regoins for testing 
            bt_thresh_lo = conf.bg_thresh_lo; conf.bg_thresh_lo = 0; 
            batch_size = conf.batch_size; conf.batch_size = -1;
            use_flipped = conf.use_flipped; conf.use_flipped = false;
            for i = 1:length(shuffled_inds_val)
                sub_db_inds = shuffled_inds_val{i};
                 if isfield(conf,'sample_vid') && conf.sample_vid
                   [shuffled_inds_val, shuffled_frames_val, sub_db_inds] = video_generate_random_minibatch(shuffled_inds_val, shuffled_frames_val, opts.imdb_val, conf);
                 end

                [image_roidb_val, ~, ~] = rfcn_prepare_image_roidb_offline(conf, opts.imdb_val, opts.roidb_val.rois(sub_db_inds) , opts.output_dir, sub_db_inds, bbox_means, bbox_stds);
                if isempty(image_roidb_val)
                  continue;
                end

                ind = arrayfun(@(x) sum(x.overlap,2) == 1, image_roidb_val, 'UniformOutput', false);
                for b = 1:numel(sub_db_inds)
                  for f = {'overlap','class','bbox_targets',}
                    image_roidb_val(b).(char(f)) = image_roidb_val(b).(char(f))(ind{b},:) ;
                  end
                end

                net_inputs = rfcn_get_minibatch(conf, image_roidb_val);
                if ~iscell(net_inputs)
                  continue;
                end
                caffe_solver.net.reshape_as_input(net_inputs);

                caffe_solver.net.forward(net_inputs);
                rst = caffe_solver.net.get_output();
                val_results = parse_rst(val_results, rst);

            end
            % use all regoins for testing 
            conf.bg_thresh_lo = bt_thresh_lo; 
            conf.batch_size = batch_size;
            conf.use_flipped = use_flipped ;
        end

        show_state(iter_, train_results, val_results);
        toc;tic;
        train_results = [];
        val_results = [];
        diary; diary; % flush diary
      end

      % snapshot
      if ~mod(iter_, opts.snapshot_interval) && iter_ >0 
        snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
      end

      iter_ = caffe_solver.iter();
    
    end
   
    % final snapshot
    snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    save_model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        
        if isfield(image_roidb_train, 'sizes')
          hori_image_inds = image_roidb_train.sizes(:,2) > image_roidb_train.sizes(:,1);
        else
          hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
        end
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);
        vert_image_inds = find(vert_image_inds);
        
        % random perm
        lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
        hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
        lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
        vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
        
        % combine sample for each ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
        vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
        
        shuffled_inds = [hori_image_inds, vert_image_inds];
        shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
        shuffled_inds = repmat(shuffled_inds, 1, 10);
        shuffled_inds = num2cell(shuffled_inds, 1);
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
    bbox_pred_layer_name = 'rfcn_bbox';
    idx = find(~cellfun(@isempty, strfind(caffe_solver.net.layer_names, bbox_pred_layer_name)));
    bbox_pred_layer_name = caffe_solver.net.layer_names{idx};
    weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
    biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
    weights_back = weights;
    biase_back = biase;
 
    rep_time = size(weights, 4)/length(bbox_means(:));
    
    bbox_stds_flatten = bbox_stds';
    bbox_stds_flatten = bbox_stds_flatten(:);
    bbox_stds_flatten = repmat(bbox_stds_flatten, [1,rep_time])';
    bbox_stds_flatten = bbox_stds_flatten(:);
    bbox_stds_flatten = permute(bbox_stds_flatten, [4,3,2,1]);
    
    bbox_means_flatten = bbox_means';
    bbox_means_flatten = bbox_means_flatten(:);
    bbox_means_flatten = repmat(bbox_means_flatten, [1,rep_time])';
    bbox_means_flatten = bbox_means_flatten(:);
    bbox_means_flatten = permute(bbox_means_flatten, [4,3,2,1]);
    
    % merge bbox_means, bbox_stds into the model
    weights = bsxfun(@times, weights, bbox_stds_flatten); % weights = weights * stds; 
    biase = biase .* bbox_stds_flatten(:) + bbox_means_flatten(:); % bias = bias * stds + means;
    
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

    model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);
    
    % restore net to original state
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end

function show_state(iter, train_results, val_results)
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    fprintf('Training : accuracy %.3g, loss (cls %.3g, reg %.3g)\n', ...
        mean(train_results.accuarcy.data), ...
        mean(train_results.loss_cls.data), ...
        mean(train_results.loss_bbox.data));
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : accuracy %.3g, loss (cls %.3g, reg %.3g)\n', ...
            mean(val_results.accuarcy.data), ...
            mean(val_results.loss_cls.data), ...
            mean(val_results.loss_bbox.data));
    end
end
