function mAP = rfcn_test(conf, imdb, roidb, varargin)
% Image-level testing using final D model and RPN proposals
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addRequired('roidb',                             @isstruct);
    ip.addParamValue('net_def_file',    '', 			@isstr);
    ip.addParamValue('net_file',        '', 			@isstr);
    ip.addParamValue('cache_name',      '', 			@isstr);
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    ip.addParamValue('proposal_path',    '',          @isstr);
    ip.addParamValue('visualize_every',   Inf,          @isscalar);
    ip.addParamValue('test_classes',    [] 			);

    
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    num_proposals = 300;
    if isfield(conf, 'num_proposals')
      num_proposals = conf.num_proposals;
    end
    visualize = false;   

%%  set cache dir
    cache_dir = fullfile(conf.root_path, 'output', 'rfcn_cachedir', opts.cache_name, imdb.name);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);

    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    
    
    try
      aboxes = cell(num_classes, 1);
      if opts.ignore_cache
          throw('');
      end
      for i = 1:num_classes
        load(fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]));
        aboxes{i} = boxes;
      end
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        max_rois_num_in_gpu = 10000;

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        %heuristic: keep an average of 160 detections per class per images prior to NMS
        max_per_set = 160 * num_images;
        % heuristic: keep at most 400 detection per class per image prior to NMS
        max_per_image = 400;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[cls][image] = N x 5 array of detections in
        %    (x1, y1, x2, y2, score)
        aboxes = cell(num_classes, 1);
        box_inds = cell(num_classes, 1);
        for i = 1:num_classes
            aboxes{i} = cell(length(imdb.image_ids), 1);
            box_inds{i} = cell(length(imdb.image_ids), 1);
        end
        count = 0;
        t_start = tic;
        for i = 1:1:num_images
            d = roidb.rois(i);
            if ~isempty(opts.test_classes),
              testThis = false;
              for ii = 1:length(opts.test_classes)
                if any(opts.test_classes(ii) == d.class), testThis = true; end
              end
              if ~testThis, continue; end
            end
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            th = tic;
            
            if ~isempty(opts.proposal_path)
              file = imdb.image_ids{i};
              file(file == '\') = '/';
              ind = find(file == '/', 1, 'first');
              videoFrame = file ;
              if strcmp(file(1:20), 'ILSVRC2015_VID_train')
                videoFrame = file(ind+1:end);
              end
              ab_fetch_file = [opts.proposal_path '/' videoFrame '.mat'];
              if ~exist(ab_fetch_file,'file')
                  fprintf('In roidb_from_proposal_manual.m --> error:file does not exist : \n %s \n', ab_fetch_file);
                  fprintf('generate this file and run script again, pausing for now...\n');
                  pause;
              end

              abst = load(ab_fetch_file);
              if isfield(abst, 'boxes')
                proposals = abst.boxes;
              else
                proposals = abst.aboxes;
              end
              
              if num_proposals < 1
                  proposals = proposals(proposals(:,5) > num_proposals,1:4);
              else
                  proposals = proposals(1:min(num_proposals,size(proposals,1)),1:4);
              end
              if isempty(proposals), continue; end
              
              d.boxes = cat(1, d.boxes, proposals);
              d.gt = cat(1, d.gt, zeros(size(proposals,1),1));
            end
            
            im_path = imdb.image_at(i);
            if isfield(conf, 'input_modality') && ~strcmp(conf.input_modality, 'rgb')
            
              img_file_u = strrep(im_path, ['Data' filesep 'VID'], ...
                ['Data' filesep 'VID' filesep 'tvl1_flow_600' filesep 'u'] ) ; 
              img_file_v = strrep(im_path, ['Data' filesep 'VID'], ...
                ['Data' filesep 'VID' filesep 'tvl1_flow_600' filesep 'v'] ) ; 
              try
                 nFrames = 1;
                 if isfield(conf, 'nFrames')
                   nFrames = conf.nFrames;
                 end
                [path_u, ~, ~] = fileparts(img_file_u);
                [path_v, frame, ext] = fileparts(img_file_v);
                frames = str2num(frame)-nFrames+1:str2num(frame)+nFrames-1;
                vid = fileparts(videoFrame);
                frames(frames <0) = 0; frames(frames > imdb.nFrames(vid)-2) =  imdb.nFrames(vid)-2;
                im_u = []; im_v = [];
                for frame=frames
                  img_file_u = fullfile(path_u, [sprintf('%06d',frame), ext]) ;
                  img_file_v = fullfile(path_v, [sprintf('%06d',frame), ext]) ;
                  im_u{end+1} = imread(img_file_u); im_v{end+1} = imread(img_file_v); 
                end
                im_u = cat(3,im_u{:});im_v = cat(3,im_v{:});

              catch % the last frame is simply replicated in optical flow case
                continue;
              end
                switch conf.input_modality
                  case 'flow'
                    flow = single(cat(3,im_u,im_v)) - 128;
                    flow = bsxfun(@minus,flow, median(median(flow,1),2)) ; 
                    mag_flow = sqrt(sum(flow.^2,3)) - 128; 
                    im = (cat(3,flow,mag_flow));
                    im = imresize( im, imdb.sizes(i,:));
                  case 'grayflow'
                    imgray = imresize( rgb2gray(imread(im_path)), sz(1:2));
                    im = cat(3,im_u,im_v,imgray);
                  case 'rgbflow'
 
                    if isfield(conf, 'subflow_mean') && ~conf.subflow_mean
                      flow = single(cat(3,im_u,im_v));
                    else
                      flow = single(cat(3,im_u,im_v)) - 128;
                    end

                    imrgb = single(imread(im_path));

                    if size(imrgb,3) < 3
                      imrgb = imrgb(:,:,[1 1 1]);
                    end
                    sz = size(imrgb);
                    im = cat(3,imrgb,imresize( flow, sz(1:2)));

      
                end
              else
                  im = imread(im_path);
              end

            
            
            
            [boxes, scores] = rfcn_im_detect(conf, caffe_net, im, d.boxes(d.gt==0, :), max_rois_num_in_gpu);

            for j = 1:num_classes
                inds = find(scores(:, j) > thresh(j));
                if ~isempty(inds)
                    [~, ord] = sort(scores(inds, j), 'descend');
                    ord = ord(1:min(length(ord), max_per_image));
                    inds = inds(ord);
                    cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                    cls_scores = scores(inds, j);
                    aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                    box_inds{j}{i} = [box_inds{j}{i}; inds];
                else
                    aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                    box_inds{j}{i} = box_inds{j}{i};
                end
            end
            
            fprintf(' time %.3fs\n', toc(th)); 
            
            if mod(count,opts.visualize_every) == 0
             boxes_cell = cell(length(imdb.classes), 1);

              vis_thres = 0.3;
              for k = 1:length(boxes_cell)
                  boxes_cell{k} = [boxes(:, (1+(k-1)*4):(k*4)), scores(:, k)];


                  boxes_cell{k} = boxes_cell{k}(nms(boxes_cell{k}, 0.3), :);

                  I = boxes_cell{k}(:, 5) >= vis_thres;
                  boxes_cell{k} = boxes_cell{k}(I, :);
              end
              figure(1);
              showboxes(uint8(im(:,:,1:3)), boxes_cell, imdb.classes, 'default'); drawnow; 
              if size(im,3) > 3
                  figure(2); 
                  showboxes(uint8(im(:,:,4:6)+128), boxes_cell, imdb.classes, 'default'); drawnow; 
              end

            end

            if mod(count, 1000) == 0
                for j = 1:num_classes
                [aboxes{j}, box_inds{j}, thresh(j)] = ...
                    keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
                end
                disp(thresh);
            end    
        end

        for j = 1:num_classes
            [aboxes{j}, box_inds{j}, thresh(j)] = ...
                keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
        end
        disp(thresh);

        for i = 1:num_classes
            top_scores{i} = sort(top_scores{i}, 'descend');  
            if (length(top_scores{i}) > max_per_set)
                thresh(i) = top_scores{i}(max_per_set);
            end

            % go back through and prune out detections below the found threshold
            for j = 1:length(imdb.image_ids)
                if ~isempty(aboxes{i}{j})
                    I = find(aboxes{i}{j}(:,end) < thresh(i));
                    aboxes{i}{j}(I,:) = [];
                    box_inds{i}{j}(I,:) = [];
                end
            end

            save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
            boxes = aboxes{i};
            inds = box_inds{i};
            save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end

    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------

    if isequal(imdb.eval_func, @imdb_eval_voc)
        new_parpool();
        parfor model_ind = 1:num_classes
          cls = imdb.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
        end
    else
      if 0
        % sanity test: 
        boxes_gt = cell(num_classes,1);
        classes = {roidb.rois.class}; 
        boxes = {roidb.rois.boxes}; 
        boxes = cellfun(@(x) cat(2,x,ones(size(x,1),1)),boxes,'UniformOutput' , false);
        for c=1:num_classes
          boxes_gt{c} = cell(num_images,1);
        end
        for j=1:num_images
          if ~isempty(classes{j})
            for c=unique(classes{j})'
              boxes_gt{c}{j} = boxes{j}(classes{j}==c,:);
            end
          end
        end
        res_gt = imdb.eval_func(boxes_gt, imdb, conf, opts.suffix);
      end
      % ilsvrc
      res = imdb.eval_func(aboxes, imdb, conf, opts.suffix);
    end

    if ~isempty(res)
        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('Results:\n');
        aps = [res(:).ap]' * 100;
        disp(aps);
        disp(mean(aps));
        fprintf('~~~~~~~~~~~~~~~~~~~~\n');
        mAP = mean(aps);
    else
        mAP = nan;
    end
    
    diary off;
end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end