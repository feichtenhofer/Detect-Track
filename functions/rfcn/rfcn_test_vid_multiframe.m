function mAP = rfcn_test_vid_multiframe(conf, imdb, roidb, varargin)
% Video-level testing using final D&T model and RPN proposals
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
    ip.addParamValue('ab_fetch_path',    '',          @isstr);
    ip.addParamValue('visualize_every',   Inf,          @isscalar);
    ip.addParamValue('test_classes',    [] 			);
    ip.addParamValue('chain_boxes_forward',    false, 			@islogical);
    ip.addParamValue('time_stride',   1,          @isscalar);
    ip.addParamValue('thresh_tracks',   0.1,          @isscalar);

    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
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
         load(fullfile(cache_dir, ['tracks_' imdb.name opts.suffix]));
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
        aboxes_track = cell(num_classes, 1);

        for i = 1:num_classes
            aboxes{i} = cell(length(imdb.image_ids), 1);
            box_inds{i} = cell(length(imdb.image_ids), 1);
        end
        ascores_track = aboxes{1};;
        aboxes_track = cell(length(imdb.image_ids), conf.nFramesPerVid);
        aboxes_fwd = aboxes;
        aboxes_bwd = aboxes;
        track_forward_temp = cell(length(imdb.image_ids), 1);
        if isfield(conf,'sample_vid') && conf.sample_vid
            [inds_val,frames_val] = video_generate_random_minibatch([], [], imdb, conf, false);
            frames_val = cellfun(@(x) padarray(x,conf.nFramesPerVid-mod(length(x),conf.nFramesPerVid),'symmetric','post'), frames_val, 'UniformOutput' , false); % sample first frames
        end
        if ~isfield(conf,'nFramesPerVid') 
          conf.nFramesPerVid = 1;
        end
        t_stride = conf.nFramesPerVid;
        t_stride = opts.time_stride ;
        count = 0;
        t_start = tic;
        center_frame = ceil(conf.nFramesPerVid/2);
        for v = 1:numel(inds_val)
          for kk = 1:t_stride:numel(frames_val{v}) - conf.nFramesPerVid + 1
              sub_db_inds = frames_val{v}(kk:1:kk+conf.nFramesPerVid-1);
              d = roidb.rois(sub_db_inds);
              if ~isempty(opts.test_classes),
                testThis = false;
                for ii = 1:length(opts.test_classes)
                  if any(opts.test_classes(ii) == d(1).class), testThis = true; end
                end
                if ~testThis, continue; end
              end
              count = count + 1;
              fprintf('%s: test (%s) vid %d/%d frame %d/%d', procid(), imdb.name, v,numel(inds_val), kk,numel(frames_val{v}));
              th = tic;


              [image_roidb_val, ~, ~] = rfcn_prepare_image_roidb_offline(conf, imdb, roidb.rois(sub_db_inds) , conf.root_path, sub_db_inds);

              im = {}; boxes = {};
              for b = 1:numel(sub_db_inds)
                  im{end+1} = imread(image_roidb_val(b).image_path);
                  boxes{end+1} = image_roidb_val(b).boxes(~image_roidb_val(b).gt,:);
              end

              if opts.chain_boxes_forward && ~isempty(track_forward_temp{sub_db_inds(1)})
                boxes{1} = cat(1, boxes{1}, track_forward_temp{sub_db_inds(1)});
              end


              [boxes_frames, scores_frames, track_boxes] = rfcn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu);
              fprintf(' time %.3fs\n', toc(th)/numel(im)); 

                           
              if ~iscell(track_boxes), track_boxes = {track_boxes}; end;
              max_scores = sum(scores_frames{2}, 2);
              tracklets = (max_scores > opts.thresh_tracks) ;
              aboxes_track{sub_db_inds(2),1} = [aboxes_track{sub_db_inds(2),1}; track_boxes{1}(tracklets,1:4)];
              aboxes_track{sub_db_inds(2),2}  = [aboxes_track{sub_db_inds(2),2}; boxes_frames{2}(tracklets,1:4)];
              aboxes_track{sub_db_inds(2),3}  = [aboxes_track{sub_db_inds(2),3}; track_boxes{2}(tracklets,1:4)];
              ascores_track{sub_db_inds(2)} = [ascores_track{sub_db_inds(2)}; scores_frames{2}(tracklets,:)] ;

              track_forward_temp{sub_db_inds(2)}  =  boxes_frames{2}(tracklets,1:4);

              %% resort frame boxes
              count2 = 0;
              for i=sub_db_inds(:)'
                count2 = count2+1;
                boxes = boxes_frames{count2};  scores = scores_frames{count2}; im_f = im{count2};
                for j = 1:num_classes
                    inds = find(scores(:, j) > thresh(j));
                    if ~isempty(inds)
                        [~, ord] = sort(scores(inds, j), 'descend');
                        ord = ord(1:min(length(ord), max_per_image));
                        inds = inds(ord); inds_tmp = inds;
                        cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                        cls_scores = scores(inds, j);
                        aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                        if ~isempty(box_inds{j}{i}), inds=  inds + numel(box_inds{j}{i}); end
                        box_inds{j}{i} = [box_inds{j}{i}; inds ];
                    else
                        aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                        box_inds{j}{i} = box_inds{j}{i};
                    end
                end


                if mod(count,opts.visualize_every) == 0
                 boxes_cell = cell(length(imdb.classes), 1); boxes_track = boxes_cell; boxes_track2 = boxes_cell;

                  vis_thres = 0.9;
                  for k = 1:length(boxes_cell)
                      boxes_cell{k} = [boxes(:, (1+(k-1)*4):(k*4)), scores(:, k)];
                      I = boxes_cell{k}(:, 5) >= vis_thres;
                      boxes_cell{k} = boxes_cell{k}(I, :);
                    if count2==2
                      I =  ascores_track{i,1}(:, k) >= vis_thres; 
                      boxes_track{k} = [aboxes_track{i,1}(I,1:4), ascores_track{i,1}(I, k)];
                      boxes_track2{k} = [aboxes_track{i,3}(I,1:4), ascores_track{i,1}(I, k)];
                    end
                  end
                  subplot(2,3,count2);
                  showboxes(uint8(im{count2}(:,:,1:3)), boxes_cell, imdb.classes, 'default'); drawnow; 
                  title(sprintf('result for frame: %s', image_roidb_val(count2).image_id)) ;
                  
                  if (count2) == 2 
                      subplot(2,3,4);

                      showboxes(uint8(im{1}(:,:,1:3)), boxes_track, imdb.classes, 'default'); drawnow; 
                      title(sprintf('tracked frame 2 -> 1 : %s', image_roidb_val(count2).image_id)) ;
                      subplot(2,3,5);
                      showboxes(uint8(im{3}(:,:,1:3)), boxes_track2, imdb.classes, 'default'); drawnow; 
                      title(sprintf('tracked frame 2 -> 3 : %s', image_roidb_val(count2).image_id)) ;

                  end
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
        save_file = fullfile(cache_dir, ['tracks_' imdb.name opts.suffix]);
        save(save_file, 'aboxes_track', 'ascores_track');

        
        caffe.reset_all(); 
        rng(prev_rng);

        
        

   %% generate object tracks
    track_boxes = repmat({cell(size(aboxes{1}))},size(aboxes));

    [inds_val,frames_val] = video_generate_random_minibatch([], [], imdb, conf, false);
    paths = cell(numel(inds_val),numel(aboxes));

    for v = 1:numel(inds_val)
      tic_toc_print('generating tracks for video: %d/%d\n', v, length(inds_val));  
      for c = 1:numel(aboxes)
        frameIdx = frames_val{v};

        frameBoxes =  aboxes{c}(frameIdx);
        nonempty_frames = (~cellfun(@isempty,frameBoxes));
                     
        if any(nonempty_frames)
            frameBoxes = frameBoxes(nonempty_frames);
            frameIdx = frameIdx(nonempty_frames);
            paths{v,c} = make_tubes(frameBoxes, 25, true,{aboxes_track(frameIdx,:), ascores_track(frameIdx), c} ); 

            for j=1:numel(paths{v,c})
                for k=1:numel(frameIdx)
                    track_boxes{c}{frameIdx(k)} = [track_boxes{c}{frameIdx(k)}; [paths{v,c}(j).boxes(k,1:4) ...
                     paths{v,c}(j).scores(k) ] ] ;
                end
            end

        end
      end
    end
    
    end

    %%
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
%       % sanity test: 
%       boxes_gt = cell(num_classes,1);
%       classes = {roidb.rois.class}; 
%       boxes = {roidb.rois.boxes}; 
%       boxes = cellfun(@(x) cat(2,x,ones(size(x,1),1)),boxes,'UniformOutput' , false);
%       for c=1:num_classes
%         boxes_gt{c} = cell(num_images,1);
%       end
%       for j=1:num_images
%         if ~isempty(classes{j})
%           for c=unique(classes{j})'
%             boxes_gt{c}{j} = boxes{j}(classes{j}==c,:);
%           end
%         end
%       end
%       res_gt = imdb.eval_func(boxes_gt, imdb, conf, opts.suffix);
    % ilsvrc
    res = imdb.eval_func(aboxes, imdb, conf, opts.suffix);
    res = imdb.eval_func(track_boxes, imdb, conf, opts.suffix, false);
    res = imdb.eval_func(track_boxes, imdb, conf, opts.suffix, false, true);

    for v = 1:numel(inds_val)
        frameIdx = frames_val{v};
        for k=1:numel(frameIdx)
          im = imread(imdb.image_at(frameIdx(k)));
          boxes_cell = cellfun(@(x) x(frameIdx(k)), track_boxes);
          boxes_cell = cellfun(@(x) x(x(:,5) > .2,:), boxes_cell, 'UniformOutput' , false);

          figure(1);
          showboxes(uint8(im(:,:,1:3)), boxes_cell, imdb.classes, 'default'); drawnow; 
          title(sprintf('tube result for frame: %s', imdb.image_ids{frameIdx(k)})) ;
          drawnow;
        end
      end
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