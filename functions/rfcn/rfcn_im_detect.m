function [pred_boxes, scores, track_boxes] = rfcn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% [pred_boxes, scores] = rfcn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% --------------------------------------------------------
% R-FCN implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2016, Jifeng Dai
% Licensed under The MIT License [see LICENSE for details]
%     im_blob, rois_blob = [];

track_boxes = [];
if isfield(conf, 'bidirTrack') && conf.bidirTrack, bidirTrack = true; else, bidirTrack = false; end
    if iscell(im)
      [im_blob, rois_blob, s] = cellfun(@(x,y) get_blobs(conf, x, y),  im, boxes, 'uniform', 0);
      tmp = rois_blob{1}(:,2:end);
      for ind = 1:numel(im_blob)
        rois_blob{ind}(:,1) = rois_blob{ind}(:,1) * ind;
      end
      if isfield(conf, 'regressTracks') && conf.regressTracks && ~bidirTrack % && conf.nFramesPerVid < 3
        centre = ceil(numel(im_blob)/2);
        tracks_blob = []; boxes_track = repmat(boxes{centre},centre,1);
        center_blobs = rois_blob{centre};
        for tr=1:centre
         	center_blobs(:,1) = ones(size(center_blobs(:,1))) * tr;
          tracks_blob = cat(1,tracks_blob,center_blobs);
        end
      end
      im_blob = cat(4, im_blob{:});       rois_blob = cat(1, rois_blob{:});
      boxes=cat(1,boxes{:});
    else
      [im_blob, rois_blob, ~] = get_blobs(conf, im, boxes);
      [~, index, inv_index] = unique(rois_blob, 'rows');
      rois_blob = rois_blob(index, :);
      boxes = boxes(index, :);
    end
    % When mapping from image ROIs to feature map ROIs, there's some aliasing
    % (some distinct image ROIs get mapped to the same feature ROI).
    % Here, we identify duplicate feature ROIs, so we only compute features
    % on the unique subset.

    % permute data into caffe c++ memory, thus [num, channels, height, width]
     im_blob(:, :, 1:3, :) = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    if size(im_blob,3) > 3
      im_blob(:, :, 4:6, :) = im_blob(:, :, [6, 5, 4], :); % from rgb to brg
    end 
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob);
    
    total_rois = size(rois_blob, 4);
    total_scores = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    total_box_deltas = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
        
        net_inputs = {im_blob, sub_rois_blob};
        if exist('tracks_blob', 'var')
          tracks_blob = single(permute(tracks_blob-1, [3, 4, 2, 1]));
          net_inputs{end+1} = tracks_blob;
        end
        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        caffe_net.forward(net_inputs);

        if conf.test_binary
            % simulate binary logistic regression
            scores = caffe_net.blobs('cls_score').get_data();
            scores = squeeze(scores)';
            % Return scores as fg - bg
            scores = bsxfun(@minus, scores, scores(:, 1));
        else
            % use softmax estimated probabilities
            scores = caffe_net.blobs('cls_prob').get_data();
            scores = squeeze(scores)';
            if any(strcmp(caffe_net.outputs, 'cls_prob_2frame'))
              scores = 0.5 * scores + .5 * squeeze(caffe_net.blobs('cls_prob_2frame').get_data())';
            end
            if any(strcmp(caffe_net.outputs, 'cls_prob_disp'))
              scores_disp = squeeze(caffe_net.blobs('cls_prob_disp').get_data())';
            end
            if any(strcmp(caffe_net.outputs, 'cls_prob_disp1'))
              scores_disp = cat(1,squeeze(caffe_net.blobs('cls_prob_disp1').get_data())', ...
                squeeze(caffe_net.blobs('cls_prob_disp3').get_data())' );
            end
        end


        % Apply bounding-box regression deltas
        box_deltas = caffe_net.blobs('bbox_pred').get_data();
        box_deltas = squeeze(box_deltas)';
        
        total_scores{i} = scores;
        total_box_deltas{i} = box_deltas;
    end 
    
    scores = cell2mat(total_scores);
    box_deltas = cell2mat(total_box_deltas);
    
    pred_boxes = rfcn_bbox_transform_inv(boxes, box_deltas);
    
    if exist('inv_index', 'var');
      % Map scores and predictions back to the original set of boxes
      scores = scores(inv_index, :);
      pred_boxes = pred_boxes(inv_index, :);
    end
    
   if bidirTrack
        track_deltas = squeeze(caffe_net.blobs('bbox_disp').get_data())';
        track_boxes = rfcn_bbox_transform_inv(boxes, track_deltas);
        track_boxes = track_boxes(:, 5:end);
    end
    % remove scores and boxes for back-ground
    pred_boxes = pred_boxes(:, 5:end);
    scores = scores(:, 2:end);
    if iscell(im)
      scrs = {};    pred_bx = {}; bidirTracks = {};
        for ind = 1:numel(im)
          inds=squeeze(rois_blob(1,1,1,:)) == ind-1;
          pred_bx{ind} = pred_boxes(inds,:) ;
          scrs{ind} = scores(inds,:) ;
          pred_bx{ind} = clip_boxes(pred_bx{ind}, size(im{ind}, 2), size(im{ind}, 1));

          if conf.bbox_class_agnostic
            pred_bx{ind} = repmat(pred_bx{ind}, [1, size(scores,2)]);
          end
          if bidirTrack
            bidirTracks{ind} = clip_boxes(track_boxes(inds,:), size(im{ind}, 2), size(im{ind}, 1));
          end
          if ind==1 && exist('scores_disp', 'var') && numel(im) == 2
            scrs{ind} = ( scrs{ind} + scores_disp(:, 2:end) ) / 2;
          elseif ind==2 && exist('scores_disp', 'var') && numel(im) == 3
            scrs{ind} = ( scrs{ind} + ...
              scores_disp(1:size(scrs{ind},1), 2:end) + ...
              scores_disp(size(scrs{ind},1)+1:end, 2:end) ) / 3;

          end
        end
        pred_boxes = pred_bx;
        scores = scrs;
        track_boxes = bidirTracks;
    else
      pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
      if conf.bbox_class_agnostic
          pred_boxes = repmat(pred_boxes, [1, size(scores,2)]);
      end
    end
            
    if isfield(conf, 'regressTracks') && conf.regressTracks && ~bidirTrack
      if any(~cellfun(@isempty,strfind(caffe_net.layer_names,'box_trans')))
        if numel(pred_boxes) == 2 % track fwd only
          track_deltas = squeeze(caffe_net.blobs('bbox_disp').get_data())';
          track_boxes = rfcn_bbox_transform_inv(pred_boxes{1}(:,1:8), track_deltas);
        elseif numel(pred_boxes) == 3 % track fwd and bwd
          track_deltas1 = squeeze(caffe_net.blobs('bbox_disp1').get_data())';
          track_deltas3 = squeeze(caffe_net.blobs('bbox_disp3').get_data())';
          num_ctr = size(pred_boxes{2},1);
          track_boxes1 =  rfcn_bbox_transform_inv(pred_boxes{2}(:,1:8), track_deltas1(:,5:end));
          track_boxes3 =  rfcn_bbox_transform_inv(pred_boxes{2}(:,1:8), track_deltas3(:,5:end));
          track_boxes = {clip_boxes(track_boxes1, size(im{1}, 2), size(im{1}, 1)), clip_boxes(track_boxes3, size(im{1}, 2), size(im{1}, 1)) } ; 
          return;
        end
      else
        track_deltas = squeeze(caffe_net.blobs('bbox_disp').get_data())';
        track_boxes = rfcn_bbox_transform_inv(boxes_track, track_deltas);
      end
      track_boxes = track_boxes(:, 5:end);
      track_boxes = clip_boxes(track_boxes, size(im{1}, 2), size(im{1}, 1));

      boxes_out = {};
      for ind=1:centre
        inds=squeeze(tracks_blob(1,1,1,:)) == ind-1;
        boxes_out{ind} = track_boxes(inds,:);
      end
      track_boxes = boxes_out;
    end
end

function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    [data_blob, im_scale_factors] = get_image_blob(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function [blob, im_scales] = get_image_blob(conf, im)
    [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    if isfield(conf,'image_std')
      ims = cellfun(@(x)  bsxfun(@rdivide, x, conf.image_std),ims,'UniformOutput',false);
    end 
    im_scales = cell2mat(im_scales);
    blob = im_list_to_blob(ims);    
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        [~, levels] = min(abs(scaled_areas - 224.^2), [], 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end