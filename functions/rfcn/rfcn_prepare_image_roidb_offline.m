% -------------------------------------------------------------------
function [image_roidb, bbox_means, bbox_stds] = rfcn_prepare_image_roidb_offline(conf, imdb, rois, output_dir, sub_db_inds, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = rfcn_prepare_image_roidb(conf, imdbs, roidbs, cache_img)
%   Load proposals and image information information from imdb to 
%   create roidb and normalize mean (bbox_means) and std (bbox_stds). 
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
    
if ~exist('bbox_means', 'var')
    bbox_means = [];
    bbox_stds = [];
end

visualize = false; % if you like to see the proposal boxes
num_proposals = 300;
if isfield(conf, 'num_proposals')
  num_proposals = conf.num_proposals;
end
if strcmp(imdb.name(1:12), 'ilsvrc15_vid')
  proposal_path = fullfile(output_dir, 'Data/VID/proposals/RPN_proposals');
else
  proposal_path = fullfile(output_dir, 'Data/DET/proposals/RPN_proposals');
end

image_roidb = cell(length(sub_db_inds),1);
ii = 0;
for i=sub_db_inds(:)'
    ii=ii+1;

    ts = tic;
    file = imdb.image_ids{i};

    file(file == '\') = '/';
    ind = find(file == '/', 1, 'first');
    videoFrame = file ;
    if strcmp(file(1:14), 'ILSVRC2015_val')
      videoFrame = ['val/' videoFrame];
      valMode = true;
    else
      valMode = false;
    end
    if strcmp(videoFrame(end-4:end), '_flip')
      videoFrame = videoFrame(1:end-5); flip_proposal = true;
    else
      flip_proposal = false;
    end
    proposal_file = [proposal_path '/' videoFrame '.mat'];

    if ~exist(proposal_file,'file')
      fprintf('error: cant find file %s \n ', proposal_file);
      image_roidb = [];
      return;
    end

    try 
      abst = load(proposal_file);
    catch
      fprintf('error: cant load file %s \n ', proposal_file);
      pause;
    end
    try
      boxes = abst.boxes;
    catch
      boxes = abst.aboxes;
    end

    if isempty(boxes),   image_roidb = []; return; end;
    if num_proposals < 1  
        boxes = boxes(boxes(:,5) > num_proposals,:);  
    else
        boxes = boxes(1:min(num_proposals,size(boxes,1)),:); 
    end
    if isempty(boxes),   image_roidb = []; return; end;

    scores = boxes(:,5);
    boxes = boxes(:,1:4); 

    if flip_proposal
      boxes(:, [1, 3]) = imdb.sizes(i, 2) + 1 - boxes(:, [3, 1]);
    end
    is_gt = rois(ii).gt;
    gt_boxes = rois(ii).boxes(is_gt, :);
    gt_classes = rois(ii).class(is_gt, :);
    all_boxes = cat(1, rois(ii).boxes, boxes);

    num_gt_boxes = size(gt_boxes, 1);
    num_boxes = size(boxes, 1);

    overlap = cat(1, rois(ii).overlap, zeros(num_boxes, size(rois(ii).overlap, 2)));
    class = cat(1, rois(ii).class, zeros(num_boxes, 1));
    for j = 1:num_gt_boxes
        overlap(:, gt_classes(j)) = ...
            max(full(overlap(:, gt_classes(j))), boxoverlap(all_boxes, gt_boxes(j, :))); % function boxoverlap() is under utils/
    end

    image_roidb{1}(ii,1).image_path = imdb.image_at(i);
    image_roidb{1}(ii,1).image_id = imdb.image_ids{i};
    image_roidb{1}(ii,1).im_size = imdb.sizes(i, :);
    image_roidb{1}(ii,1).imdb_name = imdb.name;
    image_roidb{1}(ii,1).overlap = overlap;
    image_roidb{1}(ii,1).boxes = all_boxes;
    image_roidb{1}(ii,1).class = class;
    image_roidb{1}(ii,1).image = [];
    image_roidb{1}(ii,1).bbox_targets = [];
    image_roidb{1}(ii,1).gt = [true(num_gt_boxes,1); false(size(boxes,1),1)];
    if isfield(rois(ii), 'trackids'), 
      image_roidb{1}(ii,1).trackids = rois(ii).trackids;
    end
    te = toc(ts);

    if visualize
      im = imread( image_roidb{1}(ii,1).image_path );
      boxes_cell = cell(length(imdb.classes), 1);
      thres = -0.7;
      for k = 1:length(boxes_cell)
        boxes_cell{k} = [abst.boxes;];
        I = boxes_cell{k}(:, 5) >= thres;
        boxes_cell{k} = boxes_cell{k}(I, :);
      end
      figure(1);
      showboxes(im, boxes_cell, imdb.classes, 'default');
    end

end

image_roidb = cat(1, image_roidb{:});
if ~isempty(bbox_means)
  [image_roidb] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end
if isfield(conf, 'regressTracks')
  if isfield(conf, 'nFramesPerVid') && conf.nFramesPerVid > numel(image_roidb)
    image_roidb = repmat(image_roidb,conf.nFramesPerVid,1 );
  end
  if isfield(conf, 'regressAllTracks') && conf.regressAllTracks
    [image_roidb] = append_all_track_regression_targets(conf, image_roidb  );
  else
    [image_roidb] = append_track_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
  end
   if isempty(image_roidb(1).track_rois) && ~valMode
     image_roidb = []; return;
  end
end

end


%%---------------------------------------------------------------------
function [image_roidb] = append_bbox_regression_targets(conf, image_roidb, means, stds)
%%---------------------------------------------------------------------
% means and stds -- (k+1) * 4, include background class
num_images = length(image_roidb);
% Infer number of classes from the number of columns in gt_overlaps
if conf.bbox_class_agnostic
    num_classes = 1;
else
    num_classes = size(image_roidb(1).overlap, 2);
end
valid_imgs = true(num_images, 1);
for i = 1:num_images
    rois = image_roidb(i).boxes;
    [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
        compute_targets(conf, rois, image_roidb(i).overlap);
end
if ~all(valid_imgs)
    image_roidb = image_roidb(valid_imgs);
    num_images = length(image_roidb);
    fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
end

% Normalize targets
for i = 1:num_images
    targets = image_roidb(i).bbox_targets;
    for cls = 1:num_classes
        cls_inds = find(targets(:, 1) == cls);
        if ~isempty(cls_inds)
            image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
            image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
        end
    end
end
end

%%---------------------------------------------------------------------
function [image_roidb] = append_track_regression_targets(conf, image_roidb, means, stds)
%%---------------------------------------------------------------------
% means and stds -- (k+1) * 4, include background class

num_images = length(image_roidb);
% Infer number of classes from the number of columns in gt_overlaps
if conf.bbox_class_agnostic
    num_classes = 1;
else
    num_classes = size(image_roidb(1).overlap, 2);
end

valid_imgs = true(num_images, 1);

for i = 1:2:num_images-1
    rois_A = image_roidb(i).boxes(image_roidb(i).gt,:);
    rois_B = image_roidb(i+1).boxes(image_roidb(i+1).gt,:);
    image_roidb(i).track_rois =  rois_A;
    image_roidb(i).track_targets = [ones(size(rois_A,1),1), zeros(size(rois_A,1),4)];
    
    if ~isfield(image_roidb(i), 'trackids') && ~isfield(image_roidb(i+1), 'trackids')
      trackIds = 0:size(rois_A,1)-1;
      continue;
    elseif isfield(image_roidb(i), 'trackids') && isfield(image_roidb(i+1), 'trackids')
      trackIds = intersect(image_roidb(i).trackids, image_roidb(i+1).trackids) ;
      if isempty(trackIds),continue; end
    else
      continue;
    end

    image_roidb(i).track_rois = zeros(numel(trackIds), 4, 'single');
    

    image_roidb(i).track_targets = zeros(numel(trackIds), 4, 'single');

    for k=1:numel(trackIds)
      image_roidb(i).track_targets(k,:) = rfcn_bbox_transform(rois_A(image_roidb(i).trackids==trackIds(k),:),  ...
        rois_B(image_roidb(i+1).trackids==trackIds(k),:));
      image_roidb(i).track_rois(k,:) = rois_A(image_roidb(i).trackids==trackIds(k), :);
    end
    gt_classes = image_roidb(i).class(image_roidb(i).gt,:);
    if conf.bbox_class_agnostic
        image_roidb(i).track_targets = [ones(numel(trackIds),1), image_roidb(i).track_targets];
    else
      error('non-bbox_class_agnostic not_implemented');
    end
end

end

%%---------------------------------------------------------------------
function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)
%%---------------------------------------------------------------------
overlap = full(overlap);

[max_overlaps, max_labels] = max(overlap, [], 2);

% ensure ROIs are floats
rois = single(rois);

bbox_targets = zeros(size(rois, 1), 5, 'single');

% Indices of ground-truth ROIs
gt_inds = find(max_overlaps == 1);

if ~isempty(gt_inds)
    % Indices of examples for which we try to make predictions
    ex_inds = find(max_overlaps >= conf.bbox_thresh);
    
    % Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));
    
    if ~all((abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4))
      fprintf('max overlap ~= max_gt_overlap\n');
    end
    
    % Find which gt ROI each ex ROI has max overlap with:
    % this will be the ex ROI's gt target
    [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
    gt_rois = rois(gt_inds(gt_assignment), :);
    ex_rois = rois(ex_inds, :);
   
    [regression_label] = rfcn_bbox_transform(ex_rois, gt_rois);
    
    if conf.bbox_class_agnostic
        bbox_targets(ex_inds, :) = [max_labels(ex_inds)>0, regression_label];
    else
        bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
    end
end

% Select foreground ROIs as those with >= fg_thresh overlap
is_fg = max_overlaps >= conf.fg_thresh;
% Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;

% check if there is any fg or bg sample. If no, filter out this image
is_valid = true;
if ~any(is_fg | is_bg)
    is_valid = false;
end
end

