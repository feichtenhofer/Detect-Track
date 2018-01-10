function aboxes_out = nms_box_voting_all_imgs(...
    aboxes_in, varargin)
% post_process_candidate_detections_all_imgs performs the post-processing 
% step of non-maximum-suppression and optionally of box voting on the
% candidate bounding box detections of a set of images. 
% 
% INPUTS:
% 1) aboxes_in: contains the candidate bounding box detections of each image. 
% It can be of two forms:
%   a) (the flag is_per_class must being set to false) a NI x 1 cell array 
%   where NI is the number of images. The aboxes_in{i} element contains 
%   the candidate bounding box detection of the i-th image in the form of a 
%   NB_i x (4 + C) array where C is the number of categories and NB_i
%   is the number of candidate detection boxes of the i-th image. The first 
%   4 columns of this array contain the bounding box coordinates in the 
%   form of [x0,y0,x1,y1] (where the (x0,y0) and (x1,y1) are the top-left 
%   and bottom-right corners) and the rest C columns contain the confidence 
%   score of the bounding boxes for each of the C categories.
%   b) (the flag is_per_class must being set to true) a C x 1 cell array 
%   where C is the number of categories. The aboxes_in{j} element in this case 
%   is NI x 1 cell array, where NI is the number of images, with the candidate 
%   bounding box detections of the j-th category for each image. The
%   element aboxes_in{j}{i} is a NB_{i,j} x 5 array, where NB_{i,j} is the
%   number candidate bounding box detections of the i-th image for the j-th
%   category. The first 4 columns of this array are the bounding box 
%   coordinates in the form of [x0,y0,x1,y1] (where the (x0,y0) and (x1,y1) 
%   are the top-left and bottom-right corners) and the 5-th column contains
%   the confidence score of each bounding box with respect to the j-th category. 
% 2) The rest input arguments are given in the form of Name,Value pair
% arguments and are:
% 'threshold': is a C x 1 array, where C is the number of categories.
% It must contains the threshold per category that will be used for 
% removing candidate boxes with low confidence prior to applying the 
% non-max-suppression step.
% 'nms_iou_thrs': scalar value with the IoU threshold that will be used 
% during the non-max-suppression step.
% 'is_per_class': boolean value that if set to false, then the 1.a) form
% of the aboxes_in input parameter will be expected; otherwise, if set to 
% true then the 1.b) form of the aboxes_in input parameter will be expected
% 'max_per_image': scalar value with the maximum number of detection per
% image and per category. Default value: 200.
% 'do_bbox_voting': boolean value that if is set to True then the box voting
% step is applied. Default value: false
% 'box_ave_iou_thresh': scalar value with the minimum IoU threshold that 
% is used in order to define the neighbors of bounding box during the box
% voting step. Default value: 0.5
% 'add_val': scalar value that is added to the confidence score of bounding
% boxes in order to compute the box weight during the box voting step. 
% Default value: 1.5
% 'use_not_static_thr': boolean value that if set to true, then the threshold
% (param threshold) that is applied on the candidate bounding box
% detections will be modified (decreased from its input value) such that 
% the average number of detections per image and per category to be 
% ave_per_image. Default value: true
% 'ave_per_image': scalar value with the desired average number of
% detections per image and per category. Used only when the use_not_static_thr
% input parameter is set to true. default value: 5
%       
% OUTPUT:
% 1) aboxes_out: is a C x 1 array where C is the number of 
% categories. The aboxes_out{j} element is a NI x 1 cell array, where NI is
% the number of images, with the bounding box detections of the j-th 
% category for each image. The element aboxes_out{j}{i} is a ND_{i,j} x 5 array, 
% where ND_{i,j} is the number bounding box detections for the i-th image 
% and the j-th category. The first 4 columns of this array contain the 
% bounding box coordinates in the form of [x0,y0,x1,y1] (where the (x0,y0) 
% and (x1,y1) are the top-left and bottom-right corners) and the 5-th 
% column contains the confidence score of each bounding box with respect to
% the j-th category. 
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% Part of the code in this file comes from the R-CNN code: 
% https://github.com/rbgirshick/rcnn
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addParamValue('is_per_class',      true, @islogical);
ip.addParamValue('do_bbox_voting',    true, @islogical);
ip.addParamValue('box_ave_iou_thresh',  0.5, @isnumeric);
ip.addParamValue('use_not_static_thr', true, @islogical);
ip.addParamValue('ave_per_image',         5, @isnumeric);
ip.addParamValue('max_per_image',       300, @isnumeric);
ip.addParamValue('add_val',             0.0, @isnumeric);
ip.addParamValue('use_gpu',           true, @islogical);
ip.addParamValue('nms_iou_thrs',        0.4, @isnumeric);
ip.addParamValue('threshold',            -4, @isnumeric);
ip.addParamValue('do_box_rescoring',    false, @islogical);

ip.parse(varargin{:});
opts = ip.Results;


nms_iou_thrs       = opts.nms_iou_thrs;
threshold         = opts.threshold;
is_per_class       = opts.is_per_class;
do_bbox_voting     = opts.do_bbox_voting;
box_ave_iou_thresh = opts.box_ave_iou_thresh;
use_not_static_thr = opts.use_not_static_thr;
max_per_image      = opts.max_per_image;
ave_per_image      = opts.ave_per_image;
add_val            = opts.add_val;
use_gpu            = opts.use_gpu;
do_box_rescoring     = opts.do_box_rescoring;

if ~is_per_class
    num_imgs    = length(aboxes_in);
    num_classes = size(aboxes_in{1},2) - 4;
else
    num_classes = length(aboxes_in);
    num_imgs    = length(aboxes_in{1});
end

max_per_set   = ceil(ave_per_image * num_imgs);
aboxes_out    = cell(num_classes, 1);
for i = 1:num_classes, aboxes_out{i} = cell(num_imgs, 1); end

thresh_val_all = ones(num_classes,1,'single') * threshold;

for i = 1:num_imgs    
    % get the candidate bounding box detection of one image 
    if ~is_per_class
        assert(size(aboxes_in{i},2) == (4 + num_classes));
        bbox_cand_dets = aboxes_in{i}; 
    else
        bbox_cand_dets = cell(num_classes,1);
        for j = 1:num_classes
%             assert(size(aboxes_in{j}{i},2) == 5);
            bbox_cand_dets{j} = aboxes_in{j}{i};
        end
    end
    % post-process the candidate bounding box detection of one image
    bbox_detections_per_class = nms_box_voting(bbox_cand_dets, ...
        'thresholds',thresh_val_all, 'nms_iou_thrs',nms_iou_thrs,...
        'max_per_image',max_per_image,'do_bbox_voting',do_bbox_voting,...
        'box_ave_iou_thresh',box_ave_iou_thresh,'add_val',add_val, ...
        'use_gpu',use_gpu, 'do_box_rescoring', do_box_rescoring);

    for j = 1:num_classes, aboxes_out{j}{i} = bbox_detections_per_class{j}; end

    if (mod(i, 1000) == 0) && use_not_static_thr
        for j = 1:num_classes
            [aboxes_out{j}, thresh_val_all(j)] = keep_top_k(aboxes_out{j}, i, max_per_set, thresh_val_all(j));
        end
    end
end

if use_not_static_thr
    disp(thresh_val_all(:)');
    for i = 1:num_classes
        % go back through and prune out detections below the found threshold
        aboxes_out{i} = prune_detections(aboxes_out{i}, thresh_val_all(i));
    end
end

total_num_bboxes = zeros(num_classes, 1);
for j = 1:num_classes
    total_num_bboxes(j) = sum(cellfun(@(x) size(x,1), aboxes_out{j}, 'UniformOutput', true));
end

fprintf('Average number of bounding boxes per class\n');
disp(total_num_bboxes(:)' / num_imgs);
fprintf('Average number of bounding boxes in total\n');
disp(sum(total_num_bboxes) / num_imgs);
end

function [boxes, thresh] = keep_top_k(boxes, end_at, top_k, thresh)
% keep top K detections
X = cat(1, boxes{1:end_at});
if isempty(X), return; end

scores = sort(X(:,end), 'descend');
thresh = scores(min(length(scores), top_k));
for image_index = 1:end_at
    bbox = boxes{image_index};
    keep = find(bbox(:,end) >= thresh);
    boxes{image_index} = bbox(keep,:);
end

end

function bbox_dets = prune_detections(bbox_dets, thresh)
for j = 1:length(bbox_dets)
    if ~isempty(bbox_dets{j})
        bbox_dets{j}(bbox_dets{j}(:,end) < thresh ,:) = [];
    end
end
end