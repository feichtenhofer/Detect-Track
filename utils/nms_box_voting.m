function bbox_detections_per_class = nms_box_voting( ...
    bbox_cand_dets,  varargin)
% post_process_candidate_detections performs the post-processing step of
% non-maximum-suppression and optionally of box voting. For more details
% regarding the box voting step we refer to section 5 of the technical 
% report: http://arxiv.org/abs/1505.01749
% Note that the current implementation it includes some minor modifications
% w.r.t what is described in the technical report.
% 
%
% INPUT:
% 1) bbox_cand_dets: it contains the candidate bounding box detections to
% which the post-processing steps will be applied. It can be of two
% possible forms: 
%   a) a C x 1 cell array where C is the number of categories. The i-th
%   element of this cell array is NCB_i x 5 array with the candidate 
%   bounding box detection of the i-th category; the first 4 columns of
%   this array are the bounding box coordinates in the form of [x0,y0,x1,y1]
%   (where the (x0,y0) and (x1,y1) are the top-left and bottom-right
%   corners) and the 5-th column contains the confidence score of each 
%   bounding box with respect to the i-th category. NCB_i is the number of 
%   candidate detection boxes of the i-th category.
%   b) a NCBT x (4 + C) array where C is the number of categories and NCBT
%   is the number of candidate detection boxes; the first 4 columns of this
%   array contain the bounding box coordinates in the form of [x0,y0,x1,y1]
%   (where the (x0,y0) and (x1,y1) are the top-left and bottom-right
%   corners) and the rest C columns contain the confidence score of the
%   bounding boxes for each of the C categories.
% 2) thresholds: is a C x 1 array, where C is the number of categories.
% It must contains the threshold per category that will be used for 
% removing candidate boxes with low confidence prior to applying the 
% non-max-suppression step.
% 3) nms_iou_thrs: scalar value with the IoU threshold that will be used 
% during the non-max-suppression step.
% 4) max_per_image: scalar value with the maximum number of detection per
% image and per category.
% 5) do_bbox_voting: boolean value that if is to True then the box voting
% step is applied.
% 6) box_ave_iou_thresh: scalar value with the minimum IoU threshold that 
% is used in order to define the neighbors of bounding box during the box
% voting step.
% 7) add_val: scalar value that is added to the confidence score of bounding
% boxes in order to compute the box weight during the box voting step.
% 8) use_gpu: boolean value that if set to true then the
% non-maximum-suppresion step will be performed on the GPU
%       
% OUTPUT:
% 1) bbox_detections_per_class: is a C x 1 cell array, where C is the 
% number of categories, with the resulted object detections after applying 
% the post-processing step(s). The i-th element of bbox_detections is a
% ND_i x 5 matrix arrray with object detection of the i-th category. Each row
% contains the following values [x0, y0, x1, y1, scr] where the (x0,y0) and 
% (x1,y1) are coorindates of the top-left and bottom-right corners
% correspondingly. scr is the confidence score of the bounding box w.r.t. 
% the i-th category
%
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
ip.addParamValue('do_bbox_voting',    false, @islogical);
ip.addParamValue('box_ave_iou_thresh',  0.5, @isnumeric);
ip.addParamValue('max_per_image',       200, @isnumeric);
ip.addParamValue('add_val',             1.5, @isnumeric);
ip.addParamValue('use_gpu',           false, @islogical);
ip.addParamValue('thresholds',           -3, @isnumeric);
ip.addParamValue('nms_iou_thrs',        0.3, @isnumeric);
ip.addParamValue('do_box_rescoring',    false, @islogical);


ip.parse(varargin{:});
opts = ip.Results;

do_bbox_voting     = opts.do_bbox_voting;
box_ave_iou_thresh = opts.box_ave_iou_thresh;
max_per_image      = opts.max_per_image;
add_val            = opts.add_val;
use_gpu            = opts.use_gpu;
do_box_rescoring     = opts.do_box_rescoring;

nms_iou_thrs       = opts.nms_iou_thrs;
thresholds         = opts.thresholds;

if iscell(bbox_cand_dets)
    num_classes  = length(bbox_cand_dets);
    is_per_class = 1;
elseif isnumeric(bbox_cand_dets)
    num_classes  = 1; %size(bbox_cand_dets,2) - 4;
    is_per_class = 0;
end

if numel(thresholds) == 1, thresholds = repmat(thresholds, [num_classes, 1]); end
bbox_detections_per_class = cell(num_classes,1);

for j = 1:num_classes
    if ~is_per_class
%         assert(size(bbox_cand_dets,2) == (4 + num_classes));
%         bbox_cand_dets_this_class = [bbox_cand_dets(:,1:4), bbox_cand_dets(:,4+j)];
        bbox_cand_dets_this_class = bbox_cand_dets;
    else
%         assert(size(bbox_cand_dets{j},2) == 5);
        bbox_cand_dets_this_class = bbox_cand_dets{j};
    end

    % reject candidate detection entries with either NaN or Inf values
    reject = (any(isnan(bbox_cand_dets_this_class),2) | any(isinf(bbox_cand_dets_this_class),2));
    bbox_cand_dets_this_class = bbox_cand_dets_this_class(~reject,:);
    
    % apply the non-max-suppression step
    bbox_dets = apply_nms(bbox_cand_dets_this_class, thresholds(j), nms_iou_thrs, max_per_image, use_gpu);

    if (do_bbox_voting && ~isempty(bbox_dets))
        % apply the bounding box voting step
        bbox_dets = apply_bbox_voting(bbox_cand_dets_this_class, bbox_dets, box_ave_iou_thresh, add_val, do_box_rescoring);
    end
    
    bbox_detections_per_class{j} = bbox_dets;
end

end


function [bbox_dets, indices] = apply_nms(bbox_det_cands, score_thresh, nms_iou_thrs, max_per_image, use_gpu)
% apply_nms applies the non-maximum-suppression step on the set of
% candidate bounding box detections.
% 
% INPUT:
% 1) bbox_dets_cands: a N x 5 array with candidate bounding box detections.
% The 5-th column contains the confidence score of each bounding box.
% 2) score_thresh: scalar value with a minimum score threshold that will be
% used for removing candidate bounding box detections with low confidence 
% prior to applying the non-max-suppression step. 
% 3) nms_iou_thrs: the IoU threshold that will be used during the
% non-maximum-suppression step.
% 4) max_per_image: scalar value with the maximum number of detections in
% the image
% 5) use_gpu: boolean value that if set to true then the
% non-maximum-suppresion step will be performed on the GPU
% 
% OUTPUT:
% 1) bbox_dets: a ND x 5 array with output bounding box detections
bbox_dets = zeros(0, 5, 'single');
if ~isempty(bbox_det_cands)
    indices = find(bbox_det_cands(:,5) > score_thresh);
    keep    = nms(single(bbox_det_cands(indices,1:5)), nms_iou_thrs, use_gpu);
    indices = indices(keep);
    if ~isempty(indices)
        [~, order] = sort(bbox_det_cands(indices,5), 'descend');
        order      = order(1:min(length(order), max_per_image));
        indices    = indices(order);
        bbox_dets  = bbox_det_cands(indices,:);
    end
end

end

function bbox_dets = apply_bbox_voting(bbox_dets_cands, bbox_dets, iou_thresh, add_val, do_box_rescoring)
% apply_bbox_voting applies the bounding box voting step that refines the
% bounding box coordinates of the bbox_dets detections
% 
% INPUT:
% 1) bbox_dets_cands: a N x 5 array with candidate bounding box detections
% prior to the non-max-suppression step. The 5-th column contains the 
% confidence score of each bounding box.
% 2) bbox_dets: a ND x 5 array with the bounding box detections (computed
% after applying the non-max-suppression step to the candidate detection 
% boxes of bbox_dets_cands). The 5-th column contains the confidence score 
% of each bounding box.
% 3) iou_thresh: scalar value with the minimum IoU threshold that 
% is used in order to define the neighbors of a bounding box during the box
% voting step.
% 4) add_val: scalar value that is added to the confidence score of bounding
% boxes in order to compute the box weight during the box voting step.
% 
% OUTPUT:
% 1) bbox_dets: a ND x 5 array with output bounding box detections

num_dets = size(bbox_dets,1);
for p = 1:num_dets
    % find the bounding box neighbors of bbox_dets(p,:)
    overlap        = boxoverlap(bbox_dets_cands(:,1:4), bbox_dets(p,1:4));
    neighbors_mask = overlap >= iou_thresh;
    bbox_neighbors = bbox_dets_cands(neighbors_mask,:); 
    % the weight of each bounding box during the box voting step
    bbox_neighbors(:,5) = eps + max(0, bbox_neighbors(:,5) + add_val); 
    if any(neighbors_mask)
        % perform the box voting step for the bbox_dets(p,:) bbox detection
        bbox_voted_coords = sum(bbox_neighbors(:,1:4) .* repmat(bbox_neighbors(:,5), [1, 4]), 1);
        bbox_voted_coords = bbox_voted_coords ./ repmat(sum(bbox_neighbors(:,5)), [1, 4]);
        bbox_dets(p,1:4)  = bbox_voted_coords;
        if do_box_rescoring
          bbox_dets(p,5)  = sum(bbox_neighbors(:,5) .* overlap(neighbors_mask,:)) / sum( overlap(neighbors_mask,:));
        end
    end
end

end