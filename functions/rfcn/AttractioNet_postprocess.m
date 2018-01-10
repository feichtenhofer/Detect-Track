function [bboxes_out, keep_indices] = ...
    AttractioNet_postprocess(bboxes_in, varargin)
% AttractioNet_postprocess: It implements the single/multi threshod
% non-max-suppression (NMS) steps involved in AttractioNet
% 
% INPUT:
% 1) bboxes_in: contains the candidate bounding boxes on which the NMS 
% step(s) will be applied. It is a Nb x 5 array, where Nb is the number of 
% input candidate bounding boxes.
%
% 2) thresholds: scalar value with a minimum score threshold used for
% removing candidate bounding boxes with low confidence prior to applying 
% the NMS step(s).
%
% 3) nms_iou_thrs: Nk x 1 vector with the IoU threshold(s) that will be 
% used during the NMS step(s). In case the single NMS step is applied Nk
% equals to 1.
%
% 4) max_per_image: Nk x 1 vector with the maximum number of bounding boxes 
% that will be kept after each NMS step.
% 
% 5) mult_thr_nms: boolean value, if set to true then the multi-threshold
% non-max-suppresion strategy is applied.
%       
% OUTPUT:
% 1) bboxes_out: K x 5 arrray with output bounding boxes. It has the same
% format as bboxes_in. 

ip = inputParser;
ip.addParameter('thresholds',         -inf, @isnumeric);
ip.addParameter('max_per_image',       200, @isnumeric);
ip.addParameter('nms_iou_thrs',        0.3, @isnumeric);
ip.addParameter('use_gpu',            true, @islogical);
ip.addParameter('mult_thr_nms',      false, @islogical);
ip.parse(varargin{:});
opts = ip.Results;

max_per_image      = opts.max_per_image;
use_gpu            = opts.use_gpu;
nms_iou_thrs       = opts.nms_iou_thrs;
thresholds         = opts.thresholds;

assert(isnumeric(bboxes_in))
assert(size(bboxes_in,2)==5)

% remove bounding boxes that contain NaN or Inf values (just a safeguard)
[bboxes_in, keep_indices] = remove_bboxes_with_non_valid_values(bboxes_in);
    
if opts.mult_thr_nms
    % Apply the multi-threshold non-max-suppression step
    [bboxes_out, indices] = apply_multi_thr_nms(bboxes_in, thresholds, nms_iou_thrs, max_per_image, use_gpu);
else
    % Apply the single-threshold non-max-suppression step
    [bboxes_out, indices] = apply_single_thr_nms(bboxes_in, thresholds, nms_iou_thrs, max_per_image, use_gpu);
end
keep_indices = keep_indices(indices);
end

function [bboxes_in, keep_indices] = remove_bboxes_with_non_valid_values(bboxes_in)
% remove bounding boxes that contain NaN or Inf values 
reject = (any(isnan(bboxes_in),2) | any(isinf(bboxes_in),2));
if any(reject)
    keep_indices = find(~reject);
    bboxes_in = bboxes_in(keep_indices,:);
else
    keep_indices = 1:size(bboxes_in,1);
end
keep_indices = keep_indices(:);
end

function [bboxes_out, indices] = apply_single_thr_nms(bboxes_in, score_thresh, nms_iou_thrs, max_per_image, use_gpu)
% apply_nms applies the non-maximum-suppression step on the set of scored
% candidate bounding boxes.

bboxes_out = zeros(0, 5, 'single');
indices   = zeros(0, 1, 'single');
if ~isempty(bboxes_in)
    indices = find(bboxes_in(:,5) > score_thresh);
    % apply nms
    keep    = nms(single(bboxes_in(indices,:)), nms_iou_thrs, use_gpu);
    indices = indices(keep);
    if ~isempty(indices)
        % keep top max_per_image bounding boxes
        [~, order] = sort(bboxes_in(indices,5), 'descend');
        order      = order(1:min(length(order), max_per_image));
        indices    = indices(order);
        bboxes_out  = bboxes_in(indices,:);
    end
end

end

function [bboxes_out, indices] = ...
    apply_multi_thr_nms(bboxes_in, score_thresh, nms_iou_thrs, max_per_image, use_gpu)
% Applies multi-threshold non-maximum-suppression step of scored candidate bounding boxes.

assert(length(nms_iou_thrs) == length(max_per_image))
assert(issorted(nms_iou_thrs(end:-1:1)),...
    'The IoU thresholds of the Multi-threshold NMS step must be in decreasing order')

bboxes_out = zeros(0, 5, 'single');
indices    = zeros(0, 1, 'single');
bboxes_in  = single(bboxes_in);
if ~isempty(bboxes_in)
    
    num_iou_thrs = length(nms_iou_thrs);
    inds    = cell(num_iou_thrs,1);
    inds{1} = find(bboxes_in(:,5) > score_thresh);
        
    % ----------- STAGE 1:  FIRST SINGLE THRESHOLD NMS STEP ---------------
    % Apply the first step of non-max-suppresion with IoU=nms_iou_thrs(1) 
    % and keep the top max_per_image(1) bounding boxes.
    % Basically, this is a prelimary NMS step that its purpose is to remove
    % almost identical boxes. In the case of the AttractioNet model the 
    % IoU threshold of this NMS step is 0.95 and then the top 2000 
    % candidate box proposals are kept afterwards.
    % inds{1} : the indices of bounding boxes kept after the first NMS step
    inds{1} = inds{1}(nms(bboxes_in(inds{1},:), nms_iou_thrs(1), use_gpu)); % apply NMS with IoU = nms_iou_thrs(1)
    inds{1} = inds{1}(1:min(max_per_image(1),length(inds{1}))); % keep top max_per_image(1) bounding boxes
    %----------------------------------------------------------------------
    
    % ----------- STAGE 2: MULTI-THRESHOLD NMS RE-ORDERING ----------------
    % This is where actually the multi-threshold NMS mechanism starts. Its
    % purpose is to re-order the bounding boxes (whose indices are in inds{1})
    % that the stage 1 single threshold NMS step gave. Note that each of
    % NMS steps involved in this second stage are applied directly on the
    % bounding boxes produced from the first strage and NOT in CONSEQUATIVE 
    % order. 
    for i = 2:num_iou_thrs
        inds{i} = inds{1}(nms(bboxes_in(inds{1},:), nms_iou_thrs(i), use_gpu)); % i-th NMS step
    end

    % Take the top max_per_image(end) bounding boxes of the last IoU
    % threshold. Note that it is assumed that the IoU thresholds are in
    % decreasing order. So the last IoU threshold would be the smallest.
    
    % keep top max_per_image(end) bounding boxes
    indices = inds{end}(1:min(max_per_image(end),length(inds{end}))); 
    indices = indices(:);
    % change the scores of the selected bounding boxes
    scores  = (num_iou_thrs-1) + bboxes_in(indices,5); 
    
    for i = (num_iou_thrs-1):-1:1 % For the remaining NMS steps
        % (a) Find the indices of the bounding boxes produced during the i-th 
        % NMS step and that are not in the set of the already selected output
        % bounding boxes (indices = the already selected output bounding boxes)
        inds_this_thr = setdiff(inds{i}, indices, 'stable');
        
        % (b) Take from the above set of bounding boxes the top (max_per_image(i)-length(indices)) 
        % bounding boxes where length(indices) are the number of output bounding
        % boxes already selected. 
        
        % hyli: not 100% sure
        temp = max_per_image(i)-length(indices);
        %temp = max_per_image(i);
        
        inds_this_thr = inds_this_thr( 1 : min(temp, length(inds_this_thr)) );
        % change the scores of the just picked bounding boxes
        scores_this_thr = (i-1) + bboxes_in(inds_this_thr,5);
        
        % (c) Add the bounding boxes picked on the above step (set: inds_this_thr)
        % to the set of output bounding boxes (set: indices)
        indices   = [indices; inds_this_thr(:)]; 
        scores    = [scores; scores_this_thr(:)];
    end
    bboxes_out      = bboxes_in(indices,:);
    bboxes_out(:,5) = scores;    
    %----------------------------------------------------------------------

    % Make sure that there are no more than max_per_image(1) bounding boxes
    bboxes_out      = bboxes_out(1:min(max_per_image(1),size(bboxes_out,1)),:);
end

end