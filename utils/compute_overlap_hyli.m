function [true_overlap, OVERLAP] = compute_overlap_hyli(ori_pred_bbox, GT_abs)
% input
%       pred_bbox:      pre_num x 4 absotue coordinate values
%       GT_abs:         GT_num x 4 absotue coordinate values
%
% output
%       true_overlap:   structure
%                       each with GT_num x 1 overlap values

% updated on Oct.9th, 2015

OVERLAP = zeros(size(ori_pred_bbox, 1), size(GT_abs, 1));
true_overlap = struct('overlap', [], 'max', [], 'max_ind', []);
true_overlap = repmat(true_overlap, [size(ori_pred_bbox, 1) 1]);

for mm = 1:size(ori_pred_bbox, 1)
    pred_bbox = ori_pred_bbox(mm, :);
    pred_area = (pred_bbox(4)-pred_bbox(2))*(pred_bbox(3)-pred_bbox(1));
    overlap = zeros(size(GT_abs,1), 1);
    for i = 1:size(GT_abs, 1)
        
        GT_bbox = GT_abs(i, :);
        GT_area = (GT_bbox(4)-GT_bbox(2))*(GT_bbox(3)-GT_bbox(1));
        
        if pred_bbox(3) < GT_bbox(1) || pred_bbox(1) > GT_bbox(3)
            x_overlap = 0;
        else
            total_x = max(pred_bbox(3), GT_bbox(3)) - ...
                min(pred_bbox(1), GT_bbox(1));
            x_overlap = total_x - abs(GT_bbox(3) - pred_bbox(3)) ...
                - abs(GT_bbox(1) - pred_bbox(1));
        end
        if pred_bbox(4) < GT_bbox(2) || pred_bbox(2) > GT_bbox(4)
            y_overlap = 0;
        else
            total_y = max(pred_bbox(4), GT_bbox(4)) - ...
                min(pred_bbox(2), GT_bbox(2));
            y_overlap = total_y - abs(GT_bbox(4) - pred_bbox(4)) ...
                - abs(GT_bbox(2) - pred_bbox(2));
        end
        
        intersection = x_overlap * y_overlap;
        union = GT_area + pred_area - intersection;
        overlap(i) = intersection / union;
    end
    OVERLAP(mm, :) = overlap';
    true_overlap(mm).overlap = single(overlap);
    [max_value, max_ind] = max(overlap);
    true_overlap(mm).max = single(max_value);
    true_overlap(mm).max_ind = uint8(max_ind);
end