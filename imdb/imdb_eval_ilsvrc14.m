function res = imdb_eval_ilsvrc14(all_boxes, imdb, conf, suffix, do_nms, do_tracks)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Delete results files after computing APs
rm_res = ~true;
ignore = []
box_voting = 0;
if nargin < 5
do_nms = true;
end
if nargin < 6
do_tracks = false;
end
% save results
if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
    suffix = '';
else
    if suffix(1) ~= '_'
        suffix = ['_' suffix];
    end
end
if do_nms
if box_voting
 all_boxes = nms_box_voting_all_imgs(all_boxes);
else
  top_k = Inf; % Inf; % 30000;
  for cls = 1:length(all_boxes)
      tic_toc_print('Applying NMS for class %d/%d\n', ...
          cls, length(all_boxes));

      % Apply NMS
      boxes = all_boxes{cls};
      for image_index = 1:length(boxes);
          if ~isempty(boxes{image_index})
          bbox = boxes{image_index}(:,1:end);
          keep = nms(bbox(:,1:5), 0.3);
          boxes{image_index} = bbox(keep,:);
  %           boxes{image_index} = AttractioNet_postprocess(...
  %                 boxes{image_index}, ...
  %                 'mult_thr_nms',     true, ...
  %                 'nms_iou_thrs',     .3, ...
  %                 'max_per_image', Inf);
          end
      end

      % Keep top K
      X = cat(1, boxes{:});
      scores = sort(X(:,5), 'descend');
      thresh = scores(min(length(scores), top_k));
  %     thresh = .1;
      for image_index = 1:length(boxes);
          bbox = boxes{image_index};
          %     keep = find(bbox(:,end) >= thresh);
          %     boxes{image_index} = bbox(keep,:);
          if ~isempty(bbox)
              keep = find(bbox(:,5) >= thresh);
              boxes{image_index} = bbox(keep,:);
          else
              keep = [];
              boxes{image_index} = [];
          end
      end
      all_boxes{cls} = boxes;
  end
end
end
addpath(fullfile(imdb.details.devkit_path, 'evaluation'));

pred_file = tempname();

% write out detections in ILSVRC format
fid = fopen(pred_file, 'w');
for cls = 1:length(all_boxes)
    tic_toc_print('writing out detections for class %d/%d\n', ...
        cls, length(all_boxes));
    boxes = all_boxes{cls};
    for image_index = 1:length(boxes);
        bbox = boxes{image_index}; if isempty(bbox), continue; end
        if isfield(imdb, 'image_id_sel'), id = imdb.image_id_sel(image_index);
        else, id = image_index; end
        %    <frame_id> <ILSVRC2015_VID_ID> -1 <confidence> <xmin> <ymin> <xmax> <ymax>
%         for j = 1:size(bbox,1)
%             fprintf(fid, '%d %d -1 %.3f %d %d %d %d\n', ...
%                id , cls, bbox(j,end), round(bbox(j,1:4)));
%         end
        
        if do_tracks
          for j = 1:size(bbox,1)
            if size(bbox,2) > 5
              fprintf(fid, '%d %d %d %f %.3f %.3f %.3f %.3f\n', id, cls, bbox(j,6), bbox(j,5), bbox(j,1:4) - 1);
            else
              fprintf(fid, '%d %d %d %f %.3f %.3f %.3f %.3f\n', id, cls, j+(cls-1)*1000, bbox(j,end), bbox(j,1:4) - 1);
            end
          end
        else
          for j = 1:size(bbox,1)
            fprintf(fid, '%d %d -1 %f %.3f %.3f %.3f %.3f\n', id, cls, bbox(j,5), bbox(j,1:4) - 1);
          end
        end
        % we subtract one from bbox because coordinate starts from 0 in ilsvrc
    end
end
fclose(fid);

meta_file = fullfile(imdb.details.devkit_path, 'data', 'meta_vid.mat');
eval_file = imdb.details.image_list_file{1};
blacklist_file = imdb.details.blacklist_file;

optional_cache_file = fullfile(imdb.details.root_dir, 'eval_det_cache', ...
    [imdb.name '.mat']);
mkdir_if_missing(fileparts(optional_cache_file));
gtruth_directory = imdb.details.bbox_path;

fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);

if ~do_tracks
  [ap, recall, precision] = eval_vid_detection(pred_file, gtruth_directory, ...
      meta_file, eval_file, blacklist_file, optional_cache_file);
else
  optional_cache_file = fullfile(imdb.details.root_dir, 'eval_det_cache', ...
    [imdb.name '_tracking.mat']);
  [ap, recall, precision] = eval_vid_tracking(pred_file, gtruth_directory, ...
    meta_file, eval_file, blacklist_file, optional_cache_file);
  ap = ap{2};   recall = recall{2};   precision = precision{2}; 
end
%   test_file =  fullfile(imdb.details.devkit_path, 'evaluation', 'demo.val.pred.vid.txt');
  
% [ap, recall, precision] = eval_vid_detection(test_file, gtruth_directory, ...
%     meta_file, eval_file, blacklist_file, optional_cache_file);
load(meta_file);
fprintf('-------------\n');
fprintf('Category\tAP\n');
for i = 1:30
    % for i = 1:200
    s = imdb.details.meta_det.synsets(i).name;
    if length(s) < 8
        fprintf('%s\t\t%0.3f\n',s,ap(i));
    else
        fprintf('%s\t%0.3f\n',s,ap(i));
    end
end

fprintf(' - - - - - - - - \n');
fprintf('Mean AP:\t %0.3f\n',mean(ap));
fprintf('Median AP:\t %0.3f\n',median(ap));

res.recall = recall;
res.prec = precision;
res.ap = ap;

save([conf.root_path 'eval_detection_' imdb.name suffix], ...
    'res', 'recall', 'precision', 'ap');

if rm_res
    delete(pred_file);
end

rmpath(fullfile(imdb.details.devkit_path, 'evaluation'));
