% -------------------------------------------------------------------
% Author: Suman Saha & Gurkirt Singh
% gen_action_paths(): first boosts the classification scores of fast r-cnn detection boxes
% using a novel fusion strategy for merging appearance and motion cues based on the
% softmax probability scores and spatial overlaps of the detection bounding boxes.
% This fusion strategy is explained in our BMVC 2016 paper:
% Deep Learning for Detecting Multiple Space-Time Action Tubes in Video.
% Once the scores of detection bounding boxes are boosted, a first pass of
% dynamic programming is applied to construct the action paths within each test video.

% -------------------------------------------------------------------
function paths = gen_object_paths(frameBoxes,max_per_image,box_voting, tracks_cell, downweight_notrack)
if nargin < 2
    max_per_image = 200;
end
if nargin < 3
  box_voting = false;
end
if nargin < 4
  tracks = [];
else
  if numel(tracks_cell) == 3
    tracks.boxes = tracks_cell{1};
    tracks.scores = tracks_cell{2};
    tracks.c = tracks_cell{3};
  else
    tracks.fwd = tracks_cell{1};
    tracks.bwd = tracks_cell{2};
  end
end
if nargin < 5
  downweight_notrack = false;
end

object_frames = struct([]);
nms_th = 0.4;

for f = 1:length(frameBoxes)
  boxes = frameBoxes{f};
  if box_voting
   boxes = nms_box_voting(boxes, ...
     'nms_iou_thrs',nms_th,...
    'max_per_image',max_per_image,'do_bbox_voting',box_voting,...
    'add_val',0.0, ...
    'use_gpu',true, 'do_box_rescoring', false);
  boxes = boxes{:};
  else
      pick_nms = nms(boxes, nms_th); 
      if numel(pick_nms) > max_per_image;
        pick_nms = pick_nms(1:max_per_image);    
      end
      boxes = boxes(pick_nms, :); 
    end
    object_frames(f).boxes =  boxes(:, 1:4);
    object_frames(f).scores =  boxes(:, 5);
    % using the sotmax score for action a --> boxes(:, 5) and the
    % 1-boxes(:, 5) tells us the probability of the action not happeing 
    object_frames(f).allScores = [ boxes(:, 5) 1-boxes(:, 5) ]; 
    if size(boxes,2)>5, object_frames(f).allScores = [object_frames(f).allScores  boxes(:, 6) ]; 
    else
      object_frames(f).allScores = [object_frames(f).allScores  1-boxes(:, 5) ]; 
    end
    object_frames(f).boxes_idx = 1:size(object_frames(f).boxes,1); 
    if ~isempty(tracks) 
      if isfield(tracks, 'boxes') && ~isempty(tracks.boxes{f,1})
        object_frames(f).trackedboxes = tracks.boxes(f,:);
      end
      if isfield(tracks, 'fwd') && ~isempty(tracks.fwd), object_frames(f).trackfwd = tracks.fwd{f}; end
      if isfield(tracks, 'bwd') && ~isempty(tracks.bwd), object_frames(f).trackbwd= tracks.bwd{f}; end
    end
    
end
clear boxes scores pick_nms pick_softmax;
%%---------- zero_jump_link ------------
paths = zero_jump_link(object_frames, downweight_notrack);
%%--------------------------------------
path_all_score = zeros(length(frameBoxes),3);
for p = 1 : length(paths)
    for f = 1:length(frameBoxes)
        path_all_score(f,:) = object_frames(f).allScores(paths(p).idx(f),:);
        
    end
    paths(p).allScore = path_all_score;
end

end
