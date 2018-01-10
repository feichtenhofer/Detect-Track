function paths = make_tubes(frameBoxes,max_per_image,box_voting, tracks_cell, downweight_notrack)
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
      nms_idx = nms(boxes, nms_th); 
      if numel(nms_idx) > max_per_image
        nms_idx = nms_idx(1:max_per_image);    
      end
      boxes = boxes(nms_idx, :); 
  end
    object_frames(f).boxes =  boxes(:, 1:4);
    object_frames(f).scores =  boxes(:, 5);
    object_frames(f).boxes_idx = 1:size(object_frames(f).boxes,1); 
    
    if ~isempty(tracks) 
      if isfield(tracks, 'boxes') && ~isempty(tracks.boxes{f,1})
        object_frames(f).trackedboxes = tracks.boxes(f,:);
      end
      if isfield(tracks, 'fwd') && ~isempty(tracks.fwd), object_frames(f).trackfwd = tracks.fwd{f}; end
      if isfield(tracks, 'bwd') && ~isempty(tracks.bwd), object_frames(f).trackbwd= tracks.bwd{f}; end
    end
    
end
%%---------- zero_jump_link ------------
paths = zero_jump_link(object_frames, downweight_notrack);


end
