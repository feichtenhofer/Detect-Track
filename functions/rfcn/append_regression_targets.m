%%---------------------------------------------------------------------
function [image_roidb] = append_regression_targets(conf, image_roidb, means, stds)
%%---------------------------------------------------------------------

num_images = length(image_roidb);

for i = 1:2:num_images-1
  
    if ~isfield(image_roidb(i), 'trackids') && ~isfield(image_roidb(i+1), 'trackids')
      trackIds = 0:sum(image_roidb(i).gt)-1;
      continue;
    elseif isfield(image_roidb(i), 'trackids') && isfield(image_roidb(i+1), 'trackids')
      trackIds = intersect(image_roidb(i).trackids, image_roidb(i+1).trackids) ;
      if isempty(trackIds),continue; end
    else
      continue;
    end
    
    image_roidb(i).track_rois = []; image_roidb(i).track_targets = [];
    for k=trackIds
      gt_box = image_roidb(i).boxes(k+1,:);
%       gt_class = image_roidb(i).class(k+1,:);
      all_boxes = image_roidb(i).boxes;
        
%       this_ovlp = image_roidb(i).overlap ;
      this_ovlp = boxoverlap(all_boxes,gt_box); 
    
      rois_A = image_roidb(i).boxes(this_ovlp > conf.fg_thresh,:);
      
      roi_B = image_roidb(i+1).boxes(k+1,:);
      
      track_rois =  rois_A;
      track_targets = [ones(size(rois_A,1),1), zeros(size(rois_A,1),4)];
    
      track_targets = rfcn_bbox_transform(rois_A, repmat(roi_B,size(rois_A,1),1));
      
      image_roidb(i).track_rois = cat(1,image_roidb(i).track_rois, rois_A);
      image_roidb(i).track_targets = cat(1,image_roidb(i).track_targets, track_targets);
    end

      
    if conf.bbox_class_agnostic
        image_roidb(i).track_targets = [ones(numel(trackIds),1), image_roidb(i).track_targets];
    else
      error('not_implemented');
    end
end

end
