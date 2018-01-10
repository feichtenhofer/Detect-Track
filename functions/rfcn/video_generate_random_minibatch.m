function [shuffled_vids,shuffled_frames, sub_inds] = video_generate_random_minibatch(shuffled_vids,shuffled_frames, image_roidb_train, conf, shuffle)
% Video-level minibatch sampling
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

    if nargin < 5,      shuffle = true; end
      
    % shuffle training data per batch
    if isempty(shuffled_vids)

        nFrameSample = 30;
        doReplicateFrames  = false;
        
        if isfield(image_roidb_train, 'videoSizes')
          videoSz = image_roidb_train.videoSizes.values; videoSz = cat(1,videoSz{:});
          hori_image_inds = videoSz(:,2) > videoSz(:,1);
        else
          hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
        end
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);
        vert_image_inds = find(vert_image_inds);
        
        if shuffle
          % random perm
          lim = floor(length(hori_image_inds) / conf.ims_per_batch) * conf.ims_per_batch;
          hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
          lim = floor(length(vert_image_inds) / conf.ims_per_batch) * conf.ims_per_batch;
          vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
        end
        % combine sample for each conf.ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, conf.ims_per_batch, []);
        vert_image_inds = reshape(vert_image_inds, conf.ims_per_batch, []);
        
        shuffled_vids = [hori_image_inds, vert_image_inds];

        shuffled_vids = num2cell(shuffled_vids, 1);
        
        shuffled_frames = {}; 
        for i=1:numel(image_roidb_train.video_ids)
          shuffled_frames{i} = find(image_roidb_train.vid_id == i);
        end
        
%         %% replicate low #classes
%           nFrames = image_roidb_train.num_frames;
%           video_label_freq = image_roidb_train.video_label_freq;
%           label_freq = sum(image_roidb_train.video_label_freq,1)';
%           class_rep_factor = max(label_freq)./(label_freq);
%   %         [rows,cols] = find(video_label_freq > 0.9);
%           if doReplicateFrames
%             rep_factor = video_label_freq * class_rep_factor;
%           else
%             rep_factor = ones(size(nFrames));
%           end
%             shuffled_vids = {}; shuffled_frames = {};
%             count = 0;
%             for i=1:numel(rep_factor)
%               shuffled_vids{i} = repmat(i,1,round(rep_factor(i)));
%               shuffled_frames{i} = randperm(nFrames(i),min(nFrames(i),nFrameSample)) + count;
%               count = count + nFrames(i);
%             end
%           shuffled_vids = cat(2,shuffled_vids{:}); 
%           perm = randperm(size(shuffled_vids, 2));
%           shuffled_vids = shuffled_vids(:,perm);
        
    end
    
if nargout > 2
      sub_inds = zeros(conf.ims_per_batch,conf.nFramesPerVid);
      if numel(conf.time_stride) > 1, 
        conf.time_stride = conf.time_stride(randi(numel(conf.time_stride))); 
      end

      count = 0;
      while count < conf.ims_per_batch
        % first sample a vid
  
          vid = shuffled_vids{1}; 
          if isempty(shuffled_frames{vid}), 
            shuffled_vids(shuffled_vids ==vid) = [] ;
            continue;
          elseif shuffle
            shuffled_vids = circshift(shuffled_vids,[0 1]); 
          end
          count = count+1;


          if length(shuffled_frames{vid}) < conf.nFramesPerVid * conf.time_stride,
            shuffled_frames{vid} = padarray(shuffled_frames{vid},[0 conf.nFramesPerVid * conf.time_stride],'symmetric','post');
          end
          if length(shuffled_frames{vid}) >= conf.nFramesPerVid * conf.time_stride && shuffle
            s = randi(length(shuffled_frames{vid})-conf.nFramesPerVid * conf.time_stride + 1);
            sub_inds(count,:) = shuffled_frames{vid}(s:conf.time_stride:s+conf.nFramesPerVid*conf.time_stride-1);
          elseif length(shuffled_frames{vid}) >= conf.nFramesPerVid * conf.time_stride && ~shuffle
            sub_inds(count,:) = shuffled_frames{vid}(1:conf.time_stride:conf.nFramesPerVid*conf.time_stride);
            shuffled_frames{vid}(1:conf.time_stride:conf.nFramesPerVid*conf.time_stride) = [];
          end
      end
end