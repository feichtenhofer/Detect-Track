function [ imdb, roidb ] = imdb_sample_frames(imdb, roidb,max_samples_from_cls, offset )
    if nargin < 4
      offset = 0;
    end
    valid = zeros(size(imdb.image_ids));
    if iscell(max_samples_from_cls)
      [C,ia,ib] = intersect(imdb.image_ids, max_samples_from_cls);
      valid = ia;
    else
      nFrames = imdb.num_frames;
      count=0;
      for i=1:numel(nFrames)
        % shuffled_frames{i} = randperm(nFrames(i),min(nFrames(i),max_samples_from_cls)) + count;
        shuffled_frames{i} = round(linspace(1+offset,nFrames(i)-offset,min(nFrames(i)-2*offset,max_samples_from_cls))) + count;
        count = count + nFrames(i);
      end
      valid = [shuffled_frames{:}];
    end
    
     % clear images without any object from target_imdb

    roidb.rois = roidb.rois(valid);
    imdb.is_blacklisted = imdb.is_blacklisted(valid);
    imdb.sizes = imdb.sizes(valid,:);
    if isfield(imdb,'flip_from'),  imdb.flip_from = imdb.flip_from(valid); end
    imdb.image_ids = imdb.image_ids(valid);
    imdb.image_at = @(i) ...
    fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
    if isfield(imdb,'vid_id'),    imdb.vid_id = imdb.vid_id(valid); end


end

