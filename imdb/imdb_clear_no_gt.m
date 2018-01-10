function [ imdb, roidb ] = imdb_clear_no_gt(imdb, roidb, use_flip )
%IMDB_CLEAR_NO_GT Summary of this function goes here
%   Detailed explanation goes here

     % clear images without any object
    valid = ~cellfun(@isempty, {roidb.rois.gt});
    roidb.rois = roidb.rois(valid);
    imdb.is_blacklisted = imdb.is_blacklisted(valid);
    imdb.sizes = imdb.sizes(valid,:);
    if use_flip, imdb.flip_from = imdb.flip_from(valid); end
    imdb.image_ids = imdb.image_ids(valid);
    imdb.image_at = @(i) ...
    fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
    if isfield(imdb,'vid_id'),    imdb.vid_id = imdb.vid_id(valid); end


end

