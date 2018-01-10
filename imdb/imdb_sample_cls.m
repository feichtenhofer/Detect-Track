function [ imdb, roidb ] = imdb_sample_cls(imdb, roidb,max_samples_from_cls, with_replacement )
    if nargin < 4
      with_replacement = false;
    end
    valid = [];
    lim = max_samples_from_cls;
    for k = imdb.class_ids
      valid2 = cellfun(@(x) any(x == k), {roidb.rois.class},'UniformOutput',false);

      valid2 =  find([valid2{:}]);
      if ~with_replacement
        lim = min(length(valid2), max_samples_from_cls);
      end
      fprintf('found %d valid samples for class %s\n', length(valid2), imdb.classes{k});
      
      sel = round(linspace(1, length(valid2), lim)) ;
      % sel = randperm(length(valid2), lim);
      valid = [valid valid2(sel)];

    end 
    
     % clear images without any object from target_imdb
    roidb.rois = roidb.rois(valid);
    imdb.is_blacklisted = imdb.is_blacklisted(valid);
    imdb.sizes = imdb.sizes(valid,:);
    if isfield(imdb,'flip_from'), imdb.flip_from = imdb.flip_from(valid); end
    imdb.image_ids = imdb.image_ids(valid);
    imdb.image_at = @(i) ...
    fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);



end

