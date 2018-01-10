function [ imdb, roidb ] = imdb_remove_cls(imdb, roidb, target_imdb, max_samples_from_cls )

    imdb.classes = strrep(imdb.classes,' ', '_');
    imdb.class_to_id = target_imdb.class_to_id ;
    [C,IA,IB] = intersect(imdb.classes, target_imdb.classes);
  
    valid2 = cellfun(@(x) intersect(x, IA), {roidb.rois.class},'UniformOutput',false);
    
    valid = ~cellfun(@isempty, valid2);
    
    for k = find(valid)
      roidb.rois(k).overlap = roidb.rois(k).overlap(:,IA) ;
      idx = 1;
      for l = 1:numel(roidb.rois(k).class)
        oldcls = roidb.rois(k).class(idx) ; 
        if oldcls == 0, newcls = 0;
        else, newcls = IB(oldcls == IA); end
        if isempty(newcls)
          for f = setdiff(fieldnames(roidb.rois(k))', {'feat'})
            f = char(f) ;
            roidb.rois(k).(f)(idx,:) = []; % delete that gt box
          end
        else
          roidb.rois(k).class(idx) = newcls ;
          idx=idx+1;
        end
      end
    end
    
     % clear images without any object from target_imdb
    roidb.rois = roidb.rois(valid);
    imdb.is_blacklisted = imdb.is_blacklisted(valid);
    imdb.sizes = imdb.sizes(valid,:);
    if isfield(imdb,'flip_from'), imdb.flip_from = imdb.flip_from(valid); end
    imdb.image_ids = imdb.image_ids(valid);
    imdb.image_at = @(i) ...
    fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);

    imdb.classes = target_imdb.classes;
    imdb.num_classes = target_imdb.num_classes;
    imdb.class_ids = target_imdb.class_ids;

end

