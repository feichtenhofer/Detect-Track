function dataset = ilsvrc15(dataset, usage, use_flip, root_path)

if  exist([root_path '/' usage '.mat'])
  tic; load([root_path '/' usage '.mat']); toc; return;
end

switch usage
    case {'val15'}
        dataset.imdb_test     = imdb_from_ilsvrc15(root_path, 'val', false);
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', roidb_name_suffix, 'rootDir', root_path);
          % clear images without any object
          valid = ~cellfun(@isempty, {dataset.roidb_test.rois.gt});
            dataset.roidb_test.rois = dataset.roidb_test.rois(valid);
            dataset.imdb_test.is_blacklisted = dataset.imdb_test.is_blacklisted(valid);
            dataset.imdb_test.sizes = dataset.imdb_test.sizes(valid,:);
            dataset.imdb_test.image_ids = dataset.imdb_test.image_ids(valid);
            dataset.imdb_test.image_at = @(i) ...
            fullfile(dataset.imdb_test.image_dir, [dataset.imdb_test.image_ids{i} '.' dataset.imdb_test.extension]);

    case {'train15'}
        dataset.imdb_train     = {imdb_from_ilsvrc15(root_path, 'train', use_flip)};
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'rootDir', root_path), ...
            dataset.imdb_train, 'UniformOutput', false);
        
          % clear images without any object
        valid = ~cellfun(@isempty, {dataset.roidb_train{1, 1}.rois.gt});
          dataset.roidb_train{1, 1}.rois = dataset.roidb_train{1, 1}.rois(valid);
          dataset.imdb_train{1, 1}.is_blacklisted = dataset.imdb_train{1, 1}.is_blacklisted(valid);
          dataset.imdb_train{1, 1}.sizes = dataset.imdb_train{1, 1}.sizes(valid,:);
          if use_flip, dataset.imdb_train{1, 1}.flip_from = dataset.imdb_train{1, 1}.flip_from(valid); end
          dataset.imdb_train{1, 1}.image_ids = dataset.imdb_train{1, 1}.image_ids(valid);
            dataset.imdb_train{1, 1}.image_at = @(i) ...
            fullfile(dataset.imdb_train{1, 1}.image_dir, [dataset.imdb_train{1, 1}.image_ids{i} '.' dataset.imdb_train{1, 1}.extension]);
    case {'VIDval16'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false);
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', '', 'rootDir', root_path);       
    case {'VIDtrain16'}
        dataset.imdb_train     = {imdb_from_ilsvrc15vid(root_path, 'vid_train', use_flip)};
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', '', 'rootDir', root_path), ...
            dataset.imdb_train, 'UniformOutput', false);
    case {'VID+DETtrain16'}
        dataset.imdb_train     = {imdb_from_ilsvrc15vid(root_path, 'vid_train', use_flip) } ;
       dataset.imdb_train{2} =   imdb_from_ilsvrc15(root_path, 'train', use_flip );
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', '', 'rootDir', root_path), ...
            dataset.imdb_train, 'UniformOutput', false);
       % remove classes in DET that do not match vid
       [dataset.imdb_train{2},dataset.roidb_train{2}] = ... 
         imdb_remove_cls(dataset.imdb_train{2},dataset.roidb_train{2}, dataset.imdb_train{1}) ;
    case {'VID+DETtrain16sample'}

       dataset.imdb_train     = {imdb_from_ilsvrc15vid(root_path, 'vid_train', use_flip) } ;
       dataset.imdb_train{2} =   imdb_from_ilsvrc15(root_path, 'train', use_flip );
       dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', '', 'rootDir', root_path), ...
       dataset.imdb_train, 'UniformOutput', false);    
        
       % remove classes in DET that do not match vid
       [dataset.imdb_train{2},dataset.roidb_train{2}] = ... 
         imdb_remove_cls(dataset.imdb_train{2},dataset.roidb_train{2}, dataset.imdb_train{1}) ;

       % sample 2k classes
       [dataset.imdb_train{2},dataset.roidb_train{2}] = ... 
         imdb_sample_cls(dataset.imdb_train{2},dataset.roidb_train{2}, 2000) ;
       
       [dataset.imdb_train{1},dataset.roidb_train{1}] = ... 
         imdb_sample_frames(dataset.imdb_train{1},dataset.roidb_train{1}, 30) ;

    case {'DETtrain16sample5k'}
        dataset.imdb_train{2} =   imdb_from_ilsvrc15(root_path, 'train', use_flip );
        dataset.roidb_train(2) = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', '', 'rootDir', root_path), ...
            dataset.imdb_train(2), 'UniformOutput', false);
        
       % remove classes in DET that do not match vid
       [dataset.imdb_train{2},dataset.roidb_train{2}] = ... 
         imdb_remove_cls(dataset.imdb_train{2},dataset.roidb_train{2}, dataset.imdb_train{1}) ;

       % sample 5k classes with replacement
       [dataset.imdb_train{2},dataset.roidb_train{2}] = ... 
         imdb_sample_cls(dataset.imdb_train{2},dataset.roidb_train{2}, 5000, true) ;

    case {'VIDval16lite'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false, 5e4 );
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', 'lite50k', 'rootDir', root_path);
    case {'VIDval16lite5k'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false, 5e3 );
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', 'lite5k', 'rootDir', root_path);         
    case {'VIDval16lite30frames'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false, 30 );
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', 'lite30frames', 'rootDir', root_path);     
  case {'VIDval16lite3frames'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false, 3 );
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', 'lite3frames', 'rootDir', root_path);     
  case {'VIDval16liteStride10'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false, 10, 'uniform');
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', 'liteStride10', 'rootDir', root_path);   
  case {'VIDval16liteStride5'}
        dataset.imdb_test     = imdb_from_ilsvrc15vid(root_path, 'vid_val', false, 5, 'uniform');
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', 'liteStride5', 'rootDir', root_path);   
    case {'VIDtrain16lite3frames'}
        dataset.imdb_train     = {imdb_from_ilsvrc15vid(root_path, 'vid_train', use_flip, 3)};
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', 'lite3frames', 'rootDir', root_path), ...
            dataset.imdb_train, 'UniformOutput', false);    
    case {'VIDtrain16sample'}
        dataset.imdb_train     = {imdb_from_ilsvrc15vid(root_path, 'vid_train', use_flip) } ;
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', '', 'rootDir', root_path), ...
            dataset.imdb_train, 'UniformOutput', false);       
          frame_offset = 3;
       [dataset.imdb_train{1},dataset.roidb_train{1}] = ... 
         imdb_sample_frames(dataset.imdb_train{1},dataset.roidb_train{1}, 30, frame_offset) ;
    case {'VIDtrain16sample5frames'}
        dataset.imdb_train     = {imdb_from_ilsvrc15vid(root_path, 'vid_train', use_flip) } ;
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x, 'roidb_name_suffix', '', 'rootDir', root_path), ...
            dataset.imdb_train, 'UniformOutput', false);       
          frame_offset = 3;
       [dataset.imdb_train{1},dataset.roidb_train{1}] = ... 
         imdb_sample_frames(dataset.imdb_train{1},dataset.roidb_train{1}, 5, frame_offset) ;

  otherwise
        error('usage = ''train14'' or ''test''');
end

if  0 %strfind(usage, 'VID') &&  strfind(usage, 'train')
  if isfield(dataset, 'imdb_train')
    f='_train'; 
    imdb = dataset.(['imdb' f]){1};
    roidb = dataset.(['roidb' f]){1};
  else f='_test'; 
    imdb = dataset.(['imdb' f]);
    roidb = dataset.(['roidb' f]);
  end
  video_label_freq = zeros(numel(imdb.video_ids),imdb.num_classes);
  labelCount = zeros(numel(imdb.video_ids),1);
  for i = 1:numel(imdb.image_ids)
    labels = (unique(roidb.rois(i).overlap==1,'rows'));
    [r, c] = find(labels) ;
    
%     fprintf('%s-', imdb.classes{c});
    if isfield(imdb,'image_id_sel')
      vid_idx = imdb.vid_id(imdb.image_id_sel(i));
    else
      vid_idx = imdb.vid_id(i);
    end
    video_label_freq(vid_idx,c) = video_label_freq(vid_idx,c) + 1;
    labelCount(vid_idx) = labelCount(vid_idx) + ~isempty(labels);
  end

  video_label_freq(labelCount~=0,:) = bsxfun(@rdivide, video_label_freq(labelCount~=0,:), labelCount(labelCount~=0));
  
  if isfield(dataset, 'imdb_train')
    dataset.imdb_train{1}.video_label_freq = video_label_freq;
  else 
   dataset.imdb_test.video_label_freq = video_label_freq;
  end
end
if  ~isempty(strfind(usage, 'train')) && ~isempty(strfind(usage, 'sample'))  
  save([root_path '/' usage], 'dataset', '-v7.3');
end