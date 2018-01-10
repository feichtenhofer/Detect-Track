function imdb = imdb_from_ilsvrc15vid(root_dir, image_set, flip, maxImgs, sample)


if nargin < 4
  maxImgs = 0; 
  imdb.name = ['ilsvrc15_' image_set];
else
  imdb.name = ['ilsvrc15_' image_set num2str(maxImgs)];
end;
if nargin < 5
  sample = 'first';
else
  imdb.name = ['ilsvrc15_' image_set num2str(maxImgs) '_' sample ];
end;

useFlow = false;


cacheDir = fullfile(root_dir, 'imdb' , 'cache' , 'ilsvrc');
if ~exist(cacheDir, 'dir')
    mkdir(cacheDir);
end


if flip == false
    cache_file = [cacheDir '/imdb_' imdb.name '_unflip'];
else
    cache_file = [cacheDir '/imdb_' imdb.name '_flip'];
end

try
    load(cache_file);
catch
    NUM_CLS                 = 30 ;
    bbox_path.train         = fullfile(root_dir, 'Annotations', 'VID' , 'train');
    bbox_path.val           = fullfile(root_dir, 'Annotations', 'VID' , 'val');
    im_path.vid_test            = fullfile(root_dir, 'Data', 'VID' , 'test');
    im_path.vid_train         = fullfile(root_dir, 'Data', 'VID' , 'train');
    im_path.vid_val             = fullfile(root_dir, 'Data', 'VID', 'val');
        
    devkit_path             = fullfile(root_dir, 'devkit');
    
    meta_det                = load(fullfile(devkit_path, 'data', 'meta_det.mat'));
    if ~isempty(strfind(image_set, 'vid'))
      meta_det             = load(fullfile(devkit_path, 'data', 'meta_vid.mat'));
    end
    if ~isfield(imdb, 'name'), imdb.name               = ['ilsvrc15_' image_set]; end
    imdb.extension          = 'JPEG';
    is_blacklisted          = containers.Map;
    
            imdb.image_dir = im_path.(image_set);

    if useFlow && (strcmp(image_set, 'vid_val') || strcmp(image_set, 'vid_train') || ...
              strcmp(image_set, 'vid_test'))

      imdb.flow_dir_u = strrep(im_path.(image_set),'jpegs',['tvl1_flow_scaled' filesep 'u']);
      imdb.flow_dir_v = strrep(im_path.(image_set),'jpegs',['tvl1_flow_scaled' filesep 'v']);
      imdb.flow_dir_u = strrep(im_path.(image_set),'jpegs',['tvl1_flow_600' filesep 'u']);
      imdb.flow_dir_v = strrep(im_path.(image_set),'jpegs',['tvl1_flow_600' filesep 'v']);
      imdb.hsvflow_dir = strrep(im_path.(image_set),'jpegs',['hsvflow_600']);
    end
      if strcmp(image_set, 'vid_train')
                imdb.details.image_list_file = ...
          arrayfun(@(i) fullfile(root_dir, 'ImageSets', 'VID', ['train' sprintf('_%d.txt',i)]),1:NUM_CLS, 'uniformoutput', false);
      elseif strcmp(image_set, 'vid_val') 
        imdb.details.image_list_file = ...
            {fullfile(root_dir, 'ImageSets', 'VID', ['val' '.txt'])};
      elseif strcmp(image_set, 'vid_test')
        imdb.details.image_list_file = ...
            {fullfile(root_dir, 'ImageSets', 'VID', ['test' '.txt'])};
      else
        imdb.details.image_list_file = ...
            {fullfile(root_dir, 'ImageSets', 'VID', [image_set '.txt'])};
      end

      imdb.image_ids = {};
      for i=1:numel(imdb.details.image_list_file)
        imgList = imdb.details.image_list_file{i};
        if exist(imgList, 'file')
            fid = fopen(imgList, 'r');
            temp = textscan(fid, '%s %d');
            fclose(fid);

            imdb.image_ids = [imdb.image_ids; temp{1}]; 

        else
                error('image_list_file does not exist! %s', imgList);
        end
      end
      imdb.flip = flip;
      imdb.image_ids = unique(imdb.image_ids);
      if strcmp(image_set, 'vid_train')

        imdb.video_ids = imdb.image_ids;
        imdb.image_ids = {};
        imdb.num_frames = zeros(length(imdb.video_ids ),1);
        imdb.vid_id = [];
        for i=1:length(imdb.video_ids )
          frames = dir(fullfile(imdb.image_dir, imdb.video_ids{i}));
          frames = frames(~[frames.isdir]);
          frames =  strcat([imdb.video_ids{i}, filesep ],  sprintfc('%06d', 0:length(frames)-1) );
          imdb.image_ids = [imdb.image_ids frames] ;
          
          imdb.num_frames(i) = numel(frames);
          imdb.vid_id = [imdb.vid_id i.*ones(1,numel(frames))];
          file = cat(1,imdb.image_ids{:});
        end
        frame_idx = [0; cumsum(imdb.num_frames(1:end-1))] + 1;

      else
        file = cat(1,imdb.image_ids{:});
        file(file == '\') = '/';
        [row, col] = find(file == '/');
        [imdb.vids, frame_idx, vid_idx] = unique(cellstr(file(row,1:col)));
        imdb.num_frames = [frame_idx(2:end); row(end)]- frame_idx(1:end);
        imdb.vid_id = vid_idx;

      end
      imdb.details.blacklist_file = [];

              

        
      if (maxImgs>0)
        if (maxImgs<1000) % sample first maxImgs frames
          sel = [];
          for k=1:length(frame_idx)
            if strcmp(sample, 'first')
              add = frame_idx(k):frame_idx(k)+min(imdb.num_frames(k),maxImgs);
              imdb.num_frames(k) = numel(add);
              sel = [sel add] ;
            else
              strcmp(sample, 'uniform')
%               sel = round(linspace(frame_idx(k), frame_idx(k)+imdb.num_frames(k), min(imdb.num_frames(k),maxImgs))) ;
              add = frame_idx(k):maxImgs:frame_idx(k)+imdb.num_frames(k);

              imdb.num_frames(k) = numel(add);
              sel = [sel add] ;

            end
            
          end
        else
          sel = round(linspace(1, length(imdb.image_ids), min(length(imdb.image_ids),maxImgs))) ;
        end
        bl_image_ids = 1:numel(imdb.image_ids); bl_image_ids(sel) = [];
        imdb.image_ids = imdb.image_ids(sel);
        imdb.vid_id = imdb.vid_id(sel);
        imdb.image_id_sel = sel;
        imdb.details.blacklist_file = [cache_file '_blacklist.txt'];
        fid = fopen(imdb.details.blacklist_file, 'w');
  
        for j = 1:length(bl_image_ids)
            fprintf(fid, '%d\n', bl_image_ids(j));
        end
        fclose(fid);
      end

      if strfind(image_set, 'val') 
          imdb.details.bbox_path = bbox_path.val;
      elseif strfind(image_set, 'train')
          imdb.details.bbox_path = bbox_path.train;
      end

      if flip
          image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
          flip_image_at = @(i) sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
          for i = 1:length(imdb.image_ids)
              if ~exist(flip_image_at(i), 'file')
                  imwrite(fliplr(imread(image_at(i))), flip_image_at(i));
              end
          end
          img_num = length(imdb.image_ids)*2;
          image_ids = imdb.image_ids;
          imdb.image_ids(1:2:img_num) = image_ids;
          imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
          imdb.flip_from = zeros(img_num, 1);
          imdb.flip_from(2:2:img_num) = 1:2:img_num;
      end

      imdb.classes = {meta_det.synsets(1:NUM_CLS).name};
      imdb.num_classes = length(imdb.classes);
      imdb.class_to_id = containers.Map(imdb.classes, 1:imdb.num_classes);
      imdb.class_ids = 1:imdb.num_classes;

      imdb.image_at = @(i) ...
            fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);


    
    % private ILSVRC 2014 details
    imdb.details.meta_det    = meta_det;
    imdb.details.root_dir    = root_dir;
    imdb.details.devkit_path = devkit_path;
    % VOC-style specific functions for evaluation and region of interest DB
    imdb.eval_func = @imdb_eval_ilsvrc14;
    imdb.roidb_func = @roidb_from_ilsvrc14;
    
    % read each image to get the 'imdb.sizes'
    % Some images are blacklisted due to noisy annotations
    imdb.is_blacklisted = false(length(imdb.image_ids), 1);
    imdb.sizes = zeros( length(imdb.image_ids), 2 );
    imdb.flowScales = containers.Map;
    imdb.videoSizes = containers.Map;
    imdb.nFrames = containers.Map;
    imdb.video_ids = {};

    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
        imdb.image_ids{i} = strrep(imdb.image_ids{i},'\','/');
        [videoName, f] = fileparts(imdb.image_ids{i});
      
        if ~isKey(imdb.nFrames,videoName)
            imdb.nFrames(videoName) = numel(dir(fullfile(imdb.image_dir, videoName))) - 2 ;
            imdb.video_ids{end+1} =  videoName;
        end
        if useFlow
          num = str2double(f);
          oldname = fullfile(imdb.image_dir, videoName, ['frame' sprintf('%06d',num+1) '.jpg' ]);
          newname = fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
          if exist(oldname, 'file')
            java.io.File(oldname).renameTo(java.io.File(newname));
          end
          oldname_u = fullfile(imdb.flow_dir_u, videoName, ['frame' sprintf('%06d',num+1) '.jpg' ]);
          oldname_v = fullfile(imdb.flow_dir_v, videoName, ['frame' sprintf('%06d',num+1) '.jpg' ]);
          oldname_hsv = fullfile(imdb.hsvflow_dir, videoName, ['frame' sprintf('%06d',num+1) '.jpg' ]);
          oldname_jp600 = strrep(oldname_hsv,'hsvflow_600', 'jpegs_600');


          if exist(oldname_u, 'file')
            java.io.File(oldname_u).renameTo(java.io.File(fullfile(imdb.flow_dir_u, [imdb.image_ids{i} '.' imdb.extension])));
            java.io.File(oldname_v).renameTo(java.io.File(fullfile(imdb.flow_dir_v, [imdb.image_ids{i} '.' imdb.extension])));
            java.io.File(oldname_hsv).renameTo(java.io.File(fullfile(imdb.hsvflow_dir, [imdb.image_ids{i} '.' imdb.extension])));
            
            
          end
          if exist(oldname_jp600, 'file')
            newName = strrep(oldname_jp600,['frame' sprintf('%06d',num+1) '.jpg' ], [sprintf('%06d',num) '.' imdb.extension]);
            java.io.File(oldname_jp600).renameTo(java.io.File(newName));
          end
          if ~isKey(imdb.flowScales,videoName)
            scaleFile = fullfile(imdb.flow_dir_u, [videoName '.bin']);         
            if exist(scaleFile,'file'),
              scales =    getFlowScale(scaleFile);
              imdb.flowScales(videoName) = scales;
            end
          end

        end
 
        if ~isKey(imdb.videoSizes,videoName) % first frame, read size from image
          try
              im = imread(imdb.image_at(i));
          catch lasterror
              if strcmp(lasterror.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
                  warning('converting %s from CMYK to RGB', imdb.image_at(i));
                  cmd = ['convert ' imdb.image_at(i) ' -colorspace CMYK -colorspace RGB ' imdb.image_at(i)];
                  system(cmd);
                  im = imread(imdb.image_at(i));
              else
                  error(lasterror.message);
              end
          end
          szs = [size(im, 1) size(im, 2)];
          imdb.videoSizes(videoName) = szs;
        end
        
        imdb.sizes(i, :) = imdb.videoSizes(videoName);
          

        imdb.is_blacklisted(i) = is_blacklisted.isKey(i);
        

    end
                    
    fprintf('Saving imdb to cache...');
    save(cache_file, 'imdb', '-v7.3');
    fprintf('done\n');
end

% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);

function minMaxFlow = getFlowScale(file, frame)

fid = fopen(file,'rb');
minMaxFlow = fread(fid, [4, inf],'single');
fclose(fid);

