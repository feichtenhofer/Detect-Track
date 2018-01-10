function imdb = imdb_from_ilsvrc15(root_dir, image_set, flip, maxImgs)

if nargin < 4, maxImgs = 0; end;
cacheDir = fullfile(root_dir, 'imdb' , 'cache' , 'ilsvrc');
if ~exist(cacheDir, 'dir')
    mkdir(cacheDir);
end
if flip == false
    cache_file = [cacheDir '/imdb_ilsvrc15_' image_set '_unflip'];
else
    cache_file = [cacheDir '/imdb_ilsvrc15_' image_set '_flip'];
end

try
    load(cache_file);
catch
    NUM_CLS                 = 200;
    bbox_path.train         = fullfile(root_dir, 'Annotations', 'DET' , 'train');
    bbox_path.val           = fullfile(root_dir, 'Annotations', 'DET' , 'val');
    im_path.test            = fullfile(root_dir, 'ILSVRC2013_DET_test');
    im_path.train         = fullfile(root_dir, 'Data', 'DET' , 'train');
    im_path.val             = fullfile(root_dir, 'Data', 'DET', 'val');
    

    
    devkit_path             = fullfile(root_dir, 'devkit');
    
    meta_det                = load(fullfile(devkit_path, 'data', 'meta_det.mat'));
    imdb.name               = ['ilsvrc15_' image_set];
    imdb.extension          = 'JPEG';
    is_blacklisted          = containers.Map;
    
    if strcmp(image_set, 'val') || strcmp(image_set, 'train') || ...
            strcmp(image_set, 'val2') || strcmp(image_set, 'test') || ...
            strcmp(image_set, 'train14') || strcmp(image_set, 'val2_no_GT') || ...
            strcmp(image_set, 'real_test') || strcmp(image_set, 'val1_13') || ...
            strcmp(image_set, 'val1_14') || strcmp(image_set, 'pos1k_13')
        
        imdb.image_dir = im_path.(image_set);
        imdb.details.image_list_file = ...
            fullfile(root_dir, 'ImageSets', 'DET', [image_set '.txt']);
        
        if exist(imdb.details.image_list_file, 'file')
            fid = fopen(imdb.details.image_list_file, 'r');
            temp = textscan(fid, '%s %d');
            
            if strcmp(imdb.name, 'ilsvrc14_val1_14') 
                imdb.image_ids = cellfun(@(x) x(1:end-5), temp{1}, 'uniformoutput', false);
            else 
                imdb.image_ids = temp{1}; 
            end   % cell type
            
        else
                error('image_list_file does not exist! %s', imdb.details.image_list_file);
        end       
        imdb.flip = flip;
        
        if (maxImgs>0), 
          sel = round(linspace(1, length(imdb.image_ids), min(length(imdb.image_ids),maxImgs))) ;
          imdb.image_ids = imdb.image_ids(sel);
        end
        
        % blacklist case
        if strcmp(image_set, 'val') || ...
                strcmp(image_set, 'val1') || strcmp(image_set, 'val2')
            
            imdb.details.blacklist_file = ...
                fullfile(devkit_path, 'data', ...
                'ILSVRC2015_det_validation_blacklist.txt');
            [bl_image_ids, ~] = textread(imdb.details.blacklist_file, '%d %s');
            is_blacklisted = containers.Map(bl_image_ids, ones(length(bl_image_ids), 1));
            
        else
            imdb.details.blacklist_file = [];
        end
        % bbox path case
        if strcmp(image_set, 'val') || strcmp(image_set, 'val1') ...
                || strcmp(image_set, 'val2') || strcmp(image_set, 'val2_no_GT')
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
        
        % all classes are present in val/test/train14
        imdb.classes = {meta_det.synsets(1:NUM_CLS).name};
        imdb.num_classes = length(imdb.classes);
        imdb.class_to_id = containers.Map(imdb.classes, 1:imdb.num_classes);
        imdb.class_ids = 1:imdb.num_classes;
        
        imdb.image_at = @(i) ...
            fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
        
    else
        error('unknown image set');
    end
    
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
    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));

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
          imdb.sizes(i, :) = [size(im, 1) size(im, 2)];

        imdb.is_blacklisted(i) = is_blacklisted.isKey(i);
        

    end
    
    fprintf('Saving imdb to cache...');
    save(cache_file, 'imdb', '-v7.3');
    fprintf('done\n');
end


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
