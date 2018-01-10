function roidb = roidb_from_ilsvrc14(imdb,varargin)
% roidb = roidb_from_ilsvrc14(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParameter('exclude_difficult_samples',       false,   @islogical);
ip.addParameter('with_selective_search',           false,  @islogical);
ip.addParameter('with_edge_box',                   false,  @islogical);
ip.addParameter('with_self_proposal',              false,  @islogical);
ip.addParameter('rootDir',                         '.',    @ischar);
ip.addParameter('extension',                       '',     @ischar);
ip.addParameter('roidb_name_suffix',               '',     @isstr);
ip.addParameter('regions_file_sp',               '',     @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;

cacheDir = fullfile(opts.rootDir, 'imdb' , 'cache' , 'ilsvrc');
if ~exist(cacheDir, 'dir')
    mkdir(cacheDir);
end



try
    flip = imdb.flip;
catch
    flip = false;
end


if isempty(opts.roidb_name_suffix)
   opts.roidb_name_suffix = '';
end
if flip == false
    cache_file = [cacheDir '/roidb_' imdb.name opts.roidb_name_suffix '_unflip.mat'];
else
    cache_file = [cacheDir '/roidb_' imdb.name opts.roidb_name_suffix '_flip.mat'];
end

%%
try
   load(cache_file);
catch
    addpath(fullfile(imdb.details.devkit_path, 'evaluation'));
    
    roidb.name = imdb.name;
    % wsh  regions_file = fullfile('data', 'selective_search_data', [roidb.name '.mat']);    
    regions = [];
    if opts.with_selective_search
        fprintf('Loading SS region proposals...');
        regions = load_proposals(regions_file_ss, regions);
        fprintf('done\n');
    end
    if opts.with_edge_box
        regions = load_proposals(regions_file_eb, regions);
    end
    if opts.with_self_proposal
        regions = load_proposals(opts.regions_file_sp, regions);
        regions = renameStructField(regions,'roi', 'boxes');
        regions = renameStructField(regions,'list', 'images');

    end
    
    if isempty(regions)
        fprintf('Warrning: no ADDITIONAL windows proposal is loaded!\n');
        regions.boxes = cell(length(imdb.image_ids), 1);
        if flip
            regions.images = imdb.image_ids(1:2:end);
        else
            regions.images = imdb.image_ids;
        end
    end
    
    hash = make_hash(imdb.details.meta_det.synsets);
    WNIDs = {imdb.details.meta_det.synsets.WNID};
    if ~flip
        
        for i = 1:length(imdb.image_ids)
            if ~mod(i,100)
              tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));  
            end
            try
                anno_file = fullfile(imdb.details.bbox_path, [imdb.image_ids{i} '.xml']);
                if strfind(anno_file, '_flip'), 
                  flipthat = true;
                  anno_file = strrep(anno_file, '_flip', '');
                else
                  flipthat = false;
                end
                if ~ispc,         anno_file(anno_file == '\') = '/'; end
                if strcmp(imdb.name(1:12),'ilsvrc15_vid')
                  rec = VOCreadrecTrackxml(anno_file, hash);
                else
                  rec = VOCreadrecxml(anno_file, hash);
                end
            catch
                warning('GT(xml) file empty/broken: %s\n', imdb.image_ids{i});
         
 
                rec = [];
            end
          if ~isempty(regions)
            if numel(regions.images) ~= numel(imdb.image_ids) %smaller imdb due to subsampling -> find right regions
              j = find(strcmp(regions.images,[imdb.image_ids{i} '.' imdb.extension])) ;
            else
              j = i;
            end
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{j});
                assert(strcmp(image_name1, image_name2));
          end
            roidb.rois(i) = attach_proposals(rec, regions.boxes{j}, WNIDs, opts.exclude_difficult_samples, flipthat);
        end
        
    else
        % flip case
        for i = 1:length(imdb.image_ids)/2
            
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);        
            try
                anno_file = fullfile(imdb.details.bbox_path, [imdb.image_ids{2*i-1} '.xml']);
                if strcmp(imdb.name(1:12),'ilsvrc15_vid')
                  rec = VOCreadrecTrackxml(anno_file, hash);
                else
                  rec = VOCreadrecxml(anno_file, hash);
                end
            catch
                warning('GT(xml) file empty/broken: %s\n', imdb.image_ids{2*i-1});
                rec = [];
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i*2-1) = attach_proposals(rec, regions.boxes{i*2-1}, WNIDs, opts.exclude_difficult_samples, false);
            roidb.rois(i*2) = attach_proposals(rec, regions.boxes{i*2}, WNIDs, opts.exclude_difficult_samples, true);
        end
    end
    
    rmpath(fullfile(imdb.details.devkit_path, 'evaluation'));
    
    fprintf('Saving roidb to cache...');
    save(cache_file, 'roidb', '-v7.3');
    fprintf('done\n');
end



% ------------------------------------------------------------------------
function rec = attach_proposals(ilsvrc_rec, boxes, WNIDs, exclude_difficult_samples, flip)

if size(boxes,2) > 4
  boxes = boxes(:,1:4);
end


if isfield(ilsvrc_rec, 'objects') && ~isempty(ilsvrc_rec.objects)
    if exclude_difficult_samples
        valid_objects = ~cat(1, ilsvrc_rec.objects(:).difficult);
    else
        valid_objects = 1:length(ilsvrc_rec.objects(:));
    end   
    gt_boxes = cat(1, ilsvrc_rec.objects(valid_objects).bbox);
    
   
    %%% ============ NOTE ==============
    % coordinate starts from 0 in ilsvrc
    gt_boxes = gt_boxes + 1;
    
    if flip
        gt_boxes(:, [1, 3]) = ilsvrc_rec.imgsize(1) + 1 - gt_boxes(:, [3, 1]);
    end
    all_boxes = cat(1, gt_boxes, boxes);
    gt_classes = zeros(length(valid_objects),1);
    for i = 1:length(valid_objects)
      gt_classes(i) = find(strcmp(WNIDs,ilsvrc_rec.objects(valid_objects(i)).class));
    end
    num_gt_boxes = size(gt_boxes, 1);
    
else
    gt_boxes = [];
    all_boxes = boxes;
    gt_classes = [];
    num_gt_boxes = 0;
end

num_boxes = size(boxes, 1);
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
if ~isempty(ilsvrc_rec.objects) && isfield(ilsvrc_rec.objects,'trackid')
  rec.trackids = uint8(cat(1, ilsvrc_rec.objects(:).trackid));
  rec.occluded = logical(cat(1, ilsvrc_rec.objects(:).occluded));
  rec.generated = logical(cat(1, ilsvrc_rec.objects(:).generated));
else
  rec.trackids = [];
  rec.occluded = [];
  rec.generated =[];
end
rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, length(WNIDs), 'single');
for i = 1:num_gt_boxes
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end


% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions)
% ------------------------------------------------------------------------
if isempty(regions)
    regions = load(proposal_file);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes(:), regions_more.boxes(:), 'UniformOutput', false);
end

function rec = VOCreadrecTrackxml(path,hash)

x=VOCreadxml(path);
x=x.annotation;

rec.folder=x.folder;
rec.filename=x.filename;
rec.source.database=x.source.database;

rec.size.width=str2double(x.size.width);
rec.size.height=str2double(x.size.height);

rec.imgname=[x.folder x.filename];
rec.imgsize=str2double({x.size.width x.size.height});
rec.database=rec.source.database;

if isfield(x,'object')
    for i=1:length(x.object)
        rec.objects(i)=xmlobjtopas(x.object(i),hash);
    end
else
    rec.objects = [];
end

function p = xmlobjtopas(o,hash)

p.class=o.name;

p.label= get_class2node( hash, p.class );

p.bbox=str2double({o.bndbox.xmin o.bndbox.ymin o.bndbox.xmax o.bndbox.ymax});

p.trackid = str2double(o.trackid);
p.occluded = str2double(o.occluded);
p.generated = str2double(o.generated);


