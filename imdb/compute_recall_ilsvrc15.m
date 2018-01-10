function recall_per_cls = compute_recall_ilsvrc15(ab_fetch_path, top_k, imdb)


dataset_root =  imdb.details.root_dir;
addpath([imdb.details.devkit_path '/evaluation']);
ov = 0.5;


annopath = imdb.details.bbox_path;

% init stats
ld = imdb.details.meta_det;
synsets = ld.synsets;
recall_per_cls = [];
for i = 1:30
    recall_per_cls(i).wnid = synsets(i).WNID;
    recall_per_cls(i).name = synsets(i).name;
    recall_per_cls(i).total_inst = 0;
    recall_per_cls(i).correct_inst = 0;
    recall_per_cls(i).recall = 0;
end
% wnid_list = extractfield(recall_per_cls, 'wnid')';
wnid_list = {recall_per_cls.wnid};
show_num = 3000;
for i = 1:length(imdb.image_ids)
    
    tic_toc_print('evaluate prop image: (%d/%d)\n', i, length(imdb.image_ids));
    rec = VOCreadxml(fullfile(annopath, [imdb.image_ids{i}, '.xml']));
    try
        temp = squeeze(struct2cell(rec.annotation.object));
    catch
        continue;
    end
    cls_list = unique(temp(2, :));
    
if ischar(ab_fetch_path)
    file = imdb.image_ids{i};
    file(file == '\') = '/';
    ind = find(file == '/', 1, 'first');
    videoFrame = file ;
    if strcmp(file(1:20), 'ILSVRC2015_VID_train')
      videoFrame = file(ind+1:end);
    end
    ab_fetch_file = [ab_fetch_path '/' videoFrame '.mat'];


    abst = load(ab_fetch_file);
    proposals = abst.boxes(1:end,:);
    id = nms(proposals, 0.7, true); 
    proposals = proposals(id,:);
else
  proposals = ab_fetch_path{i};
end
    for j = 1:length(cls_list)
        cls_name = cls_list{j};     % wnid   
        cls_id = find(strcmp(cls_name, wnid_list)==1);

        temp_ind = cellfun(@(x) strcmp(x, cls_name), temp(2,:));
        objects = temp(3, temp_ind);
        gt = str2double(squeeze(struct2cell(cell2mat(objects))))';
        % xmax xmin ymax ymin -> x0,y0,x1,y1,obj
        gt = gt(:, [2 4 1 3]);
                 
        try
            bbox_candidate = floor(proposals(1:top_k, 1:4));
        catch
            bbox_candidate = floor(proposals(:, 1:4));
        end
        
        [true_overlap, ~] = compute_overlap_hyli(gt, bbox_candidate);
        
%         ov_vector = compute_overlap(bbox_candidate,temp, cls_id)
        correct_inst = sum([true_overlap.max] >= ov);
        
        recall_per_cls(cls_id).correct_inst = ...
            recall_per_cls(cls_id).correct_inst + correct_inst;  
        recall_per_cls(cls_id).total_inst = ...
            recall_per_cls(cls_id).total_inst + size(gt, 1);    
    end
       
end
disp('');
for i = 1:30
    recall_per_cls(i).recall = ...
        recall_per_cls(i).correct_inst/recall_per_cls(i).total_inst;
    fprintf('cls #%3d: %s\t\trecall: %.4f\n', ...
        i, recall_per_cls(i).name, recall_per_cls(i).recall);
end
