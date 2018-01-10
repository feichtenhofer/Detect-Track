
data = caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('data')).get_data();
rois = squeeze(caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('rois')).get_data());
labels = squeeze(caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('labels')).get_data());
bbox_targets = squeeze(caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('bbox_targets')).get_data());
bbox_loss_weights = squeeze(caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('bbox_loss_weights')).get_data());
cls_score = squeeze(caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('cls_score')).get_data());


scores = squeeze(caffe_solver.net.blobs('cls_score').get_data());
classes = [{'bg'} imdb_train{1}.classes];
figure(1);
for k = 1:size(data,4)

  img = permute((data(:,:,:,k)+128)/256, [2 1 3 4 5 6]);
  img_ = img(:,:, [3 2 1]);
inds = rois(1,:)==k-1;
  boxes = rois(2:end,inds)';
  
  bbox_reg_boxes = find( bbox_loss_weights(5,inds) > 0 ) ;
  bbox_reg_weights = bbox_targets(:,bbox_reg_boxes) ;

  boxes_cell = {boxes(bbox_reg_boxes,:)};
          
  subplot(1,size(data,4),k); 
  try
    track_deltas = caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('bbox_disp')).get_data();
    track_boxes = caffe_solver.net.blob_vec(caffe_solver.net.name2blob_index('rois_disp')).get_data();
    track_boxes = squeeze(track_boxes)' +1;
    pred_boxes = rfcn_bbox_transform_inv(track_boxes(:,2:end), squeeze(track_deltas)');
    pred_boxes = pred_boxes(:, 5:end);
    if k==track_boxes(1),   boxes_cell = {track_boxes(:,2:end)};
    else
       boxes_cell = {pred_boxes};
    end
  catch
  end
  this_scrs = scores(:,inds);
  [votes,predictions] = sort(this_scrs(:,bbox_reg_boxes), 1, 'descend') ;
        top5pred =  squeeze(predictions(1:5,:));
        highest_class_names = strrep(classes(top5pred), '_', '\_');

  if size(img,3) > 3
    subplot(2,2,1);showboxes(img_, boxes_cell);
    subplot(2,2,2); image(img(:,:,[4 : 6],:)); axis image;
    if size(img,3) > 10
      subplot(2,2,3); image(img(:,:,[7 : 9],:)); axis image;
      subplot(2,2,4); image(img(:,:,[9 : 11],:)); axis image;
    end
  else
    showboxes(img_, boxes_cell);
  end
  title(['top5 classes=' highest_class_names{1:5} ]);

end
