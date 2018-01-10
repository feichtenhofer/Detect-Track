function blob = im_list_to_blob(ims)
    max_shape = max(cell2mat(cellfun(@size, ims(:), 'UniformOutput', false)), [], 1);
    nFrames = [(cellfun(@(x) size(x, 4), ims, 'UniformOutput', true))];
    frames = cumsum(nFrames);
    num_images = length(ims);
    blob = zeros(max_shape(1), max_shape(2), max_shape(3), frames(end), 'single');
    
    for i = 1:length(ims)
        im = ims{i};
        blob(1:size(im, 1), 1:size(im, 2), :, frames(i):frames(i)+nFrames(i)-1) = im; 
    end
end