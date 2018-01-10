function [ imdb ] = imdb_set_default_paths( imdb, root_path, set, mod )

imdb.image_dir = [root_path '/Data/' mod '/' set];
imdb.details.devkit_path = [root_path '/devkit'];
imdb.details.root_dir = [root_path];
imdb.details.bbox_path = [root_path '/Annotations/' mod '/' set];

imdb.image_at = @(i) ...
    fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);

end

