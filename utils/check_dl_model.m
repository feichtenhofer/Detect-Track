function check_dl_model( model )
% Download model file
if ~exist(model,'file')
  [~, baseModel] = fileparts(model);
  fprintf('Downloading base model file: %s ...\n', baseModel);
  mkdir_if_missing(fileparts(model)) ;
  urlwrite(...
  ['http://ftp.tugraz.at/pub/feichtenhofer/detect-track/models/' baseModel '.caffemodel'], ...
    model) ;
end

end

