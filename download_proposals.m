function download_proposals( root_path, subsets )
% Download proposals to root_path / Data / * / proposals

if nargin < 1
  root_path = get_root_path;
end
if nargin < 2
  subsets = {'DET','VID_train','VID_val'};
end

for s = subsets
  file = fullfile(root_path,'Data',char(s));
  mkdir_if_missing(file)
  if ~exist([file filesep 'RPN_proposals'],'dir')
    basefile = ['RPN_proposals_' ...
      char(s) '.zip'];
    file = fullfile(file,basefile);
    fprintf('Downloading+extracting base file: %s to ...\n%s\n', basefile,file);
    unzip(...
      ['http://ftp.tugraz.at/pub/feichtenhofer/detect-track/data/proposals/' basefile], file) ;
  end
end

