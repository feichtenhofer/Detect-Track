function startup()
% startup()
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

    curdir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(curdir, 'utils')));
    addpath(genpath(fullfile(curdir, 'functions')));
    addpath(genpath(fullfile(curdir, 'bin')));
    addpath(genpath(fullfile(curdir, 'experiments')));
    addpath(genpath(fullfile(curdir, 'imdb')));


    caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');
    
        caffe_path = fullfile('H:\RFCN\caffe-rfcn', 'matlab');

    if ~ispc
      caffe_path = fullfile('~', 'git', 'caffe-rfcn', 'matlab');
    end
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from %s; See README.md', caffe_path);
    end
    
    addpath(genpath(caffe_path));


    fprintf('rfcn startup done\n');
end
