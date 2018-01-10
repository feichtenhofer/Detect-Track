function model = ResNeXt101_for_DetectTrack_ILSVRCvid(model)
% ResNet 101 layers D&T with OHEM training (finetuned from res3a)

model.solver_def_file        = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNeXt-101L_ILSVRCvid_corr', 'solver_160k240k_lr1_4.prototxt');
model.test_net_def_file      = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNeXt-101L_ILSVRCvid_corr', 'test_track.prototxt');

model.net_file               = fullfile(pwd, 'models', 'pre_trained_models', 'ResNeXt-101L', 'ResNeXt101-D-ilsvrc-vid.caffemodel');
model.mean_image             = fullfile(pwd, 'models', 'pre_trained_models', 'ResNeXt-101L', 'mean_image');

end