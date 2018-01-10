function model = ResNet50_for_DetectTrack_ILSVRCvid(model)
% ResNet 50 layers D&T with OHEM training (finetuned from res3a)

model.solver_def_file        = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_ILSVRCvid_corr', 'solver_160k240k_lr1_4.prototxt');
model.test_net_def_file      = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_ILSVRCvid_corr', 'test_track.prototxt');

model.net_file               = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-D-ilsvrc-vid.caffemodel');
model.mean_image             = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image');

end