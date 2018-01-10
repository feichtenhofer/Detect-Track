function model = ResNetX101_for_RFCN_ILSVRCvid_dilation(model)
% ResNet 101layers with OHEM training (finetuned from res3a)

model.solver_def_file        = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNeXt-101L_ILSVRCvid_dilation_OHEM_res3a', 'solver_120k160k_lr1_3.prototxt');
model.test_net_def_file      = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNeXt-101L_ILSVRCvid_dilation_OHEM_res3a', 'test.prototxt');

model.net_file               = fullfile(pwd, 'models', 'pre_trained_models', 'ResNeXt-101L', 'resnext101-32x4d-merge.caffemodel');
model.mean_image             = fullfile(pwd, 'models', 'pre_trained_models', 'ResNeXt-101L', 'mean_image');

end