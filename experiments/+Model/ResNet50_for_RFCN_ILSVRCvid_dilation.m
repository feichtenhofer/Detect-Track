function model = ResNet50_for_RFCN_ILSVRCvid_dilation(model)
% ResNet 50layers with OHEM training (finetuned from res3a)

model.solver_def_file        = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_ILSVRCvid_dilation_OHEM_res3a', 'solver_160k240k_lr1_3.prototxt');
model.test_net_def_file      = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_ILSVRCvid_dilation_OHEM_res3a', 'test.prototxt');

model.net_file               = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
model.mean_image             = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image');

end