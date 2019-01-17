# RHDHand
Implementation of paper "Hand Pose Estimation via Latent 2.5 D Heatmap Regression" for 3D pose estimation of RGB hand pose dataset RHD

## This page will be updated real soon.

Preprocessed dataset has been uploaded to Google Drive at https://drive.google.com/file/d/1ahDGxYb6BmzQxRU_juv_QDFLbpHPMy29/view (~8GB)

Preprocessing VS Code is VisRHDAnnotation.cpp

See details about derivation of formula in ppt.

About caffe experiments:

1. RHDLatentHMMap

   backbone network: ShuffleNet Model size: 4.33 MB root-relative 3D error: 25mm

2. RHDLatentHMMapRawPaper
   
   backbone network: original conv-deconv network in paper Model size: 86.2 MB root-relative 3D error: 33mm

About caffe layers:

1. deep_hand_model_calc_norm_scale_layer

   calculates normalized hand scale ("s" in raw paper)
   
2. deep_hand_model_solve_global_hand_scale_layer

   predicts global hand scale during inference
   
3. deep_hand_model_solve_scale_normalized_global_location_layer

   recovers scale normalized global joints from scale normalized root-relative 3d joints (just by adding wrist location)

4. deep_hand_model_solve_scale_normalized_global_z_root_layer

   predicts global depth (z) of root: wrist joint
   
See usage in prototxt by searching keywords like "DeepHandModelSolveGlobalHandScale" "DeepHandModelSolveScaleNormalizedGlobalLocation" ...

@article{iqbal2018hand,
  title={Hand Pose Estimation via Latent 2.5 D Heatmap Regression},
  author={Iqbal, Umar and Molchanov, Pavlo and Breuel, Thomas and Gall, Juergen and Kautz, Jan},
  journal={arXiv preprint arXiv:1804.09534},
  year={2018}
}
