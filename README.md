# RHDHand
Implementation of paper "Hand Pose Estimation via Latent 2.5 D Heatmap Regression" for 3D pose estimation of RGB hand pose dataset RHD

About caffe experiments:

1. RHDLatentHMMap

   backbone network: ShuffleNet Model size: 4.33 MB root-relative 3D error: 25mm

2. RHDLatentHMMapRawPaper
   
   backbone network: original conv-deconv network in paper Model size: 86.2 MB root-relative 3D error: 33mm


@article{iqbal2018hand,
  title={Hand Pose Estimation via Latent 2.5 D Heatmap Regression},
  author={Iqbal, Umar and Molchanov, Pavlo and Breuel, Thomas and Gall, Juergen and Kautz, Jan},
  journal={arXiv preprint arXiv:1804.09534},
  year={2018}
}
