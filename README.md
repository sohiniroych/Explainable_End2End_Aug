# Explainable and End-to-end Image Data Augmentation for Semantic Segmentation using U-net
This code base augments image data sets to enable semantic segmentation using the U-net model. Explanation of an explainable end-to-end pipeline is in https://www.youtube.com/watch?v=44QzNcU0k2Y&t=1s

This codebase marks the end of the 10-week program "Build Your Own Research Internship in AI, 2020" https://www.youtube.com/watch?v=1kiPy2tvECs&list=PLQKflBw-kPeecjC345saTF2YrfImciCrr
Google drive for weekly materials: https://drive.google.com/drive/u/2/folders/1pYMjFe4bxH5qawFYl4NGM4C6yQIVDOCy

## Pre-requisite Python Packages
Video Explaning coding environment: https://www.youtube.com/watch?v=d7ktNAVHo5E&t=453s
* tensorflow>=1.6<2.0
* keras>2.0
* opencv-python
* numpy
* pydot
* graphviz
* matplotlib 
* pillow

## Proposed System Diagram
![System Diagram](imgs/BYORI_1.png)
This code base improves the explainability and generalizability of the image-tiling data augementation strategy in [1] in the following two ways:
1. The end-to-end data augmentation method in [1] generates 190,000 random tiled sub-images from a set of 20 images. The proposed method is sensitive to the random number gennerator and can over or under-represent certain regions of the image. Generating non-overlapping tiles (in this code base) reduces such imbalanced image representations and ensures generalizability across data sets.
1. Method in [1] requires significant compute power to train 3-4 u-net layers for 100 epochs. The proposed method significantly reduces compute since comparable segmentation accuracy is achieved by the proposed system in 50 epochs.
Other contributions of the proposed code base are:
1. Hyper-parameters selected for method [2] to zoom into retinal regions.
1. Batch normalization and dropout added to implementations of [1] and [2] for quicker convergence.
1. U-net model for depth 3 and 4 generalized across method [1] and [2]. 
1. Hyperparamaterization and Ablation Study performed.



