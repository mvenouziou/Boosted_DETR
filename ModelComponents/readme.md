This folder contains the full object detection model as well as a new pre-trainer model based on multi-instance classification. The model has been modularized, with component building blocks included as individual python files.

It includes an ipython notebook implementing both the full and pre-trainer models on the COCO dataset.

The folder also contains custom data download / import tools to download and prepare COCO and modified COCO format datasets as a Pandas dataframe, and a TF Dataset pipeline with image augmentations. 
