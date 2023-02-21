Code repository for my bachelor graduate design.

In this repository, I add some modifications based on [PROXD](https://github.com/mohamedhassanmus/prox) to improve the performance of 3D human reconstruction from a single RGBD image.


## Features
In the original implementation, the depth loss is defined as the Chamfer Distance between the human scan and human vertices. 

1. I find that there are many noise points in the raw scan while visualizing the data. So I drop the noise points using DBSCAN clustering algorithm in advance.

2. Fitting the whole body to the scan is a little rough. So I firstly parse the scan and then fit the body to the scan by part.

3. Changing the weights of the losses can also improve the results on some samples. However, this is not a general practice.

These modifications together help improve the fitting speed and accurancy.


## Install
This code respository is mainly based on [PROX](https://github.com/mohamedhassanmus/prox), please refer to this repository for the installation of the environment.


## Data
You need to download the following data to start the experiment:

- [PROX](https://prox.is.tue.mpg.de/) dataset 
- Some essential data from [SMPL-X](https://github.com/vchoutas/smplx)

Then you need to specify the data loactions in ```cfg_files/PROXD-test.yaml```.


## Run
Please refer to ```cmd.txt``` for how to run the code.


## Reference
The majority of ```prox``` repository is borrowed from [PROX](https://github.com/mohamedhassanmus/prox). The code in ```smpl``` dierectory is from [SMPL](https://github.com/CalciferZh/SMPL). I made some comments to help me understand the SMPL parametric human model. I also use [Self Correction for Human Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to parse human from RGB images.

These code help me learn how to fit a parametric human model to a single RGBD image. Thank these authors for their great work.