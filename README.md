# Deep Learning for Gene Pathway Mutation Prediction for Prostate Cancer 

Predict Gene Mutation from H&E WSI Image for Prostate Cancer

## Description

## Clone the repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
## Create and activate a virtual environment
```
conda env create -f paimg9.yml
```
### Dependencies
* For cancer detection
   - Python 3.8.20  [GCC 13.3.0]
   - cv2 == 4.10.0
   - fastai == 2.7.10
   - torch == 2.4.1+cu121
   - torchvision == 0.19.1+cu121
   - openslide == 1.3.1
   - histomicstk == 1.3.14 (python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels)
* Python 3.11.11 [GCC 13.3.0]

### Installing

### Executing program

* Step1: Extract tiles from WSI: 
```
conda activate paimg9
python3 -u 1_extract_patches_fixed-res.py  --cohort_name PrECOG --pixel_overlap 0
```


## Authors
Lucas J. Liu 
jliu6@fredhutch.org

## Version History
* 0.1
    * Initial Release


