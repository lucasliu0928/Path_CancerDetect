# Deep Learning for Gene Pathway Mutation Prediction for Prostate Cancer 

Predict Gene Mutation from H&E WSI Image for Prostate Cancer

## Dependencies
* For cancer detection
   - Python 3.8.20  [GCC 13.3.0]
   - cv2 == 4.10.0
   - fastai == 2.7.10
   - torch == 2.4.1+cu121
   - torchvision == 0.19.1+cu121
   - openslide == 1.3.1
   - histomicstk == 1.3.14 (python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels)
* Python 3.11.11 [GCC 13.3.0]

## Clone the repository
```
git clone https://github.com/lucasliu0928/Path_CancerDetect.git
cd Path_CancerDetect
```
## Create and activate a virtual environment
```
conda env create -f paimg9.yml
conda activate paimg9
```
### 🧩 Executing Program

#### Step 1: Extract Tiles from WSI
This step processes the Whole Slide Image (WSI) into tiles (only kept tiles with tissue coverage > 0.9 and white space < 0.9) and generates the following output files:

- **`sampleid_tiles.csv`**    — Metadata of the extracted tiles containing white space % and tissue coverage %
- **`sampleid_low-res.png`**  — Low-resolution WSI image  
- **`sampleid_tissue.png`**   — Detected tissue mask image
- **`sampleid_tissue.json`**  — Tissue region annotations  

```
conda activate paimg9
python3 -u 1_extract_patches_fixed-res.py  --cohort_name TCGA_PRAD --pixel_overlap 0
```

#### Step 2: Run Cancer Detection Model on Extracted Tiles
This step applies a trained cancer detection model to the extracted tiles and generates the following output files:

- **`sampleid_TILE_TUMOR_PERC.csv`** — Tile-level cancer probability and metadata  
- **`sampleid_cancer_prob.jpeg`** — Cancer prediction probability heatmap  
- **`TILE_@#_X#Y#_TF#.png`** — Top 5 tiles with the highest tumor fraction  
  - `@#`: Magnification level  
  - `X#`, `Y#`: Tile coordinates  
  - `TF`: Tumor fraction score  
- **`sampleid_cancer.json`** — Cancer region annotations
  
```
conda activate paimg9
python3 -u 2_cancer_inference_fixed-res.py --cohort_name TCGA_PRAD  --fine_tuned_model True --pixel_overlap 0 
```

#### 🧬 Step 3: Run Foundation Models to Extract Tile Embeddings
This step uses selected foundation models to compute tile-level embeddings.  
**Available models:** `retccl`, `uni1`, `uni2`, `prov_gigapath`, `virchow2`.
**Generated output:**
- **`sampleid/features_alltiles_modelname.h5`** — Tile-level embedding features  
  - `modelname`: One of `retccl`, `uni1`, `uni2`, `prov_gigapath`, `virchow2`
```
conda activate paimg9
python3 -u 4_get_feature.py --cohort_name TCGA_PRAD --pixel_overlap 0 --fine_tuned_model True --feature_extraction_method uni2
```

#### Step 4: Run Foundation Models to extract Tile Embeddings


## Authors
Lucas J. Liu 
jliu6@fredhutch.org

## Version History
* 0.1
    * Initial Release


