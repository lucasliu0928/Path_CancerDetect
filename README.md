# Deep Learning for Gene Pathway Mutation Prediction for Prostate Cancer 

Predict Gene Mutation from H&E WSI Image for Prostate Cancer

## Clone the repository
```
git clone https://github.com/lucasliu0928/Path_CancerDetect.git
cd Path_CancerDetect
```
## Create and activate a virtual environment

#### For Cancer Detection
```bash
conda env create -f paimg9.yml
conda activate paimg9
```

#### For Mutation Prediction:
```
conda env create -f mil.yml
conda activate mil
```

```
python3 -m venv acmil
pip install -r requirements_acmil.txt
'''


#### For Hand-crafted feature extraction:
```
python3 -m venv hf
pip install -r requirements_hf.txt
'''

#### For HistoTME:
```
conda env create -f histoTME.yml
conda activate histoTME
```
## 📋 Overview

This pipeline enables a step-by-step workflow for histopathology data analysis:

## 🧬 I. Mutation Prediction Pipeline

1. 🧱 **Extract Tiles** — Split Whole Slide Images (WSIs) into smaller image tiles  
2. 🔬 **Run Cancer Detection** — Identify and quantify tumor regions across tiles  
3. 🧠 **Generate Embeddings** — Extract tile-level representations using foundation models  
4. ⚙️ **Train Mutation Prediction Model** — **🚧 TODO:** Develop supervised learning workflow  
5. 🧭 **Run Inference for Mutation Prediction** — **🚧 TODO:** Apply trained model to unseen data  
6. 📊 **Evaluate Model Performance** — **🚧 TODO:** Compute metrics such as ROC-AUC, accuracy, and F1-score  

---

## 🧩 II. Additional Analyses

7. 🌿 **Analyze Tumor Microenvironment (TME)** — Perform TME profiling using HistoTME  
8. 🧰 **Analyze Hand-Crafted Features** — **🚧 TODO:** Integrate feature extraction pipeline  


   
### 🧬 I. Mutation Prediction Pipeline

#### 🧱 Step 1: Extract Tiles from WSI
This step processes the Whole Slide Image (WSI) into tiles (only kept tiles with tissue coverage > 0.9 and white space < 0.9).
**Generated output:**:

- **`sampleid_tiles.csv`**    — Metadata of the extracted tiles containing white space % and tissue coverage %
- **`sampleid_low-res.png`**  — Low-resolution WSI image  
- **`sampleid_tissue.png`**   — Detected tissue mask image
- **`sampleid_tissue.json`**  — Tissue region annotations  

```
conda activate paimg9
cd cancer_detection_final
python3 -u 1_extract_patches_fixed-res.py  --cohort_name TCGA_PRAD --pixel_overlap 0
```

#### 🔬 Step 2: Run Cancer Detection Model on Extracted Tiles
This step applies a trained cancer detection model to the extracted tiles.
**Generated output:**:

- **`sampleid_TILE_TUMOR_PERC.csv`** — Tile-level cancer probability and metadata  
- **`sampleid_cancer_prob.jpeg`** — Cancer prediction probability heatmap  
- **`TILE_@#_X#Y#_TF#.png`** — Top 5 tiles with the highest tumor fraction  
  - `@#`: Magnification level  
  - `X#`, `Y#`: Tile coordinates  
  - `TF`: Tumor fraction score  
- **`sampleid_cancer.json`** — Cancer region annotations
  
```
conda activate paimg9
cd cancer_detection_final
python3 -u 2_cancer_inference_fixed-res.py --cohort_name TCGA_PRAD  --fine_tuned_model True --pixel_overlap 0 
```

#### 🧠 Step 3: Run Foundation Models to Extract Tile Embeddings
This step uses selected foundation models to compute tile-level embeddings. 
**Available models:** `retccl`, `uni1`, `uni2`, `prov_gigapath`, `virchow2`.
**Generated output:**

- **`sampleid/features_alltiles_modelname.h5`** — Tile-level embedding features  
  - `modelname`: One of `retccl`, `uni1`, `uni2`, `prov_gigapath`, `virchow2`
    
```
conda activate paimg9
cd cancer_detection_final
python3 -u 4_get_feature.py --cohort_name TCGA_PRAD --pixel_overlap 0 --fine_tuned_model True --feature_extraction_method uni2
```


### 🧩 II. Additional Analyses (https://github.com/spatkar94/HistoTME)
#### Step 1: Reformat data for HistoTME
This step generate input data for running HistoTME
**Available models:** Please refer to their official website for available foundation models
**Generated output:**
- **`sampleid_features.hdf5`** — Tile-level embedding features with features and coords
  
```
conda activate histoTME
cd cancer_detection_final/histoTME
python3 0_reformat_data.py --fe_method uni2 --cohort_name TCGA_PRAD --tumor_frac 0.0
```

#### Step 2A: Run inference for a bulk of slides
This step runs the HistoTME model to compute slide-level (bulk) signatures.

```
conda activate histoTME
cd /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/HistoTME_regression
python3 predict_bulk.py  --cohort TCGA_PRAD --h5_folder /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/TCGA_PRAD/IMSIZE250_OL0/uni2 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME/TF0.0/ --num_workers 10 --embed uni2 
```

#### Step 2B: Run spatial inference for each slide
This step runs HistoTME in spatial mode to compute tile-level (spatial) signatures.

```
conda activate histoTME
cd /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/HistoTME_regression
python3 predict_spatial.py  --h5_path /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/model_data/TF0.0/TCGA_PRAD/IMSIZE250_OL0/uni2/TCGA_PRAD_XXXX_features.hdf5 --chkpts_dir /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/other_model_code/HistoTME/local_dir/checkpoints  --save_loc /fh/fast/etzioni_r/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/TME_Spatial/TF0.0/ --num_workers 10 --embed uni2 
```

Note on Modifications :
I added the following code to "data.py" in "HistoTME_regression folder" to make it easier to match all embedding model names and the names in the arguments for python predict_spatial.py [-h] [--h5_path H5_PATH] [--chkpts_dir CHKPTS_DIR] [--num_workers NUM_WORKERS]
[--embed EMBED] [--save_loc SAVE_LOC]

```
elif 'uni1' in embedding_paths[0]:
    embedding_dim = 1024
elif 'uni2' in embedding_paths[0]:
    embedding_dim = 1536
```

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
    
## Authors
Lucas J. Liu 
jliu6@fredhutch.org

## Version History
* 0.1
    * Initial Release


