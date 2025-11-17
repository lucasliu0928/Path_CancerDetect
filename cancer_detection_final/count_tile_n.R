library(ggplot2)
library(tidyr)
library(pheatmap)
library(dplyr)
library(RColorBrewer)
library(ggplotify)  # to convert pheatmap -> ggplot
library(dendsort)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(arrow)
#source("~/histoTME/tme_utils.R")


########################################################################################################################
#User input
########################################################################################################################
infer_tf <- 0.0
train_folder <- "MSI_train_restrict_to_0.9sampling"
outcome_folder <- "pred_out_100125_test"

########################################################################################################################
#DIR
########################################################################################################################
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
label_dir <- paste0(proj_dir, "3C_labels_train_test/")
pred_dir <- paste0(proj_dir, "pred_out_100125_test/", train_folder,"/ensemble_prediction/")
tme_dir   <- paste0(proj_dir, "0_HistoTME/TME/TF0.0/")
att_dir <- file.path(proj_dir, outcome_folder,train_folder, "ensemble_prediction","attention/")
out_dir <- paste0(pred_dir,"/histoTME/")

########################################################################################################################
#Load label
########################################################################################################################
label_df1 <- read.csv(paste0(label_dir, "OPX/TFT0.0/train_test_split.csv"))
label_df2 <- read.csv(paste0(label_dir, "TCGA_PRAD/TFT0.0/train_test_split.csv"))
label_df3 <- read.csv(paste0(label_dir, "Neptune/TFT0.0/train_test_split.csv"))
label_df <- do.call(rbind, list(label_df1, label_df2, label_df3))

#Keep test indexes
#opx and tcga test
cond1 <- label_df[,"TRAIN_OR_TEST"] == "TEST"
cond2 <- grepl("TCGA-|OPX_",label_df[,"SAMPLE_ID"])
#nep all
cond3 <- grepl("NEP-",label_df[,"SAMPLE_ID"])
idxes <- which((cond1 & cond2) | cond3)
label_df <- label_df[idxes,]


#IDs
ids <- unique(label_df$SAMPLE_ID)


cond1 <- label_df[,"TRAIN_OR_TEST"] == "TRAIN"
cond2 <- grepl("NEP-",label_df[,"SAMPLE_ID"])
train_ids <- label_df[which(cond1 & !cond2),"SAMPLE_ID"] #525
length(train_ids)

#opx and tcga test
cond1 <- label_df[,"TRAIN_OR_TEST"] == "TEST"
cond2 <- grepl("TCGA-|OPX_",label_df[,"SAMPLE_ID"])
#nep all
cond3 <- grepl("NEP-",label_df[,"SAMPLE_ID"])
idxes <- which((cond1 & cond2) | cond3)
test_ids <- label_df[idxes,"SAMPLE_ID"]
length(test_ids)

################################################################################################################
#Tile level comparison
################################################################################################################
list.dirs("path/to/location")

count_list <- list()
for (i in 1:length(test_ids)){
  if (i %% 50 == 0){
    print(i)
  }
  sample_id <- test_ids[i]
  
  
  if (grepl("NEP",sample_id)==TRUE){
    cohort_name <- "Neptune"
    folder_id <- sample_id
  }else if (grepl("OPX",sample_id)==TRUE){
    cohort_name <- "OPX"
    folder_id <- sample_id
  }else{
    cohort_name <- "TCGA_PRAD"
    tile_info_path <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/1_tile_pulling/TCGA_PRAD/IMSIZE250_OL0"
    file_names <- list.files(path = tile_info_path, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
    idx <- grep(sample_id, file_names)
    folder_id <-  strsplit(file_names[idx],split = "/")[[1]][10]
  }
  
  
  
  
  #Load tile info
  tile_info_dir <- paste0(proj_dir, "2_cancer_detection/", cohort_name, "/IMSIZE250_OL0/", folder_id, "/ft_model/", sample_id, "_TILE_TUMOR_PERC.csv")
  tile_info     <- read.csv(tile_info_dir)
  tile_info <- tile_info %>%
    mutate(TILE_XY_INDEXES = str_remove_all(TILE_XY_INDEXES, "[()]")) %>%   # remove parentheses
    separate(TILE_XY_INDEXES, into = c("x", "y"), sep = ",", convert = TRUE)
  
  thresholds <- seq(0.1, 0.9, by = 0.1)
  counts <- sapply(thresholds, function(t) sum(tile_info$TUMOR_PIXEL_PERC >= t))
  counts_df <- data.frame(threshold = thresholds, count = counts)
  counts_df['SAMPLE_ID'] <- sample_id
  count_list[[i]] <- counts_df
}

count_df<- do.call(rbind,count_list)
median(count_df$count)

library(dplyr)

df_summary <- count_df %>%
  group_by(threshold) %>%
  summarise(
    median_count = median(count),
    mean_count = mean(count)
  )

print(df_summary)

write_csv(df_summary,"df_summary.csv")
