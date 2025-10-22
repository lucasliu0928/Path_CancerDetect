library(ggplot2)
library(dplyr)
library(tidyr)
library(pheatmap)
library(dplyr)
library(arrow)

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
label_dir <- file.path(proj_dir, "3C_labels_train_test/")
pred_dir <-  file.path(proj_dir, "pred_out_100125_test", train_folder, "ensemble_prediction/")

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
label_df <- label_df[,-which(colnames(label_df) == "X")]

#IDs
ids <- unique(label_df$SAMPLE_ID)


########################################################################################################################
#Get ensemble prediction
########################################################################################################################
pred_df <- read.csv(paste0(pred_dir, "TF", format(infer_tf, nsmall = 1), "_pred_ensemble.csv"))
pred_df <- pred_df[,-which(colnames(pred_df) == "X")]


########################################################################################################################
#load attention
#Only get attention from the emsembel folds
########################################################################################################################
att_dir  <- file.path(proj_dir, "pred_out_100125_test" , train_folder,"/")

all_attention <- list()
for (i in 1:length(ids)){
  cur_id <- ids[i]
  #Load pred df to get voted folds
  cur_pred_df <- pred_df[which(pred_df[,"SAMPLE_ID"] == cur_id),]
  cur_folds <- strsplit(cur_pred_df[,"folds_voted"], split = ',')[[1]]
  
  fold_att <- list()
  for (j in 1:length(cur_folds)){
    fold <- cur_folds[j]
    data_dir <- file.path(att_dir,fold, paste0("predictions_", format(infer_tf, nsmall = 1)), "attention") 
    cur_df <- read.csv(file.path(data_dir, paste0(cur_id, "_att.csv")))
    coords <- do.call(rbind, strsplit(gsub("[()]", "", cur_df$pred_map_location), ",\\s*"))
    coords <- apply(coords, 2, as.numeric)
    colnames(coords) <- c("x1","x2","y1","y2")
    cur_df <- cbind(cur_df, coords)
    cur_df['SAMPLE_ID'] <- cur_id
    cur_df['FOLD'] <- as.integer(gsub("FOLD","",fold))
    fold_att[[j]] <- cur_df
  }
  attention_df <- do.call(rbind, fold_att)
  all_attention[[i]] <- attention_df
}
  
all_attention_df <- do.call(rbind, all_attention)
all_attention_df <- all_attention_df[,-which(colnames(all_attention_df) == "X")]



#ensemble attention 
ensemble_att_df <- all_attention_df %>%
  group_by(SAMPLE_ID, pred_map_location, x1, x2, y1, y2, tumor_fraction) %>%
  summarise(
    mean_att = mean(att, na.rm = TRUE),
    n_folds  = n_distinct(FOLD),
    .groups = "drop"
  )

################################################################################################################
#Merge  label and Attention
################################################################################################################
comb_df <- merge(ensemble_att_df,pred_df,  by = "SAMPLE_ID", all = TRUE)
comb_df <- merge(comb_df,label_df,  by = "SAMPLE_ID", all = TRUE)


write_parquet(comb_df, file.path(proj_dir, outcome_folder,train_folder, "ensemble_prediction","attention", paste0("TF",format(infer_tf, nsmall = 1), "_ensemble_attention.parquet")))
