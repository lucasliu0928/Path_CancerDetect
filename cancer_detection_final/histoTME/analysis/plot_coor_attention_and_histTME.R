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
source("tme_utils.R")


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


########################################################################################################################
#Get ensemble prediction
########################################################################################################################
pred_df <- read.csv(paste0(pred_dir, "TF", format(infer_tf, nsmall = 1), "_pred_ensemble.csv"))


########################################################################################################################
#Load WSI histoTME
########################################################################################################################
tme_df_opx <- read.csv(paste0(tme_dir,"OPX_predictions_uni2.csv"))
colnames(tme_df_opx)[which(colnames(tme_df_opx) == "ID")] <- "SAMPLE_ID"
tme_df_tcga <- read.csv(paste0(tme_dir,"TCGA_PRAD_predictions_uni2.csv"))
colnames(tme_df_tcga)[which(colnames(tme_df_tcga) == "ID")] <- "SAMPLE_ID"
tme_df_nep <- read.csv(paste0(tme_dir,"Neptune_predictions_uni2.csv"))
colnames(tme_df_nep)[which(colnames(tme_df_nep) == "ID")] <- "SAMPLE_ID"
tme_df <- do.call(rbind, list(tme_df_opx,tme_df_tcga, tme_df_nep))

########################################################################################################################
#merge tme with label and pred
########################################################################################################################
tme_df <- merge(tme_df,label_df, by = "SAMPLE_ID")
tme_df <- merge(tme_df,pred_df, by = "SAMPLE_ID")

################################################################################################################
#(Slide level): TME vs MSI + and MSI- (true label)
################################################################################################################
plot_heatmap_TME(tme_df, "NEP", out_dir)
plot_heatmap_TME(tme_df, "OPX", out_dir)
plot_heatmap_TME(tme_df, "TCGA", out_dir)





################################################################################################################
#correlation between predicted MSI and features
################################################################################################################
#corrected prediction
corrected_idxes <- which(tme_df[,"majority_class"] == tme_df[,"MSI_POS"])
study_df <- tme_df[corrected_idxes,]
rownames(study_df) <- study_df[,"SAMPLE_ID"]


#Feature cols
selected_cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
                   "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
                   "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
                   "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
                   "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
                   "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
                   "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
                   "CAF", "Angiogenesis")

#study_df
study_df <- study_df[,c("mean_adj_prob_votedclass", selected_cols)]


# Compute correlation + p-values
cor_list <- lapply(selected_cols, function(f) {
  test <- suppressWarnings(
    cor.test(study_df[[f]], study_df[["mean_adj_prob_votedclass"]],
             method = "spearman", use = "pairwise.complete.obs")
  )
  data.frame(
    Feature = f,
    Correlation = test$estimate,
    P_value = test$p.value
  )
})

cor_df <- do.call(rbind, cor_list)

#mark significance ----
cor_df$Significance <- cut(
  cor_df$P_value,
  breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
  labels = c("***", "**", "*", "")
)

# Sort by correlation
cor_df <- cor_df[order(-abs(cor_df$Correlation)), ]

# ---- 3️⃣ Plot ----
p <- ggplot(cor_df, aes(x = reorder(Feature, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", aes(fill = Correlation > 0)) +
  geom_text(aes(label = Significance), vjust = -0.3, size = 5) +
  scale_fill_manual(values = c("TRUE" = "firebrick3", "FALSE" = "steelblue")) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Correlation with Predicted Risk",
    x = "Feature",
    y = "Correlation coefficient"
  ) +
  theme(legend.position = "none")

ggsave(
  filename = paste0(out_dir, "Corrected_", "histTME_predicted_risk_correlation.png"),
  plot = p,
  width = 10, height = 8, dpi = 300
)



################################################################################################################
#group comparison TRUE Y
################################################################################################################
p <- plot_feature_comparison(tme_df, "NEP" ,"MSI_POS", "MSI Status")
ggsave(paste0(out_dir,"/NEP_trueMSI_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)

p <- plot_feature_comparison(tme_df, "OPX" ,"MSI_POS", "MSI Status")
ggsave(paste0(out_dir,"/OPX_trueMSI_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)

p <- plot_feature_comparison(tme_df, "TCGA" ,"MSI_POS", "MSI Status")
ggsave(paste0(out_dir,"/TCGA_trueMSI_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)

################################################################################################################
#group comparison Predicted Y
################################################################################################################
p <- plot_feature_comparison(tme_df, "NEP" ,"majority_class", "Predicted MSI Status")
ggsave(paste0(out_dir,"/NEP_predMSI_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)

p <- plot_feature_comparison(tme_df, "OPX" ,"majority_class", "Predicted MSI Status")
ggsave(paste0(out_dir,"/OPX_predMSI_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)

p <- plot_feature_comparison(tme_df, "TCGA" ,"majority_class", "Predicted MSI Status")
ggsave(paste0(out_dir,"/TCGA_predMSI_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)



################################################################################################################
#Tile level comparison
################################################################################################################
min_max_norm <- function(indata, col_name){
  min_v <- min(indata[,col_name], na.rm = TRUE)
  max_v <- max(indata[,col_name], na.rm = TRUE)
  indata[,col_name] <-  (indata[,col_name]  - min_v)/(max_v - min_v)
  
  return(indata)
}

#Load attention
att_df <- as.data.frame(read_parquet(paste0(att_dir, "TF", format(infer_tf, nsmall = 1), "_ensemble_attention.parquet")))
cond1 <- att_df[,"majority_class"] == att_df[,"True_y"]
cond2 <- att_df[,"True_y"] == 1
true_pos_ids <- unique(att_df[which(cond1 & cond2),"SAMPLE_ID"])
all_test_ids <- unique(att_df[,"SAMPLE_ID"])
plot_tf <- 0.0

for (sample_id in true_pos_ids){
  
  df <-  att_df[which(att_df[,"SAMPLE_ID"] == sample_id),]
  df <-  df[which(df[,"tumor_fraction"] >= plot_tf),]

  pred_prob <- unique(df[,"mean_adj_prob_votedclass"])
  
  # Get min/max for normalization
  df <- min_max_norm(df,"mean_att")
  folder_id <- unique(df$FOLDER_ID)
  
  #Load slide hitTME
  file_dir <- paste0(proj_dir, "0_HistoTME/TME_Spatial/TF0.0/uni2/", sample_id, "_5fold.csv")
  tme_spatial <- read.csv(file_dir)
  tme_spatial <- tme_spatial[,-which(colnames(tme_spatial) == "X")]
  
  
  if (grepl("NEP",sample_id)==TRUE){
    cohort_name <- "Neptune"
  }else if (grepl("OPX",sample_id)==TRUE){
    cohort_name <- "OPX"
  }else{
    cohort_name <- "TCGA_PRAD"
  }
  
  
  #Load tile info
  tile_info_dir <- paste0(proj_dir, "2_cancer_detection/", cohort_name, "/IMSIZE250_OL0/", folder_id, "/ft_model/", sample_id, "_TILE_TUMOR_PERC.csv")
  tile_info     <- read.csv(tile_info_dir)
  tile_info <- tile_info %>%
    mutate(TILE_XY_INDEXES = str_remove_all(TILE_XY_INDEXES, "[()]")) %>%   # remove parentheses
    separate(TILE_XY_INDEXES, into = c("x", "y"), sep = ",", convert = TRUE)
  
  #Merge
  tme_spatial_df <- merge(tme_spatial,tile_info, by = c("x","y"))
  tme_spatial_df <- merge(tme_spatial_df,df, by = c("pred_map_location","SAMPLE_ID"))
  
  #Attention binary
  med_att <- median(tme_spatial_df[,"mean_att"])
  tme_spatial_df[,"ATT_Binary"] <- 0
  idxes <- which(tme_spatial_df[,"mean_att"] >= med_att)
  tme_spatial_df[idxes,"ATT_Binary"] <- 1
  
  p <- plot_feature_comparison(tme_spatial_df, NA ,"ATT_Binary", "MSI Attention")
  ggsave(paste0(out_dir, "/attention_TME/" ,sample_id, "_ATT_TME_comparison.png"), p, width = 18, height = 18, dpi = 300)
  
  
  #Hand feature
  if (cohort_name == "OPX"){
    hf_info_dir <- paste0(proj_dir, "6_hand_crafted_feature/", cohort_name, "/IMSIZE250_OL0/", folder_id, '/',sample_id, "_handcrafted_feature.csv")
    hf_df     <- read.csv(hf_info_dir)
    hf_df <- merge(hf_df,df, by = c("pred_map_location","SAMPLE_ID"))
    med_att <- median(hf_df[,"mean_att"])
    hf_df[,"ATT_Binary"] <- 0
    idxes <- which(hf_df[,"mean_att"] >= med_att)
    hf_df[idxes,"ATT_Binary"] <- 1
    selected_features <- colnames(hf_df)[12:325]
    
    
    p <- plot_hffeature_comparison(hf_df, NA ,"ATT_Binary", "MSI Attention",selected_features, topK = 5, alpha = 0.05)
    ggsave(paste0(out_dir, "/attention_HF/" ,sample_id, "_HF_TME_comparison.png"), p, width = 10, height = 8, dpi = 300)
   }
}


# Data in two numeric vectors
women_weight <- c(38.9, 61.2, 73.3, 21.8, 63.4, 64.6, 48.4, 48.8, 48.5)
men_weight <- c(67.8, 60, 63.4, 76, 89.4, 73.3, 67.3, 61.3, 62.4) 
# Create a data frame
my_data <- data.frame( 
  group = rep(c("Woman", "Man"), each = 9),
  weight = c(women_weight,  men_weight)
)

