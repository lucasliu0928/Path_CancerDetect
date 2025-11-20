
library(ggplot2)
library(dplyr)
library(tidyr)
library(rlang)





############################################################################################################
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


hf_att_list <- list()
ct <- 0
for (sample_id in true_pos_ids){
  ct <- ct + 1
  df <-  att_df[which(att_df[,"SAMPLE_ID"] == sample_id),]
  df <-  df[which(df[,"tumor_fraction"] >= plot_tf),]
  
  pred_prob <- unique(df[,"mean_adj_prob_votedclass"])
  
  # Get min/max for normalization
  df <- min_max_norm(df,"mean_att")
  
  folder_id <- unique(df$FOLDER_ID)
  

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
    hf_att_list[[ct]] <- hf_df 
  }
}


#Normlized att for true pos
hf_att_df <- do.call(rbind,hf_att_list)

df <- hf_att_df


#Group compare
library(dplyr)
library(ggplot2)
library(ggpubr)


# Median splits 
tumor_thr <- mean(df$TUMOR_PIXEL_PERC, na.rm = TRUE) #0.27
att_thr   <- mean(df$mean_att, na.rm = TRUE) #0.41

df <- df %>%
  mutate(
    tumor_group = ifelse(TUMOR_PIXEL_PERC >= tumor_thr, "High Tumor", "Low Tumor"),
    att_group   = ifelse(mean_att >= att_thr, "High Attention", "Low Attention"),
    quadrant    = paste(tumor_group, att_group, sep = " & ")
  )

table(df$quadrant)

# list of feature 
feature_list <- selected_features

# pairwise comparisons to show on plot
comparisons <- list(
  c("High Tumor & High Attention", "High Tumor & Low Attention"),
  c("High Tumor & High Attention", "Low Tumor & High Attention"),
  c("Low Tumor & Low Attention",  "High Tumor & Low Attention"),
  c("Low Tumor & Low Attention",  "Low Tumor & High Attention")
)

# directory to save PNGs (create if needed)
dir.create("hf_feature_quadrant_plots", showWarnings = FALSE)

for (feat in feature_list) {
  
  p <- ggplot(df, aes(x = quadrant, y = .data[[feat]], fill = quadrant)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
    stat_compare_means(
      comparisons = comparisons,
      method = "wilcox.test",
      label  = "p.signif",
      step.increase = 0.08
    ) +
    labs(
      #title = paste("Quadrant Comparison for", feat),
      #x = "Quadrant",
      y = feat
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 25, hjust = 1)
    )
  
  ggsave(
    filename = file.path("hf_feature_quadrant_plots",
                         paste0("quadrant_", feat, ".png")),
    plot = p,
    width = 7, height = 5, dpi = 300
  )
}


quad_A <- "High Tumor & High Attention"
quad_B <- "Low Tumor & Low Attention"

## 2. Per-quadrant summary (means, medians, sd, n) ---------------------

feature_quadrant_summary <- lapply(feature_list, function(feat) {
  df %>%
    group_by(quadrant) %>%
    summarise(
      feature     = feat,
      mean_value  = mean(.data[[feat]], na.rm = TRUE),
      median_value= median(.data[[feat]], na.rm = TRUE),
      sd_value    = sd(.data[[feat]], na.rm = TRUE),
      n           = sum(!is.na(.data[[feat]])),
      .groups     = "drop"
    )
}) %>%
  bind_rows()

## 3. Per-feature stats: diff + p-values -------------------------------

feature_stats <- lapply(feature_list, function(feat) {
  tmp <- df %>%
    select(quadrant, !!sym(feat)) %>%
    rename(value = !!sym(feat)) %>%
    filter(!is.na(value))
  
  # means per quadrant
  means <- tmp %>%
    group_by(quadrant) %>%
    summarise(mean_value = mean(value, na.rm = TRUE),
              .groups = "drop")
  
  mean_A <- means$mean_value[means$quadrant == quad_A]
  mean_B <- means$mean_value[means$quadrant == quad_B]
  
  diff_AB <- if (length(mean_A) == 1 && length(mean_B) == 1) {
    mean_A - mean_B
  } else {
    NA_real_
  }
  
  # Wilcoxon test between A and B
  xA <- tmp$value[tmp$quadrant == quad_A]
  xB <- tmp$value[tmp$quadrant == quad_B]
  
  p_AB <- if (length(xA) > 1 && length(xB) > 1) {
    suppressWarnings(wilcox.test(xA, xB)$p.value)
  } else {
    NA_real_
  }
  
  # Kruskal-Wallis across all quadrants
  p_kruskal <- if (length(unique(tmp$quadrant)) > 1) {
    suppressWarnings(kruskal.test(value ~ quadrant, data = tmp)$p.value)
  } else {
    NA_real_
  }
  
  tibble(
    feature     = feat,
    mean_A      = ifelse(length(mean_A) == 1, mean_A, NA_real_),
    mean_B      = ifelse(length(mean_B) == 1, mean_B, NA_real_),
    diff_AB     = diff_AB,         # mean(High Tumor & High Att) - mean(Low Tumor & Low Att)
    p_AB        = p_AB,            # Wilcoxon A vs B
    p_kruskal   = p_kruskal        # Kruskal across all 4 quadrants
  )
}) %>%
  bind_rows()

## 4. Top 5 features by absolute difference ----------------------------

top5_features <- feature_stats %>%
  arrange(desc(abs(diff_AB))) %>%
  slice(1:5)

top5_features

## 5. Export to CSV -----------------------------------------------------

write.csv(feature_quadrant_summary,
          "hf_feature_quadrant_means.csv",
          row.names = FALSE)

write.csv(feature_stats,
          "hf_feature_quadrant_stats.csv",
          row.names = FALSE)

write.csv(top5_features,
          "hf_top5_features_by_diff.csv",
          row.names = FALSE)
