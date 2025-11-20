
library(ggplot2)
library(dplyr)
library(tidyr)
library(rlang)


min_max_norm <- function(indata, col_name){
  min_v <- min(indata[,col_name], na.rm = TRUE)
  max_v <- max(indata[,col_name], na.rm = TRUE)
  indata[,col_name] <-  (indata[,col_name]  - min_v)/(max_v - min_v)
  
  return(indata)
}


############################################################################################################
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
histo_data_dir <- paste0(proj_dir, "0_HistoTME/TME_Spatial/TF0.0/uni2/")

value_col <- "Checkpoint_inhibition" #Checkpoint_inhibition, T_cells

train_folder <- "MSI_train_restrict_to_0.9sampling"
outcome_folder <- "pred_out_100125_test"
infer_tf <- "0.0"
att_dir <- file.path(proj_dir, outcome_folder,train_folder, "ensemble_prediction","attention/")

#Load att
att_df <- as.data.frame(read_parquet(paste0(att_dir, "TF", format(infer_tf, nsmall = 1), "_ensemble_attention.parquet")))




ids1 <- c("OPX_129","OPX_167","OPX_207","OPX_198","OPX_216","OPX_263")
ids2 <- c("TCGA-HC-7747-01Z-00-DX1.1fd2554f-29b9-4a9e-9fe1-93a1bc29f319","TCGA-XK-AAIW-01Z-00-DX1.31B98849-4AA9-4AC1-A697-A0C4165976B5")
ids3 <- c("NEP-045PS1-1_HE_MH_03252024",
         "NEP-076PS1-1_HE_MH_03282024",
         "NEP-126PS1-1_HE_MH06032024",
         "NEP-159PS1-1_HE_MH_06102024",
         "NEP-212PS1-8_HE_MH_06282024",
         "NEP-212PS2_HE_MH_06282024",
         "NEP-280PS2-1_HE_MH_02072025")
ids4 <- c("NEP-346PS1-1_HE_MH_03202025",
         "NEP-377PS1-1_HE_MH_03212025")
ids <- c(ids1,ids2,ids3, ids4)
ct <- 0
comb_df_list <- list()
for (sample_id in ids) {
  ct <- ct+ 1
  
  if (grepl("OPX",sample_id)){
    cohort_name <- "OPX"
  }else if(grepl("TCGA",sample_id)){
    cohort_name <- "TCGA_PRAD"
  }else{
    cohort_name <- "Neptune"
  }
  cancer_data_dir <- paste0(proj_dir, "2_cancer_detection/", cohort_name, "/IMSIZE250_OL0/")
#--- Read data
df <- read.csv(paste0(histo_data_dir, sample_id, "_5fold.csv"))
if (ncol(df) > 0) df <- df[, -1, drop = FALSE]  # drop first column if it's an index
# basic guard rails
if (!all(c("x", "y") %in% names(df))) {
  stop("Input CSV must contain columns named 'x' and 'y'. Found: ",
       paste(names(df), collapse = ", "))
}
if (!value_col %in% names(df)) {
  stop("Column '", value_col, "' not found. Available numeric columns: ",
       paste(names(df)[vapply(df, is.numeric, logical(1))], collapse = ", "))
}

# tile info
if (cohort_name == "TCGA_PRAD") {
  fold_name_df <- read.csv("/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/3C_labels_train_test/TCGA_PRAD/TFT0.0/train_test_split.csv")
  
  f_name <- fold_name_df$FOLDER_ID[which(fold_name_df$SAMPLE_ID == sample_id)]
}else{
  f_name <- sample_id
}

tile_info_df <- read.csv(
  paste0(cancer_data_dir, f_name, "/ft_model/", sample_id, "_TILE_TUMOR_PERC.csv")
) %>%
  mutate(TILE_XY_INDEXES = gsub("[()]", "", TILE_XY_INDEXES)) %>%
  separate(TILE_XY_INDEXES, into = c("x", "y"), sep = ", ") %>%
  mutate(across(c(x, y), as.integer))

#Load
cur_att_df <- att_df[which(att_df$SAMPLE_ID == sample_id),] 
cur_att_df <- min_max_norm(cur_att_df,"mean_att")
cur_att_df <- merge(cur_att_df, tile_info_df, by = c("SAMPLE_ID","pred_map_location"), all.y = TRUE)
cur_att_df <- cur_att_df[,c("SAMPLE_ID","mean_att","pred_map_location","x","y")]

#--- Merge on tile indices
comb_df <- merge(df, tile_info_df, by = c("x", "y"), all.y = TRUE)
comb_df <- merge(comb_df, cur_att_df, by = c("x", "y","SAMPLE_ID"), all.y = TRUE)

#plot
comb_df_list[[ct]] <- comb_df
}

final_comb <- do.call(rbind, comb_df_list)

df <- final_comb
#Group compare
library(dplyr)
library(ggplot2)
library(ggpubr)


# Median splits 
tumor_thr <- mean(df$TUMOR_PIXEL_PERC, na.rm = TRUE)
att_thr   <- mean(df$mean_att, na.rm = TRUE)

df <- df %>%
  mutate(
    tumor_group = ifelse(TUMOR_PIXEL_PERC >= tumor_thr, "High Tumor", "Low Tumor"),
    att_group   = ifelse(mean_att >= att_thr, "High Attention", "Low Attention"),
    quadrant    = paste(tumor_group, att_group, sep = " & ")
  )

table(df$quadrant)

# list of feature 
feature_list <- c(
  "MHCI",
  "MHCII",
  "Coactivation_molecules",
  "Effector_cells",
  "T_cells",
  "T_cell_traffic",
  "NK_cells",
  "B_cells",
  "M1_signatures",
  "Th1_signature",
  "Antitumor_cytokines",
  "Checkpoint_inhibition",
  "Macrophage_DC_traffic",
  "T_reg_traffic",
  "Treg",
  "Th2_signature",
  "Macrophages",
  "Neutrophil_signature",
  "Granulocyte_traffic",
  "MDSC_traffic",
  "MDSC",
  "Protumor_cytokines",
  "Proliferation_rate",
  "EMT_signature",
  "Matrix"
)

# pairwise comparisons to show on plot
comparisons <- list(
  c("High Tumor & High Attention", "High Tumor & Low Attention"),
  c("High Tumor & High Attention", "Low Tumor & High Attention"),
  c("Low Tumor & Low Attention",  "High Tumor & Low Attention"),
  c("Low Tumor & Low Attention",  "Low Tumor & High Attention")
)

# directory to save PNGs (create if needed)
dir.create("feature_quadrant_plots", showWarnings = FALSE)

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
    filename = file.path("feature_quadrant_plots",
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
          "feature_quadrant_means.csv",
          row.names = FALSE)

write.csv(feature_stats,
          "feature_quadrant_stats.csv",
          row.names = FALSE)

write.csv(top5_features,
          "top5_features_by_diff.csv",
          row.names = FALSE)
