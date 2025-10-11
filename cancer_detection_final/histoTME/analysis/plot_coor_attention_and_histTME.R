library(ggplot2)
library(dplyr)
library(tidyr)
library(pheatmap)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
data_dir <- paste0(proj_dir, "pred_out_100125/MSI/FOLD0/predictions_0.0/attention/")
pred_dir <- paste0(proj_dir, "pred_out_100125/MSI/FOLD0/predictions_0.0/")
tme_dir   <- paste0(proj_dir, "0_HistoTME/TME/TF0.0/")
label_dir <- paste0(proj_dir, "3C_labels_train_test/")

#Load MSI risk prediction
pred_df <- read.csv(paste0(pred_dir, "after_finetune_prediction.csv"))

#Load WSI histoTME
tme_df_opx <- read.csv(paste0(tme_dir,"OPX_predictions_uni2.csv"))
colnames(tme_df_opx)[which(colnames(tme_df_opx) == "ID")] <- "SAMPLE_ID"
tme_df_opx['COHORT'] <- "OPX"
tme_df_tcga <- read.csv(paste0(tme_dir,"TCGA_PRAD_predictions_uni2.csv"))
colnames(tme_df_tcga)[which(colnames(tme_df_tcga) == "ID")] <- "SAMPLE_ID"
tme_df_tcga['COHORT'] <- "TCGA_PRAD"
tme_df_nep <- read.csv(paste0(tme_dir,"Neptune_predictions_uni2.csv"))
colnames(tme_df_nep)[which(colnames(tme_df_nep) == "ID")] <- "SAMPLE_ID"
tme_df_nep['COHORT'] <- "Neptune"
tme_df <- do.call(rbind, list(tme_df_opx,tme_df_tcga, tme_df_nep))

#Load label
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




#load attention
all_attnetion <- list()
for (i in 1:length(ids)){
  df <- read.csv(paste0(data_dir, ids[i], "_att.csv"))
  coords <- do.call(rbind, strsplit(gsub("[()]", "", df$pred_map_location), ",\\s*"))
  coords <- apply(coords, 2, as.numeric)
  colnames(coords) <- c("x1","x2","y1","y2")
  df <- cbind(df, coords)
  df['SAMPLE_ID'] <- ids[i]
  all_attnetion[[i]] <- df
  
}

attnetion_df <- do.call(rbind, all_attnetion)


#Merge 
comb_df <- merge(attnetion_df,label_df, by = "SAMPLE_ID", all = TRUE)


################################################################################################################
#TME vs MSI + and MSI- (predicted)
################################################################################################################
cohort <- "Neptune"
selected_label <- "Pred_Class_adj" #Pred_Class_adj, MSI_POS, actual outcome
corrected_falg <- "Incorrect" #Incorrect, Correct


tme_label_pred_comb_df <- merge(tme_df,label_df, by = "SAMPLE_ID")
tme_label_pred_comb_df <- merge(tme_label_pred_comb_df,pred_df, by = "SAMPLE_ID")



tme_label_pred_comb_df <- tme_label_pred_comb_df[which(tme_label_pred_comb_df$COHORT == cohort),]
if (corrected_falg == "Correct"){
  idxes <- which(tme_label_pred_comb_df[,"Pred_Class_adj"] == tme_label_pred_comb_df[,"MSI_POS"])
  tme_label_pred_comb_df <- tme_label_pred_comb_df[idxes,]
}else if (corrected_falg == "Incorrect"){
  idxes <- which(tme_label_pred_comb_df[,"Pred_Class_adj"] != tme_label_pred_comb_df[,"MSI_POS"])
  tme_label_pred_comb_df <- tme_label_pred_comb_df[idxes,]
}
prediction_n <- nrow(tme_label_pred_comb_df)

# Extract label vector and ensure it's a factor
labels <- as.factor(tme_label_pred_comb_df[,selected_label])

# Build feature matrix (numeric only)
feature_df <- tme_label_pred_comb_df[ ,colnames(tme_df)[2:30]]

# make sure all are numeric
feature_df[] <- lapply(feature_df, function(x) as.numeric(as.character(x)))
feature_mat <- as.matrix(t(feature_df))  # rows = features, cols = samples for heatmap

# ---- Order samples: label 0 first, then label 1 ----
ord <- order(labels)               # will place 0s before 1s
feature_mat <- feature_mat[, ord]  # reorder columns
labels_ord <- labels[ord]

# ---- Annotation for column labels ----
annotation_col <- data.frame(Label = labels_ord)
rownames(annotation_col) <- colnames(feature_mat)
ann_colors <- list(Label = c("0" = "darkgreen", "1" = "salmon"))

# (Optional) colors + a vertical gap between label groups
gap_pos <- sum(as.character(labels_ord) == "0")  # compare as character


p <- pheatmap(
  feature_mat,
  scale = "row",
  cluster_rows = FALSE,
  cluster_cols = TRUE,        # <-- THIS is the key fflag
  treeheight_row = 0,    # hides row dendrogram
  treeheight_col = 0,    # hides column dendrogram
  annotation_col = annotation_col,
  annotation_colors = ann_colors,
  gaps_col = gap_pos,          # vertical divider between 0 and 1
  show_colnames = FALSE,
  color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
  main = paste0(cohort, " " ,corrected_falg, " prediction N:", prediction_n)
)
ggsave(
  filename = paste0(pred_dir, cohort, corrected_falg, "histTME_prediction_heatmap.png"),
  plot = p,
  width = 10, height = 8, dpi = 300
)

# ################################################################################################################
# #correlation between predicted MSI and features
# ################################################################################################################
# tme_label_pred_comb_df <- merge(tme_df,label_df, by = "SAMPLE_ID")
# tme_label_pred_comb_df <- merge(tme_label_pred_comb_df,pred_df, by = "SAMPLE_ID")
# 
# selected_label <- "Pred_Class_adj" #Pred_Class_adj, MSI_POS, actual outcome
# corrected_idxes <- which(tme_label_pred_comb_df[,"Pred_Class_adj"] == tme_label_pred_comb_df[,"MSI_POS"])
# tme_label_pred_comb_df <- tme_label_pred_comb_df[corrected_idxes,]
# 
# 
# #study_df
# rownames(tme_label_pred_comb_df) <- tme_label_pred_comb_df[,"SAMPLE_ID"]
# tme_label_pred_comb_df <- tme_label_pred_comb_df[,c("adj_prob_1",colnames(tme_df)[2:30])]
# 
# 
# library(ggplot2)
# 
# # ---- 1️⃣ Compute correlation and p-values vs. Predicted Risk ----
# target_col <- "adj_prob_1"
# 
# numeric_cols <- sapply(tme_label_pred_comb_df, is.numeric)
# num_df <- tme_label_pred_comb_df[, numeric_cols]
# 
# features <- setdiff(names(num_df), target_col)
# 
# # Compute correlation + p-values
# cor_list <- lapply(features, function(f) {
#   test <- suppressWarnings(
#     cor.test(num_df[[f]], num_df[[target_col]],
#              method = "spearman", use = "pairwise.complete.obs")
#   )
#   data.frame(
#     Feature = f,
#     Correlation = test$estimate,
#     P_value = test$p.value
#   )
# })
# 
# cor_df <- do.call(rbind, cor_list)
# 
# # ---- 2️⃣ Adjust p-values and mark significance ----
# cor_df$FDR <- p.adjust(cor_df$P_value, method = "BH")
# cor_df$Significance <- cut(
#   cor_df$FDR,
#   breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
#   labels = c("***", "**", "*", "")
# )
# 
# # Sort by correlation
# cor_df <- cor_df[order(-abs(cor_df$Correlation)), ]
# 
# # ---- 3️⃣ Plot ----
# p <- ggplot(cor_df, aes(x = reorder(Feature, Correlation), y = Correlation)) +
#   geom_bar(stat = "identity", aes(fill = Correlation > 0)) +
#   geom_text(aes(label = Significance), vjust = -0.3, size = 5) +
#   scale_fill_manual(values = c("TRUE" = "firebrick3", "FALSE" = "steelblue")) +
#   coord_flip() +
#   theme_minimal(base_size = 14) +
#   labs(
#     title = "Correlation with Predicted Risk",
#     x = "Feature",
#     y = "Correlation coefficient"
#   ) +
#   theme(legend.position = "none")
# 
# ggsave(
#   filename = paste0(pred_dir, "ALL", corrected_falg, "histTME_prediction_correlation.png"),
#   plot = p,
#   width = 10, height = 8, dpi = 300
# )
