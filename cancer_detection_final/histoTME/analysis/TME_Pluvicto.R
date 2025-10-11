library(openxlsx)
library(stringr)
library(dplyr)
library(ggplot2)
library(ggpubr)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) BiocManager::install("ComplexHeatmap")

library(tidyverse)
library(ComplexHeatmap)
library(circlize)

#################### Dir ####################
cohort <- "Pluvicto_TMA_Cores"
label_var <- "Responder"
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/"
tme_dir   <- paste0(proj_dir, "intermediate_data/0_HistoTME/TME/TF0.0/")
label_dir <- paste0(proj_dir, "data/MutationCalls/",cohort, "/")

#################### Load ####################
tme_df   <- read.csv(paste0(tme_dir, cohort, "_predictions_uni2.csv"), check.names = FALSE)
label_df <- read.xlsx(paste0(label_dir, "TMA_sample_mapping.xlsx"))
colnames(label_df)[which(colnames(label_df) == "Sample")] <- "ID"
label_df <- label_df[which(label_df[,"Group"] == "pluvicto"),]
comb_df  <- merge(tme_df, label_df, by = "ID")

#################### Plot ####################
# Columns of interest
cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
          "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
          "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
          "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
          "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
          "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
          "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
          "CAF", "Angiogenesis")


# ---- First dataset (sh_df) ----
data_matrix1 <- as.matrix(comb_df[, cols])
t_matrix1 <- t(scale(data_matrix1))

# Run clustering once (don't draw yet)
ph1 <- pheatmap(t_matrix1,
                color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
                cluster_rows = TRUE,
                cluster_cols = TRUE,
                treeheight_row = 0,
                treeheight_col = 0,
                show_rownames = TRUE,
                show_colnames = TRUE,
                fontsize_row = 8,
                fontsize_col = 8,
                border_color = NA,
                silent = FALSE)   # prevents plotting immediately
ph1





###########2nd way:
# Extract label vector and coerce to factor
labels <- as.factor(comb_df[, label_var])

# Build feature matrix (numeric only)
feature_df <- comb_df[, cols]
feature_df[] <- lapply(feature_df, function(x) as.numeric(as.character(x)))
feature_mat <- as.matrix(t(feature_df))  # rows = features, cols = samples

# ---- Map labels to readable names (adjust mapping if needed) ----
# Assumes "1" = Responder, "0" = Non-responder; tweak if your meaning differs
label_names <- ifelse(as.character(labels) %in% c("1", "Responder"),
                      "Responder", "Non-responder")
label_names <- factor(label_names, levels = c("Non-responder", "Responder"))

# ---- Order: Non-responder first, then Responder ----
ord <- order(label_names)
feature_mat  <- feature_mat[, ord, drop = FALSE]
labels_ord   <- label_names[ord]

# ---- Annotation for columns ----
annotation_col <- data.frame(Label = labels_ord)
rownames(annotation_col) <- colnames(feature_mat)

ann_colors <- list(Label = c("Responder" = "darkgreen",
                             "Non-responder" = "salmon"))

# ---- Add a vertical gap between groups & keep columns unclustered ----
gap_pos <- sum(labels_ord == "Non-responder")

p <- pheatmap(
  feature_mat,
  scale = "row",
  cluster_rows = TRUE,
  cluster_cols = TRUE,        # <-- keep the label-based separation
  treeheight_row = 0,
  treeheight_col = 0,
  annotation_col = annotation_col,
  annotation_colors = ann_colors,
  #gaps_col = gap_pos,          # vertical divider between groups
  show_colnames = FALSE,
  color = colorRampPalette(c("navy", "white", "firebrick3"))(50)
)

p




#3rd way
library(ComplexHeatmap)
library(circlize)

# feature_mat and label_names from above
col_split <- label_names  # factor with desired order

Heatmap(
  feature_mat,
  name = "z",
  color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
  cluster_rows = TRUE,
  cluster_columns = TRUE,       # clusters within each split
  column_split = col_split,     # <-- separates by label
  show_column_names = FALSE,
  top_annotation = HeatmapAnnotation(
    Label = col_split,
    col = list(Label = c("Responder" = "darkgreen",
                         "Non-responder" = "salmon"))
  ),
  column_title = NULL
)

