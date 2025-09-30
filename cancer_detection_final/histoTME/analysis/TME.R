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

# ------------------ PARAMS ------------------
label_var   <- "MSI_POS"          # <— AR,HR1,HR2,PTEN,RB1, TP53,TMB_HIGHorINTERMEDITATE,MSI_POS
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
label_dir <- paste0(proj_dir, "3C_labels_train_test/", "OPX", "/TFT0.9/")
label_dir2 <- paste0(proj_dir, "3C_labels_train_test/", "TCGA_PRAD", "/TFT0.9/")
label_dir3 <- paste0(proj_dir, "3C_labels_train_test/", "Neptune", "/TFT0.9/")

tme_dir   <- paste0(proj_dir, "0_HistoTME/TME/TF0.9/")

# ------------------ Load ------------------
tmd_df_opx   <- read.csv(paste0(tme_dir, "OPX", "_predictions_uni2.csv"), check.names = FALSE)
colnames(tmd_df_opx)[1] <- "SAMPLE_ID"
label_df_opx <- read.csv(paste0(label_dir, "train_test_split.csv"), check.names = FALSE)
comb_df_opx <- merge(tmd_df_opx, label_df_opx, by = "SAMPLE_ID")

tmd_df_tcga   <- read.csv(paste0(tme_dir, "TCGA_PRAD", "_predictions_uni2.csv"), check.names = FALSE)
colnames(tmd_df_tcga)[1] <- "SAMPLE_ID"
label_df_tcga <- read.csv(paste0(label_dir2, "train_test_split.csv"), check.names = FALSE)
comb_df_tcga <- merge(tmd_df_tcga, label_df_tcga, by = "SAMPLE_ID")

tmd_df_nep  <- read.csv(paste0(tme_dir, "Neptune", "_predictions_uni2.csv"), check.names = FALSE)
colnames(tmd_df_nep)[1] <- "SAMPLE_ID"
label_df_nep <- read.csv(paste0(label_dir3, "train_test_split.csv"), check.names = FALSE)
comb_df_nep <- merge(tmd_df_nep, label_df_nep, by = "SAMPLE_ID")

comb_df <- do.call(rbind,list(comb_df_opx,comb_df_tcga,comb_df_nep))


feature_cols <- colnames(tmd_df_opx)[-1]
# ------------------ Clean metadata ------------------
# Keep only one row per SAMPLE_ID (in case of accidental dups)
comb_df <- comb_df %>% distinct(SAMPLE_ID, .keep_all = TRUE)

# ------------------ Column filtering (but keep the chosen label!) ------------------
comb_df <- comb_df[, (names(comb_df) %in% c(feature_cols,label_var,"SAMPLE_ID"))]



# sanity check
stopifnot(all(c("SAMPLE_ID", label_var) %in% names(comb_df)))

# ------------------ Prepare label vector ------------------
lab_raw <- comb_df[[label_var]]

# Coerce various possibilities to a clean factor:
# - logical -> "0"/"1"
# - numeric 0/1 -> "0"/"1"
# - character/factor -> as is (trim whitespace)
if (is.logical(lab_raw)) {
  lab_chr <- ifelse(lab_raw, "1", "0")
} else if (is.numeric(lab_raw)) {
  # keep as "0"/"1" if it is binary; otherwise just as character
  uniq_num <- sort(unique(na.omit(lab_raw)))
  if (length(uniq_num) <= 2 && all(uniq_num %in% c(0,1))) {
    lab_chr <- as.character(as.integer(lab_raw))
  } else {
    lab_chr <- as.character(lab_raw)
  }
} else {
  lab_chr <- trimws(as.character(lab_raw))
}

# build meta
meta <- comb_df %>% transmute(SAMPLE_ID, !!label_var := lab_chr)

# ------------------ Build numeric matrix (exclude the label feature) ------------------
features <- comb_df %>%
  select(SAMPLE_ID, where(is.numeric), -all_of(label_var))   # <— drop label_var even if numeric

stopifnot(!anyDuplicated(features$SAMPLE_ID))
rownames(features) <- features$SAMPLE_ID
features$SAMPLE_ID <- NULL
mat <- t(as.matrix(features))

# ------------------ Align labels to matrix columns ------------------
lab_vec <- meta[[label_var]][match(colnames(mat), meta$SAMPLE_ID)]
keep <- !is.na(lab_vec)
mat  <- mat[, keep, drop = FALSE]
lab_vec <- lab_vec[keep]

# Factor with stable order: if binary 0/1, keep c("0","1"); else alphabetical
uniq_levels <- sort(unique(lab_vec))
if (setequal(uniq_levels, c("0","1"))) {
  lab <- factor(lab_vec, levels = c("0","1"))
} else {
  lab <- factor(lab_vec, levels = uniq_levels)
}

# # Optional: cluster within groups but keep group split visually
# ord <- order(lab)
# mat <- mat[, ord, drop = FALSE]
# lab <- lab[ord]



# ------------------ Colors ------------------
# Quantile-based color function for matrix values
rng <- quantile(mat, probs = c(0.02, 0.5, 0.98), na.rm = TRUE)
col_fun <- colorRamp2(c(rng[1], rng[2], rng[3]), c("#3B4CC0","white","#B40426"))

# Label colors: special-case binary 0/1; otherwise auto-generate distinct colors
if (nlevels(lab) == 2 && all(levels(lab) == c("0","1"))) {
  label_cols_vec <- c("0" = "darkturquoise", "1" = "darkorange1")
} else {
  label_cols_vec <- setNames(
    circlize::rand_color(nlevels(lab)),
    levels(lab)
  )
}

ha_col <- HeatmapAnnotation(
  label = lab,
  col = list(label = label_cols_vec),
  annotation_legend_param = list(label = list(title = label_var))
)





# ------------------ Plot ------------------
ht <- Heatmap(
  mat,
  name = "score",
  col = col_fun,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  show_row_names = TRUE,
  show_column_names = FALSE,
  column_split = lab,
  top_annotation = ha_col,
  use_raster = FALSE
)
draw(ht)

# ------------------ Save ------------------
out_pdf <- paste0("immune_landscape_heatmap_", label_var, ".pdf")
pdf(out_pdf, width = 12, height = 6)
draw(ht)
dev.off()
message("Saved: ", out_pdf)
